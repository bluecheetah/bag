# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module implements bag's interface with spectre simulator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Sequence, Optional, List, Tuple, Union, Mapping, Set

import re
import shutil
import time
from pathlib import Path

from pybag.enum import DesignOutput
from pybag.core import get_cdba_name_bits

from ..math import float_to_si_string
from ..io.file import read_yaml, open_file, is_valid_file, read_file
from ..io.string import wrap_string
from ..util.immutable import ImmutableList
from .data import (
    MDSweepInfo, SimData, SetSweepInfo, SweepLinear, SweepLog, SweepList, SimNetlistInfo,
    SweepSpec, MonteCarlo, AnalysisInfo, AnalysisAC, AnalysisSP, AnalysisNoise, AnalysisTran,
    AnalysisSweep1D, AnalysisPSS, AnalysisPNoise
)
from .base import SimProcessManager, get_corner_temp
from .hdf5 import load_sim_data_hdf5, save_sim_data_hdf5

# The use of pysrr to parse simulation data requires an external Python package.
# Since this import is only required when pysrr is used, it is treated as an optional import
try:
    from .srr import srr_to_sim_data
except ModuleNotFoundError as e:
    srr_to_sim_data = None  # assume pysrr is not needed, error later if assumption is false
    srr_import_err = e
else:
    srr_import_err = None

if TYPE_CHECKING:
    from .data import SweepInfo

reserve_params = {'freq', 'time'}


class SpectreInterface(SimProcessManager):
    """This class handles interaction with Spectre simulators.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary. Contains the following options:

        env_file : str
            the yaml path for PVT corners.
        use_pysrr : bool
            True to use pysrr. Defaults to False.
        compress : bool
            True to compress simulation data when saving to HDF5 file. Defaults to True.
        rtol: float
            relative tolerance for checking if 2 simulation values are the same. Defaults to 1e-8.
        atol: float
            absolute tolerance for checking if 2 simulation values are the same. Defaults to 1e-22.
        kwargs : Dict[str, Any]
            additional spectre simulation arguments. Contains the following options:

            command : str
                the command to launch simulator. Defaults to spectre.
            env : Optional[Dict[str, str]]
                an optional dictionary of environment variables.  None to inherit from parent. Defaults to None.
            run_64 : bool
                True to run in 64-bit mode. Defaults to True.
            format : str
                the output raw data file format. Defaults to psfxl.
            psfversion : str
                the version of psfxl to use. If not specified, defaults to the simulator's default psfversion.
            options : List[str]
                the command line simulator options. Defaults to an empty list.
    """

    def __init__(self, tmp_dir: str, sim_config: Dict[str, Any]) -> None:
        SimProcessManager.__init__(self, tmp_dir, sim_config)
        self._model_setup: Dict[str, List[Tuple[str, str]]] = read_yaml(sim_config['env_file'])
        self._use_pysrr: bool = sim_config.get('use_pysrr', False)
        self._sim_kwargs: Dict[str, Any] = sim_config.get('kwargs', {})
        self._out_fmt: str = self._sim_kwargs.get('format', 'psfxl')
        self._psf_version: str = self._sim_kwargs.get('psfversion', '')
        if self._out_fmt != 'psfxl':  # clear psf_version since it's only used for psfxl
            self._psf_version = ''

    @property
    def netlist_type(self) -> DesignOutput:
        return DesignOutput.SPECTRE

    def create_netlist(self, output_path: Path, sch_netlist: Path, info: SimNetlistInfo,
                       precision: int = 6) -> None:
        output_path_str = str(output_path.resolve())
        sch_netlist_path_str = str(sch_netlist.resolve())
        if ('<' in output_path_str or '>' in output_path_str or
                '<' in sch_netlist_path_str or '>' in sch_netlist_path_str):
            raise ValueError('spectre does not support directory names with angle brackets.')

        sim_envs = info.sim_envs
        analyses = info.analyses
        params = info.params
        env_params = info.env_params
        swp_info = info.swp_info
        monte_carlo = info.monte_carlo
        sim_options = info.options
        init_voltages = info.init_voltages
        if monte_carlo is not None and (isinstance(swp_info, SweepInfo) or len(sim_envs) > 1):
            raise NotImplementedError('Monte Carlo simulation not implemented for parameter sweep '
                                      'and/or process sweep')

        with open_file(sch_netlist, 'r') as f:
            lines = [l.rstrip() for l in f]

        # write simulator options
        if self._psf_version and 'psfversion' not in sim_options:
            sim_options = sim_options.copy(append=dict(psfversion=self._psf_version))
        if sim_options:
            sim_opt_list = ['simulatorOptions', 'options']
            for opt, val in sim_options.items():
                sim_opt_list.append(f'{opt}={val}')
            sim_opt_str = wrap_string(sim_opt_list)
            lines.append(sim_opt_str)

        # write parameters
        param_fmt = 'parameters {}={}'
        param_set = reserve_params.copy()
        for par, val in swp_info.default_items():
            if par not in param_set:
                lines.append(param_fmt.format(par, _format_val(val, precision)))
                param_set.add(par)
        for par, val_list in env_params.items():
            if par in param_set:
                raise ValueError('Cannot set a sweep parameter as environment parameter.')
            lines.append(param_fmt.format(par, _format_val(val_list[0], precision)))
            param_set.add(par)
        for par, val in params.items():
            if par not in param_set:
                lines.append(param_fmt.format(par, _format_val(val, precision)))
                param_set.add(par)
        for ana in analyses:
            par = ana.param
            if par and par not in param_set:
                lines.append(param_fmt.format(par, _format_val(ana.param_start, precision)))
                param_set.add(par)

        lines.append('')

        if isinstance(swp_info, SetSweepInfo):
            # write paramset declaration if needed
            _write_param_set(lines, swp_info.params, swp_info.values, precision)
            lines.append('')

        if init_voltages:
            # write initial conditions
            ic_line = 'ic'
            for key, val in init_voltages.items():
                key = get_cdba_name_bits(key, DesignOutput.SPECTRE)
                if len(key) > 1:
                    raise ValueError("Separate initial bus into multiple values")
                key = key[0]
                ic_line += f' {key}={_format_val(val, precision)}'

            lines.append(ic_line)
            lines.append('')
            has_ic = True
        else:
            has_ic = False

        # write statements for each simulation environment
        # write default model statements
        for idx, sim_env in enumerate(sim_envs):
            corner, temp = get_corner_temp(sim_env)
            if idx != 0:
                # start altergroup statement
                lines.append(f'{sim_env} altergroup {{')
            _write_sim_env(lines, self._model_setup[corner], temp)
            if idx != 0:
                # write environment parameters for second sim_env and on
                for par, val_list in env_params.items():
                    lines.append(param_fmt.format(par, val_list[idx]))
                # close altergroup statement
                lines.append('}')
            lines.append('')

            # write sweep statements
            num_brackets = _write_sweep_start(lines, swp_info, idx, precision)

            # write Monte Carlo statements if present
            if isinstance(monte_carlo, MonteCarlo):
                num_brackets += _write_monte_carlo(lines, monte_carlo)

            if num_brackets > 0:
                lines.append('')

            # write analyses
            save_outputs = set()
            jitter_event = []
            for ana in analyses:
                jitter_event = _write_analysis(lines, sim_env, ana, precision, has_ic)
                lines.append('')
                for output in ana.save_outputs:
                    try:
                        save_outputs.update(get_cdba_name_bits(output, DesignOutput.SPECTRE))
                    except ValueError:
                        save_outputs.update([output])

            # close sweep statements
            for _ in range(num_brackets):
                lines.append('}')
            if num_brackets > 0:
                lines.append('')

            # jitterevent is not an analysis and has to be written outside sweep analysis (message from Spectre)
            lines += jitter_event
            lines.append('')

            # write save statements
            _write_save_statements(lines, save_outputs)

        with open_file(output_path, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n')

    def get_sim_file(self, dir_path: Path, sim_tag: str) -> Path:
        return dir_path / f'{sim_tag}.hdf5'

    def load_sim_data(self, dir_path: Path, sim_tag: str) -> SimData:
        hdf5_path = self.get_sim_file(dir_path, sim_tag)
        import time
        print('Reading HDF5')
        start = time.time()
        ans = load_sim_data_hdf5(hdf5_path)
        stop = time.time()
        print(f'HDF5 read took {stop - start:.4g} seconds.')
        return ans

    async def async_run_simulation(self, netlist: Path, sim_tag: str) -> None:
        netlist = netlist.resolve()
        if not netlist.is_file():
            raise FileNotFoundError(f'netlist {netlist} is not a file.')

        sim_kwargs: Dict[str, Any] = self._sim_kwargs
        compress: bool = self.config.get('compress', True)
        rtol: float = self.config.get('rtol', 1e-8)
        atol: float = self.config.get('atol', 1e-22)

        cmd_str: str = sim_kwargs.get('command', 'spectre')
        env: Optional[Dict[str, str]] = sim_kwargs.get('env', None)
        run_64: bool = sim_kwargs.get('run_64', True)
        options = sim_kwargs.get('options', [])

        sim_cmd = [cmd_str, '-cols', '100', '-colslog', '100',
                   '-format', self._out_fmt, '-raw', f'{sim_tag}.raw']

        if run_64:
            sim_cmd.append('-64')
        for opt in options:
            sim_cmd.append(opt)

        sim_cmd.append(str(netlist))

        cwd_path = netlist.parent.resolve()
        log_path = cwd_path / 'spectre_output.log'
        raw_path: Path = cwd_path / f'{sim_tag}.raw'
        hdf5_path: Path = cwd_path / f'{sim_tag}.hdf5'

        try:
            if raw_path.is_dir():
                shutil.rmtree(str(raw_path))
        except FileNotFoundError:  # Ignore errors from race conditions
            pass

        if hdf5_path.is_file():
            hdf5_path.unlink()

        ret_code = await self.manager.async_new_subprocess(sim_cmd, str(log_path),
                                                           env=env, cwd=str(cwd_path))
        if ret_code is None or ret_code != 0:
            raise ValueError(f'Spectre simulation ended with error.  See log file: {log_path}')

        # Check if raw_path is created (as a directory). Give some slack for IO latency
        iter_cnt = 0
        while not raw_path.is_dir():
            if iter_cnt > 120:
                raise ValueError(f'Spectre simulation ended with error.  See log file: {log_path}')
            time.sleep(1)
            iter_cnt += 1

        if not is_valid_file(log_path, 'spectre completes with', 120, 1):
            raise ValueError(f'Spectre simulation ended with error.  See log file: {log_path}')

        log_contents = read_file(log_path)

        if 'spectre completes with 0 errors' not in log_contents:
            raise ValueError(f'Spectre simulation ended with error.  See log file: {log_path}')

        # check if Monte Carlo sim
        for fname in raw_path.iterdir():
            if str(fname).endswith('Distributed'):
                analysis_info: Path = fname / 'Analysis.info'
                with open_file(analysis_info, 'r') as f:
                    line = f.readline()
                num_proc = int(re.search(r'(.*) (\d*)\n', line).group(2))

                raw_sep: Path = raw_path / f'{num_proc}'
                for fname_sep in raw_sep.iterdir():
                    if str(fname_sep).endswith('.mapping'):
                        # Monte Carlo sim in multiprocessing mode
                        mapping_lines = []
                        for i in range(num_proc):
                            with open_file(raw_path / f'{i + 1}' / fname_sep.name, 'r') as fr:
                                for line_in in fr:
                                    mapping_lines.append(line_in)

                        await self._format_monte_carlo(mapping_lines, cwd_path, compress, rtol,
                                                       atol, hdf5_path)
                        return

            elif str(fname).endswith('.mapping'):
                # Monte Carlo sim in single processing mode
                mapping_lines = open_file(fname, 'r').readlines()
                await self._format_monte_carlo(mapping_lines, cwd_path, compress, rtol, atol,
                                               hdf5_path)
                return

        # convert to HDF5
        log_path = cwd_path / 'srr_to_hdf5.log'
        await self._srr_to_hdf5(compress, rtol, atol, raw_path, hdf5_path, log_path, cwd_path)

    async def _srr_to_hdf5(self, compress: bool, rtol: float, atol: float, raw_path: Path,
                           hdf5_path: Path, log_path: Path, cwd_path: Path) -> None:
        comp_str = '1' if compress else '0'
        rtol_str = f'{rtol:.4g}'
        atol_str = f'{atol:.4g}'

        if self._use_pysrr:
            if srr_to_sim_data is None:  # re-raise error from loading pysrr
                raise srr_import_err
            sim_data = srr_to_sim_data(raw_path, rtol, atol)
            save_sim_data_hdf5(sim_data, hdf5_path, compress)
        else:
            sim_cmd = ['srr_to_hdf5', str(raw_path), str(hdf5_path), comp_str, rtol_str, atol_str]
            ret_code = await self.manager.async_new_subprocess(sim_cmd, str(log_path),
                                                               cwd=str(cwd_path))
            if ret_code is None or ret_code != 0:
                raise ValueError(f'srr_to_hdf5 ended with error.  See log file: {log_path}')
            if not is_valid_file(hdf5_path, None, 120, 1):
                raise ValueError(f'srr_to_hdf5 ended with error.  See log file: {log_path}')

        # post-process HDF5 to convert to MD array
        _process_hdf5(hdf5_path, rtol, atol)

    async def _format_monte_carlo(self, lines: List[str], cwd_path: Path, compress: bool,
                                  rtol: float, atol: float, final_hdf5_path: Path) -> None:
        # read mapping file and convert each sub-directory into hdf5 files
        sim_data_list = []
        for line in lines:
            reg = re.search(r'(\d*)\t(.*)\n', line)
            idx, raw_str = reg.group(1), reg.group(2)
            raw_path: Path = cwd_path / raw_str
            hdf5_path: Path = cwd_path / f'{raw_path.name}.hdf5'
            log_path: Path = cwd_path / f'{raw_path.name}_srr_to_hdf5.log'
            await self._srr_to_hdf5(compress, rtol, atol, raw_path, hdf5_path, log_path,
                                    cwd_path)
            sim_data_list.append(load_sim_data_hdf5(hdf5_path))

        # combine all SimData to one SimData
        new_sim_data = SimData.combine(sim_data_list, 'monte_carlo')
        save_sim_data_hdf5(new_sim_data, final_hdf5_path, compress)


def _write_sim_env(lines: List[str], models: List[Tuple[str, str]], temp: int) -> None:
    for fname, section in models:
        if section:
            lines.append(f'include "{fname}" section={section}')
        else:
            lines.append(f'include "{fname}"')
    lines.append(f'tempOption options temp={temp}')


def _write_param_set(lines: List[str], params: Sequence[str],
                     values: Sequence[ImmutableList[float]], precision: int) -> None:
    # get list of lists of strings to print, and compute column widths
    data = [params]
    col_widths = [len(par) for par in params]
    for combo in values:
        str_list = []
        for idx, val in enumerate(combo):
            cur_str = _format_val(val, precision)
            col_widths[idx] = max(col_widths[idx], len(cur_str))
            str_list.append(cur_str)
        data.append(str_list)

    # write the columns
    lines.append('swp_data paramset {')
    for row in data:
        lines.append(' '.join(val.ljust(width) for val, width in zip(row, col_widths)))
    lines.append('}')


def _get_sweep_str(par: str, swp_spec: Optional[SweepSpec], precision: int) -> str:
    if not par or swp_spec is None:
        return ''

    if isinstance(swp_spec, SweepList):
        val_list = swp_spec.values
        # abstol check
        num_small = 0
        for val in val_list:
            if abs(val) < 3.0e-16:
                num_small += 1
        if num_small > 1:
            raise ValueError('sweep values are below spectre abstol, try to find a work around')

        tmp = ' '.join((_format_val(val, precision) for val in val_list))
        val_str = f'values=[{tmp}]'
    elif isinstance(swp_spec, SweepLinear):
        # spectre: stop is inclusive, lin = number of points excluding the last point
        val_str = f'start={swp_spec.start} stop={swp_spec.stop_inc} lin={swp_spec.num - 1}'
    elif isinstance(swp_spec, SweepLog):
        # spectre: stop is inclusive, log = number of points excluding the last point
        val_str = f'start={swp_spec.start} stop={swp_spec.stop_inc} log={swp_spec.num - 1}'
    else:
        raise ValueError('Unknown sweep specification.')

    if par in reserve_params:
        return val_str
    else:
        return f'param={par} {val_str}'


def _get_options_str(options: Mapping[str, str]) -> str:
    return ' '.join((f'{key}={val}' for key, val in options.items()))


def _write_sweep_start(lines: List[str], swp_info: SweepInfo, swp_idx: int, precision: int) -> int:
    if isinstance(swp_info, MDSweepInfo):
        for dim_idx, (par, swp_spec) in enumerate(swp_info.params):
            statement = _get_sweep_str(par, swp_spec, precision)
            lines.append(f'swp{swp_idx}{dim_idx} sweep {statement} {{')
        return swp_info.ndim
    else:
        lines.append(f'swp{swp_idx} sweep paramset=swp_data {{')
        return 1


def _write_monte_carlo(lines: List[str], mc: MonteCarlo) -> int:
    cur_line = f'__{mc.name}__ montecarlo numruns={mc.numruns} seed={mc.seed}'
    options_dict = dict(savefamilyplots='yes', appendsd='yes', savedatainseparatedir='yes',
                        donominal='yes', variations='all')
    options_dict.update(mc.options)
    opt_str = _get_options_str(options_dict)
    if opt_str:
        cur_line += ' '
        cur_line += opt_str
    cur_line += ' {'
    lines.append(cur_line)
    return 1


def _write_analysis(lines: List[str], sim_env: str, ana: AnalysisInfo, precision: int,
                    has_ic: bool) -> List[str]:
    cur_line = f'__{ana.name}__{sim_env}__'
    if hasattr(ana, 'p_port') and ana.p_port:
        cur_line += f' {ana.p_port}'
    if hasattr(ana, 'n_port') and ana.n_port:
        cur_line += f' {ana.n_port}'
    cur_line += f' {ana.name}'

    if isinstance(ana, AnalysisTran):
        cur_line += (f' start={_format_val(ana.start, precision)}'
                     f' stop={_format_val(ana.stop, precision)}')
        if isinstance(ana.out_start, str) or ana.out_start > 0:
            val_str = _format_val(ana.out_start, precision)
            cur_line += f' outputstart={val_str} strobestart={val_str}'
        if ana.strobe != 0:
            cur_line += f' strobeperiod={_format_val(ana.strobe)}'
        if has_ic:
            cur_line += ' ic=node'
    elif isinstance(ana, AnalysisSweep1D):
        par = ana.param
        sweep_str = _get_sweep_str(par, ana.sweep, precision)
        cur_line += ' '
        cur_line += sweep_str
        if isinstance(ana, AnalysisAC) and par != 'freq':
            cur_line += f' freq={float_to_si_string(ana.freq, precision)}'

        if isinstance(ana, AnalysisSP):
            cur_line += f' ports=[{" ".join(ana.ports)}] paramtype={ana.param_type.name.lower()}'
        elif isinstance(ana, AnalysisNoise):
            if ana.out_probe:
                cur_line += f' oprobe={ana.out_probe}'
            elif hasattr(ana, 'measurement') and ana.measurement:
                meas_list = [f'pm{idx}' for idx in range(len(ana.measurement))]
                cur_line += f' measurement=[{" ".join(meas_list)}]'
            elif not (hasattr(ana, 'p_port') and ana.p_port):
                raise ValueError('Either specify out_probe, or specify p_port and n_port, or specify measurement.')
            if ana.in_probe:
                cur_line += f' iprobe={ana.in_probe}'
    elif isinstance(ana, AnalysisPSS):
        if ana.period == 0.0 and ana.fund == 0.0 and ana.autofund is False:
            raise ValueError('For PSS simulation, either specify period or fund, '
                             'or set autofund = True')
        if ana.period > 0.0:
            cur_line += f' period={ana.period}'
        if ana.fund > 0.0:
            cur_line += f' fund={ana.fund}'
        if ana.autofund:
            cur_line += f' autofund=yes'
        if ana.strobe != 0:
            cur_line += f' strobeperiod={_format_val(ana.strobe)}'
    else:
        raise ValueError('Unknown analysis specification.')

    opt_str = _get_options_str(ana.options)
    if opt_str:
        cur_line += ' '
        cur_line += opt_str

    if ana.save_outputs:
        cur_line += ' save=selected'

    lines.append(cur_line)

    jitter_event = []
    if isinstance(ana, AnalysisPNoise):
        if ana.measurement:
            for idx, event in enumerate(ana.measurement):
                cur_line = f'pm{idx} jitterevent trigger=[{event.trig_p} {event.trig_n}] ' \
                           f'triggerthresh={event.triggerthresh} triggernum={event.triggernum} ' \
                           f'triggerdir={event.triggerdir} target=[{event.targ_p} {event.targ_n}]'
                jitter_event.append(cur_line)
    return jitter_event


def _write_save_statements(lines: List[str], save_outputs: Set[str]):
    for save_out in sorted(save_outputs):
        lines.append(f'save {save_out}')
    lines.append('')


def _format_val(val: Union[float, str], precision: int = 6) -> str:
    if isinstance(val, str):
        return val
    else:
        return float_to_si_string(val, precision)


def _process_hdf5(path: Path, rtol: float, atol: float) -> None:
    proc = 'process'
    sim_data = load_sim_data_hdf5(path)
    modified = False
    for grp in sim_data.group_list:
        sim_data.open_group(grp)
        if proc in sim_data.sweep_params:
            modified |= sim_data.remove_sweep(proc, rtol=rtol, atol=atol)

    if modified:
        save_sim_data_hdf5(sim_data, path)
