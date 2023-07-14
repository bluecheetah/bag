# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
from typing import Mapping, BinaryIO, Any, Dict, Sequence, Union, Tuple
from pathlib import Path
import numpy as np

from pybag.enum import DesignOutput

from .data import AnalysisData, SimData, _check_is_md, combine_ana_sim_envs


class NutBinParser:
    def __init__(self, raw_path: Path, rtol: float, atol: float, monte_carlo: bool) -> None:
        self._cwd_path = raw_path.parent
        self._monte_carlo = monte_carlo
        nb_data = self.parse_raw_file(raw_path)
        self._sim_data = self.convert_to_sim_data(nb_data, rtol, atol)

    @property
    def sim_data(self) -> SimData:
        return self._sim_data

    @property
    def monte_carlo(self) -> bool:
        return self._monte_carlo

    def parse_raw_file(self, raw_path: Path) -> Mapping[str, Any]:
        with open(raw_path, 'rb') as f:
            f.readline()    # skip title
            f.readline()    # skip date

            # read all the individual analyses
            ana_dict = {}
            while True:
                # read Plotname or EOF
                plotname = f.readline().decode('ascii')
                if len(plotname) == 0:  # EOF
                    break
                data = self.parse_analysis(f)
                self.populate_dict(ana_dict, plotname, raw_path, data)
        return ana_dict

    @staticmethod
    def parse_analysis(f: BinaryIO) -> Dict[str, Union[np.ndarray, float]]:
        # read flags
        flags = f.readline().decode('ascii').split()
        if flags[1] == 'real':
            nptype = float
        elif flags[1] == 'complex':
            nptype = complex
        else:
            raise ValueError(f'Flag type {flags[1]} is not recognized.')

        # read number of variables and points
        num_vars = int(f.readline().decode('ascii').split()[-1])
        # TODO: hack for an example case where '\n' goes missing after the next line
        points_line = f.readline().decode('ascii').split()
        if points_line[-1].isdigit():
            num_points = int(points_line[-1])
            next_line = None
        else:
            num_points = int(re.split('(\d+)', points_line[-4])[1])
            next_line = points_line[-3:]
            next_line.insert(0, 'Variables:')

        # get the variable names, ignore units and other flags
        var_names = []
        for idx in range(num_vars):
            if idx == 0 and next_line:
                _line = next_line
            else:
                _line = f.readline().decode('ascii').split()
            if idx == 0:
                var_names.append(_line[2])
            else:
                var_names.append(_line[1])

        f.readline()    # skip "Binary:"
        # read big endian binary data
        bin_data = np.fromfile(f, dtype=np.dtype(nptype).newbyteorder(">"), count=num_vars * num_points)
        data = {}
        if 'dummy' in var_names and num_points == 1:
            # single sim with no inner sweep
            inner_sweep = False
        else:
            inner_sweep = True
        for idx, var_name in enumerate(var_names):
            if var_name == 'dummy':
                # skip dummy variable
                continue
            if var_name == 'freq':
                # convert frequency data to real
                _sig = np.real(bin_data[idx::num_vars])
            else:
                _sig = bin_data[idx::num_vars]
            if not inner_sweep:
                # no inner sweep, store singular value
                _sig = _sig[0]
            data[var_name] = _sig

        return data

    @staticmethod
    def get_info_from_plotname(plotname: str) -> Mapping[str, Any]:
        # get ana_name from plotname
        ana_name = re.search('`.*\'', plotname).group(0)[1:-1]

        # get ana_type and sim_env from ana_name
        ana_type_fmt = '[a-zA-Z]+'
        sim_env_fmt = '[a-zA-Z0-9]+_[a-zA-Z0-9]+'
        m = re.search(f'({ana_type_fmt})__({sim_env_fmt})', ana_name)
        ana_type = m.group(1)

        # get inner sweep information from plotname, if any
        m_in = re.search(r': (\w+) =', plotname)
        if m_in is not None:
            inner_sweep = m_in.group(1)
        else:
            inner_sweep = ''

        # For PSS, edit ana_type based on inner sweep
        if ana_type == 'pss':
            if inner_sweep == 'fund':
                ana_type = 'pss_fd'
                inner_sweep = 'freq'
            elif inner_sweep == 'time':
                ana_type = 'pss_td'
            else:
                raise NotImplementedError(f'Unrecognized inner_sweep = {inner_sweep}')

        # get outer sweep information from ana_name, if any
        m_swp = re.findall('swp[0-9]{2}', ana_name)
        m_swp1 = re.findall('swp[0-9]{2}-[0-9]{3}', ana_name)
        swp_combo = []
        for val in m_swp1:
            swp_combo.append(int(val[-3:]))

        return dict(
            ana_name=ana_name,
            sim_env=m.group(2),
            ana_type=ana_type,
            swp_info=m_swp,
            swp_combo=swp_combo,
            inner_sweep=inner_sweep,
        )

    def populate_dict(self, ana_dict: Dict[str, Any], plotname: str, raw_path: Path,
                      data: Dict[str, Union[np.ndarray, float]]) -> None:
        # get analysis name and sim_env
        info = self.get_info_from_plotname(plotname)
        ana_name: str = info['ana_name']
        if self.monte_carlo and not ana_name.startswith('__mc_'):
            # ignore the nominal sim in Monte Carlo
            return
        ana_type: str = info['ana_type']
        sim_env: str = info['sim_env']
        swp_info: Sequence[str] = info['swp_info']
        swp_combo: Sequence[int] = info['swp_combo']
        inner_sweep: str = info['inner_sweep']

        if ana_type not in ana_dict:
            ana_dict[ana_type] = {}

        if sim_env not in ana_dict[ana_type]:
            # get outer sweep, if any
            swp_vars, swp_data = parse_sweep_info(swp_info, self._cwd_path / f'{raw_path.name}.psf', ana_type, sim_env,
                                                  offset=44)
            ana_dict[ana_type][sim_env] = {'data': [], 'swp_combos': [], 'inner_sweep': inner_sweep,
                                           'swp_vars': swp_vars, 'swp_data': swp_data, 'ana_name': []}

        if ana_name in ana_dict[ana_type][sim_env]['ana_name']:
            if ana_type == 'pss_td':
                # In PSS simulations with saveinit=yes, the only way to detect the difference between the initial
                # transient sim and the subsequent time-domain PSS is the order of the results in the raw file. Hence,
                # if we get the same ana_name, the previous result was pss_tran, and we are currently seeing pss_td.
                prev_ana_type = 'pss_tran'

                if prev_ana_type not in ana_dict:
                    ana_dict[prev_ana_type] = {}

                if sim_env not in ana_dict[prev_ana_type]:
                    # get outer sweep, if any
                    prev_swp_vars, prev_swp_data = parse_sweep_info(swp_info, self._cwd_path / f'{raw_path.name}.psf',
                                                                    prev_ana_type, sim_env, offset=44)
                    ana_dict[prev_ana_type][sim_env] = {'data': [], 'swp_combos': [], 'inner_sweep': inner_sweep,
                                                        'swp_vars': prev_swp_vars, 'swp_data': prev_swp_data,
                                                        'ana_name': []}

                ana_dict[prev_ana_type][sim_env]['data'].append(ana_dict[ana_type][sim_env]['data'].pop())
                ana_dict[prev_ana_type][sim_env]['ana_name'].append(ana_dict[ana_type][sim_env]['ana_name'].pop())
                if swp_combo:
                    ana_dict[prev_ana_type][sim_env]['swp_combos'].append(ana_dict[ana_type][sim_env]['swp_combos'].pop())
            else:
                raise NotImplementedError('This should not be possible; see developer.')

        ana_dict[ana_type][sim_env]['data'].append(data)
        ana_dict[ana_type][sim_env]['ana_name'].append(ana_name)
        # get outer sweep combo, if any
        if swp_combo:
            swp_combo_val = []
            swp_vars = ana_dict[ana_type][sim_env]['swp_vars']
            swp_data = ana_dict[ana_type][sim_env]['swp_data']
            for _vridx, _vlidx in enumerate(swp_combo):
                swp_combo_val.append(swp_data[swp_vars[_vridx]][_vlidx])
            ana_dict[ana_type][sim_env]['swp_combos'].append(swp_combo_val)

    def convert_to_sim_data(self, nb_data, rtol: float, atol: float) -> SimData:
        ana_dict = {}
        sim_envs = []
        for ana_type, sim_env_dict in nb_data.items():
            sim_envs = sorted(sim_env_dict.keys())
            sub_ana_dict = {}
            for sim_env, nb_dict in sim_env_dict.items():
                sub_ana_dict[sim_env] = self.convert_to_analysis_data(nb_dict, rtol, atol, ana_type)
            ana_dict[ana_type] = combine_ana_sim_envs(sub_ana_dict, sim_envs)
        return SimData(sim_envs, ana_dict, DesignOutput.SPECTRE)

    def convert_to_analysis_data(self, nb_dict: Mapping[str, Any], rtol: float, atol: float, ana_type: str
                                 ) -> AnalysisData:
        data = {}

        # get sweep information
        inner_sweep: str = nb_dict['inner_sweep']
        swp_combos = nb_dict['swp_combos']
        if swp_combos:  # create sweep combinations
            num_swp = len(swp_combos[0])
            swp_vars = nb_dict['swp_vars']

            # if PAC, configure harmonic sweep
            if ana_type == 'pac':
                swp_vars.append('harmonic')
                # When maxsideband > 0, spectre errors with this message:
                # "ERROR (SPECTRE-7012): Output for analysis of type `pac' is not supported in Nutmeg."
                harm_len = len(nb_dict['data']) // len(swp_combos)
                harmonics = harm_len // 2
                harm_swp = np.linspace(-harmonics, harmonics, harm_len, dtype=int)
                new_swp_combos = []
                for swp_combo in swp_combos:
                    for _harm in harm_swp:
                        new_swp_combos.append(swp_combo + [_harm])
                swp_combos = new_swp_combos
                num_swp += 1

            swp_len = len(swp_combos)
            swp_combo_list = [np.array(swp_combos)[:, i] for i in range(num_swp)]
        else:   # no outer sweep
            # if PAC, configure harmonic sweep
            if ana_type == 'pac':
                swp_vars = ['harmonic']
                # When maxsideband > 0, spectre errors with this message:
                # "ERROR (SPECTRE-7012): Output for analysis of type `pac' is not supported in Nutmeg."
                harm_len = swp_len = 1
                harmonics = harm_len // 2
                swp_combo_list = [np.linspace(-harmonics, harmonics, harm_len, dtype=int)]
            else:
                swp_vars = []
                swp_len = 0
                swp_combo_list = []

        # get Monte Carlo information (no parametric sweep)
        if self.monte_carlo:
            swp_vars.insert(0, 'monte_carlo')
            swp_len = len(nb_dict['data'])  # this works because only PAC harmonic may be present with maxsideband = 0
            if ana_type == 'pac':
                # This has PAC harmonics, since no parametric sweep is allowed with Monte Carlo
                harm_swp = swp_combo_list[0]
                swp_combos = []
                for mc_idx in range(swp_len):
                    for _harm in harm_swp:
                        swp_combos.append([mc_idx, _harm])
                swp_combo_list = [np.array(swp_combos)[:, i] for i in range(2)]
            else:
                swp_combo_list = [np.linspace(0, swp_len - 1, swp_len, dtype=int)]

        swp_shape, swp_vals = _check_is_md(1, swp_combo_list, rtol, atol, None)  # single corner per set
        is_md = swp_shape is not None
        if is_md:
            swp_combo = {var: swp_vals[i] for i, var in enumerate(swp_vars)}
        else:
            raise NotImplementedError("Parametric sweeps must be formatted multi-dimensionally")
        data.update(swp_combo)

        # parse each signal
        if swp_len == 0:    # no outer sweep
            for sig_name, sig_y in nb_dict['data'][0].items():
                if isinstance(sig_y, float):    # no inner sweep
                    data_shape = swp_shape
                else:
                    data_shape = (*swp_shape, sig_y.shape[-1])
                _new_sig = sig_name.replace('/', '.')
                data[_new_sig] = sig_y if _new_sig == inner_sweep else np.reshape(sig_y, data_shape)
        else:   # combine outer sweeps
            sig_names = list(nb_dict['data'][0].keys())
            for sig_name in sig_names:
                yvecs = [nb_dict['data'][i][sig_name] for i in range(swp_len)]
                if isinstance(yvecs[0], float):
                    sub_dims = ()   # not used
                    max_dim = 0     # not used
                    is_same_len = True
                    data_shape = swp_shape
                else:
                    sub_dims = tuple(yvec.shape[0] for yvec in yvecs)
                    max_dim = max(sub_dims)
                    is_same_len = all((sub_dims[i] == sub_dims[0] for i in range(swp_len)))
                    data_shape = (*swp_shape, max_dim)
                _new_sig = sig_name.replace('/', '.')
                if not is_same_len:
                    yvecs_padded = [np.pad(yvec, (0, max_dim - dim), constant_values=np.nan)
                                    for yvec, dim in zip(yvecs, sub_dims)]
                    sig_y = np.stack(yvecs_padded)
                    data[_new_sig] = np.reshape(sig_y, data_shape)
                else:
                    if _new_sig == inner_sweep:
                        same_inner = True
                        for _yvec in yvecs[1:]:
                            if not np.allclose(yvecs[0], _yvec, rtol=rtol, atol=atol):
                                same_inner = False
                                break
                        if same_inner:
                            data[_new_sig] = yvecs[0]
                        else:
                            # same length but unequal sweep values
                            sig_y = np.stack(yvecs)
                            data[_new_sig] = np.reshape(sig_y, data_shape)
                    else:
                        sig_y = np.stack(yvecs)
                        data[_new_sig] = np.reshape(sig_y, data_shape)

        if inner_sweep:
            swp_vars.append(inner_sweep)

        return AnalysisData(['corner'] + swp_vars, data, is_md)


def parse_sweep_info(swp_info: Sequence[str], raw_path: Path, ana_type: str, sim_env: str, offset: int
                     ) -> Tuple[Sequence[str], Mapping[str, np.ndarray]]:
    # read from innermost sweep outwards
    new_swp_info = list(swp_info)
    swp_vars = []
    swp_data = {}
    suf = get_sweep_file_suf(ana_type, sim_env)
    while len(new_swp_info) > 0:
        # assume nested sweeps are always consistent across outer sweeps, and read 0th sweep file only
        name = '-000_'.join(new_swp_info) + suf + '.sweep'
        file_path = raw_path / name
        if file_path.is_file():
            _swp_var, _swp_vals = parse_sweep_file(raw_path / name, offset)
            swp_vars.insert(0, _swp_var)
            swp_data[_swp_var] = _swp_vals
        else:
            # this is in multiprocessing mode, so sweep is already read in 0th fork
            swp_vars.insert(0, '')
        new_swp_info.pop()
    return swp_vars, swp_data


def parse_sweep_file(file_path: Path, offset: int) -> Tuple[str, np.ndarray]:
    with open(file_path, 'rb') as f:
        pattern = re.compile(r'([a-zA-Z0-9_]+)')
        while True:
            name_bin = f.readline()
            name_ascii = name_bin.decode('ascii', errors='ignore')
            name_find = re.findall(pattern, name_ascii)
            try:
                sd_idx = name_find.index('sweep_direction')
                # this line has "sweep_direction" and "grid"
                break
            except ValueError:
                continue

        # the ASCII string before "sweep_direction" is the sweep variable name
        swp_name = name_find[sd_idx - 1]

        # TODO: HACK - if swp_name has len == 1, it's not the actual swp_name; read the previous entry instead
        if len(swp_name) == 1:
            swp_name = name_find[sd_idx - 2]

        # find the location of the end of "grid" from the end
        val_idx = len(name_bin) - name_bin.find(b'grid') - 4
        val_idx -= offset  # extra offset to reach beginning of binary data
        f.seek(-val_idx, 1)
        bin_data = np.fromfile(f, dtype=np.dtype(float).newbyteorder(">"))

        swp_vals = []
        for val0, val1 in zip(bin_data[::2], bin_data[1::2]):
            # when val0 is the following float, val1 is a sweep parameter value
            if np.isclose(val0, 3.39519327e-313, atol=1e-321):
                swp_vals.append(val1)
            else:
                break
    return swp_name, np.array(swp_vals)


def get_sweep_file_suf(ana_type: str, sim_env: str) -> str:
    if ana_type.startswith('pss'):
        pss_type = ana_type.split('_')[-1]
        return f'___pss__{sim_env}___{pss_type}'
    return f'___{ana_type}__{sim_env}__'
