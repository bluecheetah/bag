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

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Mapping, Any, List, Tuple
from pathlib import Path

from ..concurrent.core import batch_async_task
from ..io.file import write_yaml, is_valid_file, write_file
from .base import EmSimProcessManager


class EMXInterface(EmSimProcessManager):
    """This class handles interaction with EMX.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Mapping[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir: str, sim_config: Mapping[str, Any]) -> None:
        EmSimProcessManager.__init__(self, tmp_dir, sim_config)

        # check for proc file
        self._proc_file: Path = Path(sim_config['proc_file']).resolve()
        if not self._proc_file.exists():
            raise Exception(f'Cannot find process file: {self._proc_file}')
        self._key: str = sim_config.get('key', '')
        self._parallel: int = sim_config.get('parallel', 1)
        self._simul_freq: int = sim_config.get('simul_freq', 0)

    def _get_model_path(self, root_path: Path, cell_name: str) -> Path:
        em_base_path = self._get_em_base_path(root_path)
        return em_base_path / cell_name

    def _set_em_option(self, cell_name: str, params: Mapping[str, Any], model_path: Path
                       ) -> Tuple[List[str], List[Path]]:
        em_options: Mapping[str, Any] = params['em_options']
        fmin: float = em_options['fmin']
        fmax: float = em_options['fmax']
        fstep: float = em_options['fstep']
        edge_mesh: float = em_options['edge_mesh']
        thickness: float = em_options['thickness']
        via_separation: float = em_options['via_separation']
        show_log: bool = em_options['show_log']
        show_cmd: bool = em_options['show_cmd']

        # mesh option
        mesh_opts = ['-e', f'{edge_mesh}', '-t', f'{thickness}', '-v', f'{via_separation}', '--3d=*']
        # freq option
        freq_opts = ['--sweep', f'{fmin}', f'{fmax}', '--sweep-stepsize', f'{fstep}']
        # print options
        pr_num = 3 if show_log else 0
        pr_opts = [f'--verbose={pr_num}']
        # print cmd options
        cmd_opts = ['--print-command-line', '-l', '0'] if show_cmd else []

        # get port list
        port_list: List[str] = params['port_list']
        gnd_list: List[str] = params['gnd_list']

        # port options: avoid changing the original list
        portlist_n = port_list.copy()
        gndlist_n = gnd_list.copy()
        # remove repeated ports
        for gnd in gndlist_n:
            if gnd in portlist_n:
                portlist_n.remove(gnd)
        port_string = []
        for idx, port in enumerate(portlist_n):
            # port_string.extend(['-p', f'P{idx:03d}=\'{port}\'', '-i', f'P{idx:03d}'])
            port_string.extend(['-p', f'P{idx:03d}={port}', '-i', f'P{idx:03d}'])
        n_ports = len(portlist_n)
        for idx, port in enumerate(gndlist_n):
            # port_string.extend(['-p', f'P{(idx + n_ports):03d}=\'{port}\''])
            port_string.extend(['-p', f'P{(idx + n_ports):03d}={port}'])

        # get s/y parameters and model
        model_path.mkdir(parents=True, exist_ok=True)
        # s parameter file
        sp_file = model_path / f'{cell_name}.s{n_ports}p'
        sp_opts = ['--format=touchstone', '-s', str(sp_file)]
        # y parameter file
        yp_file = model_path / f'{cell_name}.y{n_ports}p'
        yp_opts = ['--format=touchstone', '-y', str(yp_file)]
        # y matlab file
        ym_file = model_path / f'{cell_name}.y'
        ym_opts = ['--format=matlab', '-y', str(ym_file)]
        # pz model (removed as it slows down simulation)
        # state_file = self._model_path / f'{self._cell_name}.pz'
        # st_opts = ['--format=spectre', f'--model-file={state_file}', '--save-model-state']

        # log file
        log_file = model_path / f'{cell_name}.log'
        log_opts = [f'--log-file={log_file}']

        # other options
        other_opts = [f'--parallel={self._parallel}', f'--simultaneous-frequencies={self._simul_freq}',
                      '--max-memory=80%', '--quasistatic', '--dump-connectivity', '--via-sidewall=*',
                      '--via-inductance=*', '--cadence-pins=1']

        # get extra options
        extra_opts = []
        extra_options: Mapping[str, Any] = params.get('extra_options')
        if extra_options:
            for opt, value in extra_options.items():
                extra_opts.append(f'--{opt}={value}')

        # key
        if self._key:
            extra_opts.append(f'--key={self._key}')

        emx_opts = mesh_opts + freq_opts + port_string + pr_opts + cmd_opts + extra_opts + other_opts + sp_opts + \
                   yp_opts + ym_opts + log_opts
        return emx_opts, [sp_file, yp_file, ym_file, log_file]

    async def async_gen_nport(self, cell_name: str, gds_file: Path, params: Mapping[str, Any], root_path: Path,
                              run_sim: bool = False) -> Path:
        """
        Run EM sim to get nport for the current module.
        """
        # get paths and options
        model_path = self._get_model_path(root_path, cell_name)
        emx_opts, outfiles = self._set_em_option(cell_name, params, model_path)

        if run_sim:
            # delete log file if exist -- use it for error checking
            if is_valid_file(outfiles[-1], None, 1, 1):
                outfiles[-1].unlink()

            # get emx simulation working
            emx_cmd = [f'{os.environ["EMX_HOME"]}/bin/emx', str(gds_file.resolve()), cell_name, str(self._proc_file)]
            print("EMX simulation started.")
            start = time.time()
            em_base_path = self._get_em_base_path(root_path)
            bag_log = self.get_log_path(root_path)
            ret_code = await self.manager.async_new_subprocess(emx_cmd + emx_opts, cwd=str(em_base_path),
                                                               log=str(bag_log))

            # check whether ends correctly
            if ret_code is None or ret_code != 0 or not is_valid_file(outfiles[-1], None, 60, 1):
                raise Exception(f'EMX stops with error.\nLog file is in {outfiles[-1]}')
            else:
                period = (time.time() - start) / 60
                print(f'EMX simulation finished successfully.\nLog file is in {outfiles[-1]}')
                print(f'EMX simulation takes {period} minutes')
                write_file(bag_log, 'SUCCESS', append=True)

        return outfiles[0]

    @staticmethod
    def _set_mdl_option(cell_name: str, model_type: str, model_path: Path) -> Tuple[List[str], Path, List[Path]]:
        # model type
        type_opts = f'--type={model_type}'

        ym_file = model_path / f'{cell_name}.y'

        # scs model
        scs_model = model_path / f'{cell_name}.scs'
        scs_opts = f'--spectre-file={scs_model}'

        # spice
        sp_model = model_path / f'{cell_name}.sp'
        sp_opts = f'--spice-file={sp_model}'

        mdl_opts = [type_opts, str(ym_file), scs_opts, sp_opts]

        return mdl_opts, ym_file, [scs_model, sp_model]

    async def async_gen_model(self, cell_name: str, params: Mapping[str, Any], root_path: Path):
        # get options
        model_type: str = params['model_type']
        model_path = self._get_model_path(root_path, cell_name)
        mdl_opts, infile, outfiles = self._set_mdl_option(cell_name, model_type, model_path)

        # delete model file if exist -- use it for error checking
        for _file in outfiles:
            if _file.exists():
                _file.unlink()

        # emx command
        mdl_cmd = [f'{os.environ["EMX_HOME"]}/bin/modelgen'] + mdl_opts
        print("Model generation started.")
        start = time.time()
        em_base_path = self._get_em_base_path(root_path)
        ret_code = await self.manager.async_new_subprocess(mdl_cmd, cwd=str(em_base_path),
                                                           log=f'{em_base_path}/bag_modelgen.log')

        # check whether ends correctly
        files_created = True
        for _file in outfiles:
            files_created = files_created and is_valid_file(_file, None, 60, 1)
        if ret_code is None or ret_code != 0 or not files_created:
            raise Exception('Model generation stops with error.')
        else:
            period = (time.time() - start) / 60
            print('Model generation finished successfully.')
            print(f'Model generation takes {period} minutes')

    def run_simulation(self, cell_name: str, gds_file: Path, params: Mapping[str, Any], root_path: Path) -> None:
        coro = self.async_gen_nport(cell_name, gds_file, params, root_path, run_sim=True)
        batch_async_task([coro])
        if 'model_type' in params:
            coro = self.async_gen_model(cell_name, params, root_path)
            batch_async_task([coro])

    def process_output(self, cell_name: str, params: Mapping[str, Any], root_path: Path) -> None:
        # calculate inductance and quality factor
        center_tap = params.get('center_tap', False)
        print(f'center_tap is {center_tap}')
        model_path = self._get_model_path(root_path, cell_name)
        calculate_ind_q(model_path, cell_name, center_tap)


def calculate_ind_q(model_path: Path, cell_name: str, center_tap: bool) -> None:
    """
    Calculate inductance and Q for given y parameter file.
    """
    ym_file = model_path / f'{cell_name}.y'
    try:
        with open(ym_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'Y parameter file {ym_file} is not found')
    else:
        results = []
        num = len(lines)
        freq = np.zeros(num)
        ldiff = np.zeros(num)
        qdiff = np.zeros(num)
        zdiff = np.zeros(num, dtype=np.complex_)
        for i, line in enumerate(lines):
            if line[0] == '%':
                # Reading a comment line, so skip it.
                continue
            yparam = np.fromstring(line, sep=' ')
            # get frequency and real yparam
            freq[i] = yparam[0]
            yparam = yparam[1:]
            if center_tap:
                # get real part and imag part
                real_part = yparam[::2].reshape(3, 3)
                imag_part = yparam[1::2].reshape(3, 3)
                # get complex value
                y = real_part + imag_part * 1j
                # get z parameters
                zdiff0 = 2 * np.divide(y[1, 2] + y[0, 2],
                                       np.multiply(y[1, 2], (y[0, 0] - y[0, 1])) -
                                       np.multiply(y[0, 2], (y[1, 0] - y[1, 1])))
                # det_10 = y[0, 2] * y[2, 1] - y[0, 1] * y[2, 2]
                # det_01 = y[2, 0] * y[1, 2] - y[1, 0] * y[2, 2]
                # det_00 = y[1, 1] * y[2, 2] - y[2, 1] * y[1, 2]
                # det_11 = y[0, 0] * y[2, 2] - y[2, 0] * y[0, 2]
                # det_ = det_10 * det_01 - det_00 * det_11
                # zdiff0 = y[2, 2] * (det_10 + det_01 - det_00 - det_11) / det_ if det_ != 0 else 0.0
            else:
                # get real part and imag part
                real_part = yparam[::2].reshape(2, 2)
                imag_part = yparam[1::2].reshape(2, 2)
                # get complex value
                y = real_part + imag_part * 1j
                # get z parameters
                zdiff0 = np.divide(4, y[0, 0] + y[1, 1] - y[0, 1] - y[1, 0])
                # det_y = np.linalg.det(y)
                # zdiff0 = (y[0, 0] + y[0, 1] + y[1, 0] + y[1, 1]) / det_y if det_y != 0 else 0.0

            # z11 = np.divide(1, y[0, 0])
            # z22 = np.divide(1, y[1, 1])

            # get l and qs
            ldiff[i] = np.imag(zdiff0)/2/np.pi/freq[i] if freq[i] != 0 else 0.0
            qdiff[i] = np.imag(zdiff0) / np.real(zdiff0) if zdiff0 != 0 else 0.0
            zdiff[i] = zdiff0

            # add to list
            results.append(dict(freq=float(freq[i]), ldiff=float(ldiff[i]), qdiff=float(qdiff[i])))

        # store results
        result_yaml = model_path / f'{cell_name}.yaml'
        write_yaml(result_yaml, results)
        print(f'Results are in {result_yaml}.')

        # find max Q
        q_max_idx = np.argmax(qdiff)
        q_max = qdiff[q_max_idx]
        q_max_freq = freq[q_max_idx]
        print(f'Max Q is {q_max} at {q_max_freq / 1e9} GHz.')

        # find self resonant frequency
        srf_idx = np.argmax(np.abs(zdiff))
        srf_freq = freq[srf_idx]
        print(f'Self resonant frequency is {srf_freq / 1e9} GHz.')

        # plot results
        fig, (ax0, ax1) = plt.subplots(2)
        ax0.plot(freq / 1e9, ldiff * 1e12)
        ax0.set_xlabel('Frequency (GHz)')
        ax0.set_ylabel('Inductance (pH)')
        ax0.grid()
        ax1.plot(freq / 1e9, qdiff)
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Quality factor')
        ax1.grid()
        plt.tight_layout()
        plt.show()
