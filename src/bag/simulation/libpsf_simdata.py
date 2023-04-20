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
from typing import Mapping, Any, Dict, Tuple, Union, Optional, Sequence
from pathlib import Path
from libpsf import PSFDataSet
import numpy as np

from pybag.enum import DesignOutput

from .data import AnalysisData, SimData, _check_is_md, combine_ana_sim_envs
from .nutbin import parse_sweep_info


class LibPSFParser:
    def __init__(self, raw_path: Path, rtol: float, atol: float, num_proc: int = 1) -> None:
        self._cwd_path = raw_path.parent
        self._num_proc = num_proc
        lp_data = self.parse_raw_folder(raw_path)
        self._sim_data = self.convert_to_sim_data(lp_data, rtol, atol)

    @property
    def sim_data(self) -> SimData:
        return self._sim_data

    def parse_raw_folder(self, raw_path: Path) -> Mapping[str, Any]:
        ana_dict = {}
        if self._num_proc == 1:
            self._parse_raw_folder_helper(raw_path, ana_dict)
        else:
            # multi-processing mode
            for pidx in range(self._num_proc):
                self._parse_raw_folder_helper(raw_path / f'{pidx + 1}', ana_dict)
        return ana_dict

    def _parse_raw_folder_helper(self, raw_path: Path, ana_dict: Dict[str, Any]) -> None:
        for fname in raw_path.iterdir():
            # some files have multiple suffixes, so split name at first '.' to get entire suffix
            suf = fname.name.split('.', 1)[-1]
            if suf not in ['logFile', 'sweep', 'pac']:
                info = self.get_info_from_fname(fname.name)
                data, inner_sweep = self.parse_raw_file(fname)
                info['inner_sweep'] = inner_sweep
                self.populate_dict(ana_dict, info, raw_path, data)

    @staticmethod
    def parse_raw_file(fname: Path) -> Tuple[Dict[str, Union[np.ndarray, float]], str]:
        ds = PSFDataSet(str(fname))
        data = {}

        # get inner sweep, if any
        if ds.is_swept():
            swp_vars = ds.get_sweep_param_names()
            if len(swp_vars) == 1:
                inner_sweep = swp_vars[0]
                data[inner_sweep] = ds.get_sweep_values()
            else:
                # Only innermost sweep is in this file.
                raise NotImplementedError
        else:
            inner_sweep = ''

        # get signals
        for sig_name in ds.get_signal_names():
            sig = ds.get_signal(sig_name)
            if isinstance(sig, float):
                data[sig_name] = sig
            elif isinstance(sig[0], dict):
                # PNoise: separate out the noise components
                pn_dict = {}
                for _sig in sig:
                    for bkey, val in _sig.items():
                        key = f'{sig_name}.{bkey.decode("ascii")}'
                        if key in pn_dict:
                            pn_dict[key].append(val)
                        else:
                            pn_dict[key] = [val]
                for key, val in pn_dict.items():
                    data[key] = np.array(val)
            else:
                data[sig_name] = sig

        return data, inner_sweep

    @staticmethod
    def get_info_from_fname(fname: str) -> Dict[str, Any]:
        # get ana_name from fname
        ana_name, suf = fname.split('.', 1)

        # get ana_type and sim_env from ana_name
        ana_type_fmt = '[a-zA-Z]+'
        sim_env_fmt = '[a-zA-Z0-9]+_[a-zA-Z0-9]+'
        m = re.search(f'({ana_type_fmt})__({sim_env_fmt})', ana_name)
        ana_type = m.group(1)

        # For PSS, edit ana_type based on inner sweep
        if ana_type == 'pss':
            if suf == 'fd.pss':
                ana_type = 'pss_fd'
            elif suf == 'td.pss':
                ana_type = 'pss_td'
            else:
                raise NotImplementedError

        # For PAC, get harmonic number
        if ana_type == 'pac':
            harmonic = int(suf.split('.')[0])
        else:
            harmonic = None

        # get outer sweep information from ana_name, if any
        m_swp = re.findall('swp[0-9]{2}', ana_name)
        m_swp1 = re.findall('swp[0-9]{2}-[0-9]{3}', ana_name)
        key_list = []
        for idx, val in enumerate(m_swp1):
            key_list.append(val[-3:])

        return dict(
            ana_name=ana_name,
            sim_env=m.group(2),
            ana_type=ana_type,
            swp_info=m_swp,
            swp_key='_'.join(key_list),
            harmonic=harmonic,
        )

    def populate_dict(self, ana_dict: Dict[str, Any], info: Mapping[str, Any], raw_path: Path,
                      data: Dict[str, Union[np.ndarray, float]]) -> None:
        ana_type: str = info['ana_type']
        sim_env: str = info['sim_env']
        swp_key: str = info['swp_key']
        swp_info: Sequence[str] = info['swp_info']
        harmonic: Optional[int] = info['harmonic']
        inner_sweep: str = info['inner_sweep']

        if ana_type not in ana_dict:
            ana_dict[ana_type] = {}

        if sim_env not in ana_dict[ana_type]:
            # get outer sweep, if any
            swp_vars, swp_data = parse_sweep_info(swp_info, raw_path, f'___{ana_type}__{sim_env}__',
                                                  offset=16)
            ana_dict[ana_type][sim_env] = {
                'inner_sweep': inner_sweep,
                'outer_sweep': swp_key is not '',
                'harmonics': harmonic is not None,
                'swp_vars': swp_vars,
                'swp_data': swp_data,
            }
            if swp_key or (harmonic is not None):
                ana_dict[ana_type][sim_env]['data'] = {}
        elif self._num_proc > 1:
            # multi-processing mode, need to parse more sweep files
            swp_vars, swp_data = parse_sweep_info(swp_info, raw_path, f'___{ana_type}__{sim_env}__',
                                                  offset=16)
            for _key, _val in swp_data.items():
                _val_ini = ana_dict[ana_type][sim_env]['swp_data'][_key]
                ana_dict[ana_type][sim_env]['swp_data'][_key] = np.concatenate((_val_ini, _val))

        # PAC harmonics are handled separately from parametric sweep
        if swp_key:
            if harmonic is not None:
                if swp_key not in ana_dict[ana_type][sim_env]['data']:
                    ana_dict[ana_type][sim_env]['data'][swp_key] = {}
                ana_dict[ana_type][sim_env]['data'][swp_key][harmonic] = data
            else:
                ana_dict[ana_type][sim_env]['data'][swp_key] = data
        else:
            if harmonic is not None:
                ana_dict[ana_type][sim_env]['data'][harmonic] = data
            else:
                ana_dict[ana_type][sim_env]['data'] = data

    def convert_to_sim_data(self, lp_data, rtol: float, atol: float) -> SimData:
        ana_dict = {}
        sim_envs = []
        for ana_type, sim_env_dict in lp_data.items():
            sim_envs = sorted(sim_env_dict.keys())
            sub_ana_dict = {}
            for sim_env, lp_dict in sim_env_dict.items():
                sub_ana_dict[sim_env] = self.convert_to_analysis_data(lp_dict, rtol, atol)
            ana_dict[ana_type] = combine_ana_sim_envs(sub_ana_dict, sim_envs)
        return SimData(sim_envs, ana_dict, DesignOutput.SPECTRE)

    def convert_to_analysis_data(self, lp_dict: Mapping[str, Any], rtol: float, atol: float) -> AnalysisData:
        data = {}

        # get sweep information
        inner_sweep: str = lp_dict['inner_sweep']
        outer_sweep: bool = lp_dict['outer_sweep']
        harmonics: bool = lp_dict['harmonics']
        swp_vars = lp_dict['swp_vars']
        swp_data = lp_dict['swp_data']
        lp_data = lp_dict['data']
        if outer_sweep:
            swp_combos = []
            swp_keys = sorted(lp_data.keys())
            for key in swp_keys:
                swp_combo = []
                for _vridx, _vlidx in enumerate(key.split('_')):
                    swp_combo.append(swp_data[swp_vars[_vridx]][int(_vlidx)])
                swp_combos.append(swp_combo)
            num_swp = len(swp_combos[0])

            if harmonics:
                swp_vars.append('harmonic')
                new_swp_combos = []
                harm_swp = sorted(lp_data[swp_keys[0]].keys())
                for swp_combo in swp_combos:
                    for _harm in harm_swp:
                        new_swp_combos.append(swp_combo + [_harm])
                swp_combos = new_swp_combos
                num_swp += 1
            else:
                harm_swp = []

            swp_len = len(swp_combos)
            swp_combo_list = [np.array(swp_combos)[:, i] for i in range(num_swp)]
        else:
            swp_keys = []
            if harmonics:
                swp_vars = ['harmonic']
                harm_swp = sorted(lp_data.keys())
                swp_len = len(harm_swp)
                swp_combo_list = [np.array(harm_swp)]
            else:
                harm_swp = []
                swp_len = 0
                swp_combo_list = []

        swp_shape, swp_vals = _check_is_md(1, swp_combo_list, rtol, atol, None)  # single corner per set
        is_md = swp_shape is not None
        if is_md:
            swp_combo = {var: swp_vals[i] for i, var in enumerate(swp_vars)}
        else:
            raise NotImplementedError("Parametric sweeps must be formatted multi-dimensionally")
        data.update(swp_combo)

        # parse each signal
        if swp_len == 0:    # no outer sweep
            for sig_name, sig_y in lp_data.items():
                if isinstance(sig_y, float):
                    data_shape = swp_shape
                else:
                    data_shape = (*swp_shape, sig_y.shape[-1])
                _new_sig = sig_name.replace('/', '.')
                data[_new_sig] = sig_y if _new_sig == inner_sweep else np.reshape(sig_y, data_shape)
        else:   # combine outer sweeps
            if outer_sweep:
                if harmonics:
                    sig_names = lp_data[swp_keys[0]][0].keys()
                else:
                    sig_names = lp_data[swp_keys[0]].keys()
            else:
                if harmonics:
                    sig_names = lp_data[0].keys()
                else:
                    raise NotImplementedError('Not possible.')
            for sig_name in sig_names:
                yvecs = []
                if outer_sweep:
                    if harmonics:
                        yvecs = [lp_data[key0][key1][sig_name] for key1 in harm_swp for key0 in swp_keys]
                    else:
                        yvecs = [lp_data[key][sig_name] for key in swp_keys]
                else:
                    if harmonics:
                        yvecs = [lp_data[key][sig_name] for key in harm_swp]
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
