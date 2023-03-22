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

"""This module handles high level simulation routines.

This module defines SimAccess, which provides methods to run simulations
and retrieve results.
"""

from __future__ import annotations
from typing import (
    Tuple, Union, Iterable, List, Dict, Any, Optional, TypeVar, Type, Sequence, ItemsView, Mapping
)

import math
from enum import Enum
from dataclasses import dataclass

import numpy as np

from pybag.enum import DesignOutput
from pybag.core import convert_cdba_name_bit

from ..util.immutable import ImmutableList, ImmutableSortedDict


###############################################################################
# Sweep specifications
###############################################################################

class SweepSpecType(Enum):
    LIST = 0
    LINEAR = 1
    LOG = 2


class SweepInfoType(Enum):
    MD = 0
    SET = 1


@dataclass(eq=True, frozen=True)
class SweepList:
    values: ImmutableList[float]

    def __len__(self) -> int:
        return len(self.values)

    @property
    def start(self) -> float:
        return self.values[0]


@dataclass(eq=True, frozen=True)
class SweepLinear:
    """stop is inclusive"""
    start: float
    stop: float
    num: int
    endpoint: bool = True

    def __len__(self) -> int:
        return self.num

    @property
    def step(self) -> float:
        den = self.num - 1 if self.endpoint else self.num
        return (self.stop - self.start) / den

    @property
    def stop_inc(self) -> float:
        return self.stop if self.endpoint else self.start + (self.num - 1) * self.step

    @property
    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.num, self.endpoint)


@dataclass(eq=True, frozen=True)
class SweepLog:
    """stop is inclusive"""
    start: float
    stop: float
    num: int
    endpoint: bool = True

    def __len__(self) -> int:
        return self.num

    @property
    def start_log(self) -> float:
        return math.log10(self.start)

    @property
    def stop_log(self) -> float:
        return math.log10(self.stop)

    @property
    def step_log(self) -> float:
        den = self.num - 1 if self.endpoint else self.num
        return (self.stop_log - self.start_log) / den

    @property
    def stop_inc(self) -> float:
        if self.endpoint:
            return self.stop
        return 10.0 ** (self.start_log + (self.num - 1) * self.step_log)


SweepSpec = Union[SweepLinear, SweepLog, SweepList]


def swp_spec_from_dict(table: Mapping[str, Any]) -> SweepSpec:
    swp_type = SweepSpecType[table['type']]
    if swp_type is SweepSpecType.LIST:
        return SweepList(ImmutableList(table['values']))
    elif swp_type is SweepSpecType.LINEAR:
        return SweepLinear(table['start'], table['stop'], table['num'], table.get('endpoint', True))
    elif swp_type is SweepSpecType.LOG:
        return SweepLog(table['start'], table['stop'], table['num'], table.get('endpoint', True))
    else:
        raise ValueError(f'Unsupported sweep type: {swp_type}')


@dataclass(eq=True, frozen=True)
class MDSweepInfo:
    params: ImmutableList[Tuple[str, SweepSpec]]

    @property
    def ndim(self) -> int:
        return len(self.params)

    @property
    def stype(self) -> SweepInfoType:
        return SweepInfoType.MD

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple((len(val[1]) for val in self.params))

    def __contains__(self, item: str) -> bool:
        for name, _ in self.params:
            if name == item:
                return True
        return False

    def __iter__(self) -> Iterable[str]:
        return (item[0] for item in self.params)

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for name, spec in self.params:
            yield name, spec.start


@dataclass(eq=True, frozen=True)
class SetSweepInfo:
    params: ImmutableList[str]
    values: ImmutableList[ImmutableList[float]]

    @property
    def stype(self) -> SweepInfoType:
        return SweepInfoType.SET

    @property
    def shape(self) -> Tuple[int, ...]:
        # NOTE: one-element tuple, not typo
        return len(self.values),

    def __contains__(self, item: str) -> bool:
        return item in self.params

    def __iter__(self) -> Iterable[str]:
        return self.params

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for idx, name in enumerate(self.params):
            yield name, self.values[0][idx]


SweepInfo = Union[MDSweepInfo, SetSweepInfo]


def swp_info_from_struct(table: Union[Sequence[Tuple[str, Mapping[str, Any]]], Mapping[str, Any]]
                         ) -> SweepInfo:
    if isinstance(table, dict) or isinstance(table, ImmutableSortedDict):
        params = ImmutableList(table['params'])
        values = []
        num_par = len(params)
        for combo in table['values']:
            if len(combo) != num_par:
                raise ValueError('Invalid param set values.')
            values.append(ImmutableList(combo))

        return SetSweepInfo(params, ImmutableList(values))
    else:
        par_list = [(par, swp_spec_from_dict(spec)) for par, spec in table]
        return MDSweepInfo(ImmutableList(par_list))


###############################################################################
# Analyses
###############################################################################

class AnalysisType(Enum):
    DC = 0
    AC = 1
    TRAN = 2
    SP = 3
    NOISE = 4
    PSS = 5
    PAC = 6
    PNOISE = 7


class SPType(Enum):
    S = 0
    Y = 1
    Z = 2
    YZ = 3


T = TypeVar('T', bound='AnalysisSweep1D')


@dataclass(eq=True, frozen=True)
class AnalysisSweep1D:
    param: str
    sweep: Optional[SweepSpec]
    options: ImmutableSortedDict[str, str]
    save_outputs: ImmutableList[str]

    @classmethod
    def from_dict(cls: Type[T], table: Dict[str, Any], def_param: str = '') -> T:
        param = table.get('param', def_param)
        sweep = table.get('sweep', None)
        opt = table.get('options', {})
        out = table.get('save_outputs', [])
        if not param or sweep is None:
            param = ''
            swp = None
        else:
            swp = swp_spec_from_dict(sweep)

        return cls(param, swp, ImmutableSortedDict(opt), ImmutableList(out))

    @property
    def param_start(self) -> float:
        if self.param:
            return self.sweep.start
        return 0.0


@dataclass(eq=True, frozen=True)
class AnalysisDC(AnalysisSweep1D):
    @property
    def name(self) -> str:
        return 'dc'


@dataclass(eq=True, frozen=True)
class AnalysisAC(AnalysisSweep1D):
    freq: float

    @property
    def name(self) -> str:
        return 'ac'

    @classmethod
    def from_dict(cls: Type[T], table: Dict[str, Any], def_param: str = '') -> T:
        base = AnalysisSweep1D.from_dict(table, def_param='freq')
        if base.param != 'freq':
            freq_val = table['freq']
        else:
            freq_val = 0.0

        return cls(base.param, base.sweep, base.options, base.save_outputs, freq_val)


@dataclass(eq=True, frozen=True)
class AnalysisSP(AnalysisAC):
    ports: ImmutableList[str]
    param_type: SPType

    @property
    def name(self) -> str:
        return 'sp'


@dataclass(eq=True, frozen=True)
class AnalysisNoise(AnalysisAC):
    p_port: str
    n_port: str
    out_probe: str
    in_probe: str

    @property
    def name(self) -> str:
        return 'noise'


@dataclass(eq=True, frozen=True)
class AnalysisTran:
    start: float
    stop: float
    strobe: float
    out_start: float
    options: ImmutableSortedDict[str, str]
    save_outputs: ImmutableList[str]

    @property
    def param(self) -> str:
        return ''

    @property
    def param_start(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return 'tran'


@dataclass(eq=True, frozen=True)
class AnalysisPSS:
    p_port: str
    n_port: str
    period: float
    fund: float
    autofund: bool
    strobe: float
    options: ImmutableSortedDict[str, str]
    save_outputs: ImmutableList[str]

    @property
    def param(self) -> str:
        return ''

    @property
    def name(self) -> str:
        return 'pss'


@dataclass(eq=True, frozen=True)
class AnalysisPAC(AnalysisAC):

    @property
    def name(self) -> str:
        return 'pac'


@dataclass(eq=True, frozen=True)
class AnalysisPNoise(AnalysisNoise):
    measurement: Optional[ImmutableList[JitterEvent]] = None

    @property
    def name(self) -> str:
        return 'pnoise'


@dataclass(eq=True, frozen=True)
class JitterEvent:
    trig_p: str
    trig_n: str
    triggerthresh: float
    triggernum: int
    triggerdir: str
    targ_p: str
    targ_n: str


AnalysisInfo = Union[AnalysisDC, AnalysisAC, AnalysisSP, AnalysisNoise, AnalysisTran,
                     AnalysisPSS, AnalysisPAC, AnalysisPNoise]


def analysis_from_dict(table: Dict[str, Any]) -> AnalysisInfo:
    ana_type = AnalysisType[table['type']]
    if ana_type is AnalysisType.DC:
        return AnalysisDC.from_dict(table)
    elif ana_type is AnalysisType.AC:
        return AnalysisAC.from_dict(table)
    elif ana_type is AnalysisType.SP:
        base = AnalysisAC.from_dict(table)
        return AnalysisSP(base.param, base.sweep, base.options, base.save_outputs, base.freq,
                          ImmutableList(table['ports']), SPType[table['param_type']])
    elif ana_type is AnalysisType.NOISE:
        base = AnalysisAC.from_dict(table)
        return AnalysisNoise(base.param, base.sweep, base.options, base.save_outputs, base.freq,
                             table.get('p_port', ''), table.get('n_port', ''),
                             table.get('out_probe', ''), table.get('in_probe', ''))
    elif ana_type is AnalysisType.TRAN:
        return AnalysisTran(table.get('start', 0.0), table['stop'], table.get('strobe', 0.0),
                            table.get('out_start', -1.0),
                            ImmutableSortedDict(table.get('options', {})),
                            ImmutableList(table.get('save_outputs', [])))
    elif ana_type is AnalysisType.PSS:
        return AnalysisPSS(table.get('p_port', ''), table.get('n_port', ''),
                           table.get('period', 0.0), table.get('fund', 0.0),
                           table.get('autofund', False), table.get('strobe', 0.0),
                           ImmutableSortedDict(table.get('options', {})),
                           ImmutableList(table.get('save_outputs', [])))
    elif ana_type is AnalysisType.PAC:
        base = AnalysisAC.from_dict(table)
        return AnalysisPAC(base.param, base.sweep, base.options, base.save_outputs, base.freq)
    elif ana_type is AnalysisType.PNOISE:
        base = AnalysisAC.from_dict(table)
        pnoise_meas = table.get('measurement', None)
        if pnoise_meas:
            pnoise_meas = ImmutableList([JitterEvent(**_dict) for _dict in pnoise_meas])
        return AnalysisPNoise(base.param, base.sweep, base.options, base.save_outputs, base.freq,
                              table.get('p_port', ''), table.get('n_port', ''),
                              table.get('out_probe', ''), table.get('in_probe', ''), pnoise_meas)
    else:
        raise ValueError(f'Unknown analysis type: {ana_type}')


###############################################################################
# Simulation Netlist Info
###############################################################################

@dataclass(eq=True, frozen=True)
class MonteCarlo:
    numruns: int
    seed: int
    options: ImmutableSortedDict[str, Any]

    @property
    def name(self) -> str:
        return 'mc'


def monte_carlo_from_dict(mc_dict: Optional[Dict[str, Any]]) -> Optional[MonteCarlo]:
    if not mc_dict:
        return None

    numruns: int = mc_dict['numruns']
    seed: int = mc_dict['seed']
    options: Dict[str, Any] = mc_dict.get('options', {})

    return MonteCarlo(numruns, seed, options=ImmutableSortedDict(options))


@dataclass(eq=True, frozen=True)
class SimNetlistInfo:
    sim_envs: ImmutableList[str]
    analyses: ImmutableList[AnalysisInfo]
    params: ImmutableSortedDict[str, float]
    env_params: ImmutableSortedDict[str, ImmutableList[float]]
    swp_info: SweepInfo
    outputs: ImmutableSortedDict[str, str]
    options: ImmutableSortedDict[str, Any]
    monte_carlo: Optional[MonteCarlo]
    init_voltages: ImmutableSortedDict[str, Union[str, float]]

    @property
    def sweep_type(self) -> SweepInfoType:
        return self.swp_info.stype


def netlist_info_from_dict(table: Dict[str, Any]) -> SimNetlistInfo:
    sim_envs: List[str] = table['sim_envs']
    analyses: List[Dict[str, Any]] = table['analyses']
    params: Dict[str, float] = table.get('params', {})
    env_params: Dict[str, List[float]] = table.get('env_params', {})
    swp_info: Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]] = table.get('swp_info', [])
    outputs: Dict[str, str] = table.get('outputs', {})
    options: Dict[str, Any] = table.get('options', {})
    monte_carlo: Optional[Dict[str, Any]] = table.get('monte_carlo', None)
    init_voltages: Dict[str, Union[str, float]] = table.get('init_voltages', {})

    if not sim_envs:
        raise ValueError('simulation environments list is empty')

    env_par_dict = {}
    num_env = len(sim_envs)
    for key, val in env_params.items():
        if len(val) != num_env:
            raise ValueError("Invalid env_param value.")
        env_par_dict[key] = ImmutableList(val)

    ana_list = [analysis_from_dict(val) for val in analyses]

    return SimNetlistInfo(ImmutableList(sim_envs), ImmutableList(ana_list),
                          ImmutableSortedDict(params), ImmutableSortedDict(env_par_dict),
                          swp_info_from_struct(swp_info), ImmutableSortedDict(outputs),
                          ImmutableSortedDict(options), monte_carlo_from_dict(monte_carlo),
                          ImmutableSortedDict(init_voltages))


###############################################################################
# Simulation data classes
###############################################################################

class AnalysisData:
    """A data struct that stores simulation data from a single analysis"""

    def __init__(self, sweep_params: Sequence[str], data: Dict[str, np.ndarray],
                 is_md: bool) -> None:
        self._swp_pars = ImmutableList(sweep_params)
        self._data = data
        self._is_md = is_md
        swp_set = set(sweep_params)
        self._signals = [key for key in data.keys() if key not in swp_set]

    def __getitem__(self, item: str) -> np.ndarray:
        return self._data[item]

    def __contains__(self, item: str) -> bool:
        return item in self._data

    @property
    def data_shape(self) -> Tuple[int, ...]:
        if not self._signals:
            return ()
        return self._data[self._signals[0]].shape

    @property
    def is_md(self) -> bool:
        return self._is_md

    @property
    def sweep_params(self) -> ImmutableList[str]:
        return self._swp_pars

    @property
    def signals(self) -> List[str]:
        return self._signals

    @classmethod
    def combine(cls, data_list: Sequence[AnalysisData], swp_name: str,
                swp_vals: Optional[np.ndarray] = None, axis: int = 0) -> AnalysisData:
        ndata = len(data_list)
        if ndata < 1:
            raise ValueError('Must combine at least 1 data.')
        if swp_vals is None:
            swp_vals = np.arange(ndata)

        data0 = data_list[0]
        new_data = {}
        swp_par_list = list(data0.sweep_params)

        # get all signals
        max_size = None
        for sig in data0.signals:
            arr_list = [arr[sig] for arr in data_list]
            sizes = [x.shape for x in arr_list]
            max_size = np.max(list(zip(*sizes)), -1)
            cur_ans = np.full((len(arr_list),) + tuple(max_size), np.nan, dtype=arr_list[0].dtype)
            for idx, arr in enumerate(arr_list):
                # noinspection PyTypeChecker
                select = (idx,) + tuple(slice(0, s) for s in sizes[idx])
                cur_ans[select] = arr
            new_data[sig] = np.moveaxis(cur_ans, 0, axis)

        # get last sweep parameter
        last_par = swp_par_list[-1]
        last_xvec = data0[last_par]
        xvec_list = [data[last_par] for data in data_list]
        for xvec in xvec_list:
            if not np.array_equal(xvec_list[0], xvec):
                # last sweep parameter has to be a multi dimensional array
                sizes = [x.shape for x in xvec_list]
                cur_ans = np.full((len(xvec_list),) + tuple(max_size), np.nan)
                for idx, _xvec in enumerate(xvec_list):
                    # noinspection PyTypeChecker
                    select = (idx, ...) + tuple(slice(0, s) for s in sizes[idx])
                    cur_ans[select] = _xvec
                last_xvec = np.moveaxis(cur_ans, 0, axis)
                break
        new_data[last_par] = last_xvec

        # get all other sweep params
        for sn in swp_par_list[:-1]:
            if sn != 'corner':
                new_data[sn] = data0[sn]

        swp_par_list.insert(axis, swp_name)
        new_data[swp_name] = swp_vals

        return AnalysisData(swp_par_list, new_data, data0.is_md)

    def get_param_value(self, name: str) -> np.ndarray:
        param_idx = self._swp_pars.index(name)

        shape = self.data_shape[:-1]
        shape_init = [1] * len(shape)
        shape_init[param_idx] = shape[param_idx]
        arr = self._data[name].reshape(tuple(shape_init))
        return np.broadcast_to(arr, shape)

    def items(self) -> ItemsView[str, np.ndarray]:
        return self._data.items()

    def insert(self, name: str, data: np.ndarray) -> None:
        self._data[name] = data
        if name not in self._signals:
            self._signals.append(name)

    def copy(self) -> AnalysisData:
        _data = {}
        for k, v in self._data.items():
            _data[k] = self._data[k].copy()
        return AnalysisData(self._swp_pars, _data, self._is_md)

    """Adds combination to simulation results"""

    def add(self, new_data: Dict[str, np.ndarray]):
        if self.is_md:
            raise AttributeError('Currently only supported in is_md = False mode')

        # check that the size of new data is the same as existing data
        assert len(self._data.keys()) == len(new_data.keys())

        # check that all sweep parameters are provided
        for param in self.sweep_params:
            if param not in new_data.keys():
                raise ValueError('Param %s not provided in data' % param)

        ref_length = len(list(new_data.values())[0])
        # add data points
        for name, arr in new_data.items():
            # check that all new data arrays are the correct length
            if name in self.sweep_params or name == 'hash':
                assert len(arr) == ref_length
            else:
                assert len(arr[0]) == ref_length

            # new sweep point
            if name in self.sweep_params:
                self._data[name] = np.append(self._data[name], arr)
            # sweep data
            else:
                self._data[name] = np.hstack((self._data[name], arr))

    def remove_sweep(self, name: str, rtol: float = 1e-8, atol: float = 1e-20) -> bool:
        new_swp_vars = list(self._swp_pars)
        try:
            idx = new_swp_vars.index(name)
        except ValueError:
            return False

        if self._is_md:
            swp_vals = self._data.pop(name)
            if swp_vals.size != 1:
                self._data[name] = swp_vals
                raise ValueError('Can only remove sweep with 1 value in a MD sweep.')

            for sig in self._signals:
                self._data[sig] = np.squeeze(self._data[sig], axis=idx)

            last_var_name = self._swp_pars[-1]
            last_var_arr = self._data[last_var_name]
            if len(last_var_arr.shape) != 1:
                # also need to squeeze last x axis values
                self._data[last_var_name] = np.squeeze(last_var_arr, axis=idx)
            del new_swp_vars[idx]
            self._swp_pars = ImmutableList(new_swp_vars)
        else:
            del new_swp_vars[idx]

            # remove corners
            swp_names = new_swp_vars[1:]
            sig_shape = self._data[self._signals[0]].shape
            num_env = sig_shape[0]
            if len(sig_shape) == 2:
                # inner most dimension is part of param sweep
                swp_shape, swp_vals = _check_is_md(num_env, [self._data[par] for par in swp_names],
                                                   rtol, atol, None)
                if swp_shape is not None:
                    for par, vals in zip(swp_names, swp_vals):
                        self._data[par] = vals
            else:
                # inner most dimension is not part of param sweep
                last_par = swp_names[-1]
                last_dset = self._data[last_par]
                swp_names = swp_names[:-1]
                swp_shape, swp_vals = _check_is_md(num_env, [self._data[par] for par in swp_names],
                                                   rtol, atol, last_dset.shape[-1])

                if len(swp_names) == 0:  # TODO: this is a hack to fix for 1 variable sweep
                    swp_shape = list(swp_shape)
                    swp_shape[-1] *= self._data[name].size
                    swp_shape = tuple(swp_shape)
                    
                if swp_shape is not None:
                    for par, vals in zip(swp_names, swp_vals):
                        self._data[par] = vals

                    if len(last_dset.shape) > 1:
                        self._data[last_par] = last_dset.reshape(swp_shape)

            self._swp_pars = ImmutableList(new_swp_vars)
            del self._data[name]
            if swp_shape is not None:
                # this is multi-D
                for sig in self._signals:
                    self._data[sig] = self._data[sig].reshape(swp_shape)

                self._is_md = True

        return True


class SimData:
    """A data structure that stores simulation data as a multi-dimensional array."""

    def __init__(self, sim_envs: Sequence[str], data: Dict[str, AnalysisData],
                 sim_netlist_type: DesignOutput) -> None:
        if not data:
            raise ValueError('Empty simulation data.')

        self._sim_envs = ImmutableList(sim_envs)
        self._table = data
        self._cur_name = next(iter(self._table.keys()))
        self._cur_ana: AnalysisData = self._table[self._cur_name]
        self._netlist_type = sim_netlist_type

    @property
    def group(self) -> str:
        return self._cur_name

    @property
    def group_list(self) -> List[str]:
        return list(self._table.keys())

    @property
    def sim_envs(self) -> ImmutableList[str]:
        return self._sim_envs

    @property
    def sweep_params(self) -> ImmutableList[str]:
        return self._cur_ana.sweep_params

    @property
    def signals(self) -> List[str]:
        return self._cur_ana.signals

    @property
    def is_md(self) -> bool:
        return self._cur_ana.is_md

    @property
    def data_shape(self) -> Tuple[int, ...]:
        return self._cur_ana.data_shape

    @property
    def netlist_type(self) -> DesignOutput:
        return self._netlist_type

    def __getitem__(self, item: str) -> np.ndarray:
        if item.endswith('>'):
            item = convert_cdba_name_bit(item, self._netlist_type)
        return self._cur_ana[item]

    def __contains__(self, item: str) -> bool:
        return item in self._cur_ana

    def items(self) -> ItemsView[str, np.ndarray]:
        return self._cur_ana.items()

    def open_group(self, val: str) -> None:
        tmp = self._table.get(val, None)
        if tmp is None:
            raise ValueError(f'Group {val} not found.')

        self._cur_name = val
        self._cur_ana = tmp

    def open_analysis(self, atype: AnalysisType) -> None:
        self.open_group(atype.name.lower())

    def insert(self, name: str, data: np.ndarray) -> None:
        self._cur_ana.insert(name, data)

    def add(self, new_data: Dict[str, np.ndarray]):
        self._cur_ana.add(new_data)

    def copy(self, rename: Optional[Dict[str, str]] = None) -> SimData:
        if rename is None:
            rename = {}
        _table = {}
        for k, v in self._table.items():
            key = rename.get(k, k)
            _table[key] = self._table[k]
        return SimData(self._sim_envs, _table, self.netlist_type)

    def deep_copy(self, rename: Optional[Dict[str, str]] = None) -> SimData:
        if rename is None:
            rename = {}
        _table = {}
        for k, v in self._table.items():
            key = rename.get(k, k)
            _table[key] = self._table[k].copy()
        return SimData(self._sim_envs, _table, self.netlist_type)

    def remove_sweep(self, name: str, rtol: float = 1e-8, atol: float = 1e-20) -> bool:
        return self._cur_ana.remove_sweep(name, rtol=rtol, atol=atol)

    def get_param_value(self, name: str) -> np.ndarray:
        return self._cur_ana.get_param_value(name)

    @classmethod
    def combine(cls, data_list: List[SimData], swp_name: str,
                swp_vals: Optional[np.ndarray] = None) -> SimData:
        ndata = len(data_list)
        if ndata < 1:
            raise ValueError('Must combine at least 1 data.')

        data0 = data_list[0]
        sim_envs = data0.sim_envs
        new_data = {}
        for grp in data0.group_list:
            ana_list = [sim_data._table[grp] for sim_data in data_list]
            new_data[grp] = AnalysisData.combine(ana_list, swp_name, swp_vals=swp_vals, axis=1)

        return SimData(sim_envs, new_data, data0.netlist_type)


def _check_is_md(num_env: int, swp_vals: List[np.ndarray], rtol: float, atol: float,
                 last: Optional[int]) -> Tuple[Optional[Tuple[int, ...]], List[np.ndarray]]:
    num = len(swp_vals)
    shape_list = [num_env] * (num + 1)
    new_vals = [np.nan] * num
    prev_size = 1
    for idx in range(num - 1, -1, -1):
        cur_vals = swp_vals[idx]
        if prev_size > 1:
            rep_prev = cur_vals.size // prev_size
            for start_idx in range(0, rep_prev * prev_size, prev_size):
                if not np.allclose(cur_vals[start_idx:start_idx + prev_size], cur_vals[start_idx],
                                   rtol=rtol, atol=atol):
                    # is not MD
                    return None, []
            cur_vals = cur_vals[0::prev_size]

        occ_vec = np.nonzero(np.isclose(cur_vals, cur_vals[0], rtol=rtol, atol=atol))[0]
        if occ_vec.size < 2:
            unique_size = cur_vals.size
        else:
            unique_size = occ_vec[1]
            rep, remain = divmod(cur_vals.size, unique_size)
            if remain != 0 or not np.allclose(cur_vals, np.tile(cur_vals[:unique_size], rep),
                                              rtol=rtol, atol=atol):
                # is not MD
                return None, []

        new_vals[idx] = cur_vals[:unique_size]
        shape_list[idx + 1] = unique_size
        prev_size *= unique_size

    if last is not None:
        shape_list.append(last)
    return tuple(shape_list), new_vals
