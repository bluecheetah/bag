# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
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

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Optional, Dict, Any, Tuple, List, Iterable, Sequence, Type, Mapping, Union, cast
)

import abc
import importlib
import itertools
from pathlib import Path
from copy import deepcopy

import numpy as np

from pybag.enum import DesignOutput, LogLevel
from pybag.core import FileLogger

from ..math import float_to_si_string
from ..io.file import read_yaml, write_yaml
from ..util.immutable import ImmutableList
from ..util.math import Calculator
from ..layout.template import TemplateDB, TemplateBase
from ..design.database import ModuleDB, ModuleType
from ..design.module import Module
from ..concurrent.core import batch_async_task
from ..design.netlist import add_mismatch_offsets

from .base import SimAccess
from .data import SimNetlistInfo, SimData, swp_info_from_struct

if TYPE_CHECKING:
    from ..core import BagProject


class TestbenchManager(abc.ABC):
    """A class that creates and setups up a testbench for simulation, then save the result.

    This class is used by MeasurementManager to run simulations.

    Parameters
    ----------
    sim : SimAccess
        the simulator interface object.
    work_dir : Path
        working directory path.
    tb_name : str
        testbench name.
    impl_lib : str
        unused.  Remain for backward compatibility.
    specs : Mapping[str, Any]
        testbench specs.
    sim_view_list : Optional[Sequence[Tuple[str, str]]]
        unused.  Remain for backward compatibility.
    env_list : Optional[Sequence[str]]
        unused.  Remain for backward compatibility.
    precision : int
        numeric precision in simulation netlist generation.
    logger : Optional[FileLogger]
        the logger object.

    Notes
    -----
    The specification dictionary for all testbenches have the following entries:

    sim_envs : Sequence[str]
        list of simulation environments.
    sim_params : Mapping[str, Any]
        simulation parameters dictionary.
    swp_info : Union[Sequence[Any], Mapping[str, Any]]
        Optional.  the parameter sweep data structure.
    sim_options: Mapping[str, Any]
        Optional.  Simulator-specific options.
    monte_carlo_params: Mapping[str, Any]
        Optional.  If specified, will run Monte Carlo with the given parameters.
    """

    # noinspection PyUnusedLocal
    def __init__(self, sim: Optional[SimAccess], work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Mapping[str, Any], sim_view_list: Optional[Sequence[Tuple[str, str]]],
                 env_list: Optional[Sequence[str]], precision: int = 6,
                 logger: Optional[FileLogger] = None) -> None:
        # TODO: refactor TestbenchManager to remove unused variables
        # TODO: remova TestbenchManager dependency on SimAccess.  Make it look more like
        # TODO: MeasurementManager in general
        self._sim = sim
        self._work_dir = work_dir.resolve()
        self._tb_name = tb_name
        self._precision = precision
        self._logger = logger

        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._specs: Dict[str, Any] = {k: deepcopy(v) for k, v in specs.items()
                                       if k is not 'sim_params' and k is not 'env_params'}
        self._specs['sim_params'] = {k: v for k, v in specs['sim_params'].items()}
        self._specs['env_params'] = {k: v.copy() for k, v in specs.get('env_params', {}).items()}
        self.commit()

    @property
    def logger(self) -> FileLogger:
        return self._logger

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @property
    def swp_info(self) -> Union[Sequence[Any], Mapping[str, Any]]:
        return self._specs.get('swp_info', [])

    @property
    def sim_params(self) -> Dict[str, Union[float, str]]:
        """Dict[str, Union[float, str]]: Simulation parameters dictionary, can be modified."""
        return self._specs['sim_params']

    @property
    def env_params(self) -> Dict[str, Dict[str, float]]:
        return self._specs['env_params']

    @property
    def num_sim_envs(self) -> int:
        return len(self._specs['sim_envs'])

    @property
    def sim_envs(self) -> Sequence[str]:
        return sorted(self._specs['sim_envs'])

    @property
    def tb_netlist_path(self) -> Path:
        return self._work_dir / f'tb.{self._sim.netlist_type.extension}'

    @property
    def sim_netlist_path(self) -> Path:
        return self._work_dir / f'sim.{self._sim.netlist_type.extension}'

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def sweep_shape(self) -> Tuple[int, ...]:
        return self.get_sweep_shape(self.num_sim_envs, self.swp_info)

    @classmethod
    @abc.abstractmethod
    def get_schematic_class(cls) -> Type[Module]:
        pass

    @classmethod
    def get_sweep_shape(cls, num_envs: int, swp: Union[Sequence[Any], Mapping[str, Any]]
                        ) -> Tuple[int, ...]:
        obj = swp_info_from_struct(swp)
        return (num_envs,) + obj.shape

    @abc.abstractmethod
    def get_netlist_info(self) -> SimNetlistInfo:
        """Returns the netlist information object.

        Returns
        -------
        netlist_info : SimNetlistInfo
            the simulation netlist information object.
        """
        pass

    def set_sim_envs(self, env_list: Sequence[str]) -> None:
        self._specs['sim_envs'] = env_list

    def set_swp_info(self, new_swp: Union[Sequence[Any], Mapping[str, Any]]) -> None:
        self._specs['swp_info'] = new_swp

    def commit(self) -> None:
        """Commit changes to specs dictionary.  Perform necessary initialization."""
        pass

    # noinspection PyMethodMayBeStatic
    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """Override to perform any operations prior to calling the setup() function.

        Parameters
        ----------
        sch_params :
            the testbench schematic parameters.  None means the previous testbench will be reused.
            This dictionary should not be modified.

        Returns
        -------
        new_params :
            the schematic parameters to use.  Could be a modified copy of the original.
        """
        return sch_params

    def print_results(self, data: SimData) -> None:
        """Override to print results."""
        pass

    def get_netlist_info_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representing the SimNetlistInfo object.

        This is a helper function that performs common boiler-plate setup (set corners,
        swp_info, sim_params, etc.)
        """
        sim_envs: Sequence[str] = self._specs['sim_envs']
        env_values = {var: [val_table[env] for env in sim_envs]
                      for var, val_table in self.env_params.items()}
        sim_setup = dict(
            sim_envs=sim_envs,
            params=self.sim_params,
            swp_info=self.swp_info,
            options=self._specs.get('sim_options', {}),
            monte_carlo=self._specs.get('monte_carlo_params', {}),
            env_params=env_values,
        )
        return sim_setup

    def get_env_param_value(self, name: str, sim_envs: Sequence[str]) -> np.ndarray:
        env_params: Mapping[str, Mapping[str, float]] = self._specs.get('env_params', {})
        val_table = env_params[name]
        return np.array([val_table[env] for env in sim_envs])

    def get_sim_param_value(self, val: Union[float, str]) -> float:
        cur_val = val
        sim_params = self.sim_params
        while isinstance(cur_val, str):
            cur_val = sim_params[cur_val]
        return cur_val

    def get_param_value(self, name: str, data: SimData) -> np.ndarray:
        try:
            return data.get_param_value(name)
        except ValueError:
            # this parameter is not swept
            env_params: Mapping[str, Mapping[str, float]] = self._specs.get('env_params', {})

            data_shape = self.sweep_shape
            val_table = env_params.get(name, None)
            if val_table is None:
                # this parameter is constant
                val = self.sim_params[name]
                if isinstance(val, str):
                    # this parameter is an expression
                    return self.get_calculator(data).eval(val)
                return np.full(data_shape, val)
            else:
                # this param is not constant
                return _get_env_param_value(data.sim_envs, data_shape, val_table)

    def get_calculator(self, data: SimData) -> Calculator:
        # TODO: to avoid potential re-creation of get_calculator() over and over again,
        # TODO: consider just storing these data in SimData on creation.
        env_params: Mapping[str, Mapping[str, float]] = self._specs.get('env_params', {})

        # get sweep param values
        swp = swp_info_from_struct(self.swp_info)
        namespace = {name: data.get_param_value(name) for name in swp}

        # get env param values
        sim_envs = data.sim_envs
        data_shape = self.sweep_shape
        for name, val_table in env_params.items():
            if name not in namespace:
                namespace[name] = _get_env_param_value(sim_envs, data_shape, val_table)

        # get sim param values
        expr_map = {}
        for name, val in self.sim_params.items():
            if name not in namespace:
                if isinstance(val, str):
                    expr_map[name] = val
                else:
                    namespace[name] = np.full(data_shape, val)

        while expr_map:
            success_once = False
            new_expr_map = {}
            for name, expr in expr_map.items():
                try:
                    val = Calculator(namespace).eval(expr)
                    namespace[name] = val
                    success_once = True
                except KeyError:
                    new_expr_map[name] = expr
            if not success_once:
                raise ValueError('sim_params circular dependency found.')
            expr_map = new_expr_map

        return Calculator(namespace)

    def get_sim_param_string(self, val: Union[float, str]) -> str:
        if isinstance(val, str):
            return val
        return float_to_si_string(val, self.precision)

    def update(self, work_dir: Path, tb_name: str, sim: Optional[SimAccess] = None) -> None:
        """Update the working directory and testbench name.

        This method allows you to reuse the same testbench manager objects for different
        testbenches.
        """
        if work_dir is not None:
            self._work_dir = work_dir.resolve()
            self._work_dir.mkdir(parents=True, exist_ok=True)
        if tb_name:
            self._tb_name = tb_name
        if sim is not None:
            self._sim = sim

    def setup(self, sch_db: ModuleDB, sch_params: Optional[Mapping[str, Any]],
              dut_cv_info_list: List[Any], dut_netlist: Optional[Path], gen_sch: bool = True,
              work_dir: Optional[Path] = None, tb_name: str = '') -> None:
        self.update(work_dir, tb_name)

        tb_netlist_path = self.tb_netlist_path
        sim_netlist_path = self.sim_netlist_path
        sch_params = self.pre_setup(sch_params)
        if sch_params is not None:
            # noinspection PyTypeChecker
            sch_master = sch_db.new_master(self.get_schematic_class(), sch_params)
            if gen_sch:
                self.log(f'Creating testbench {self._tb_name} schematic master')
                sch_db.batch_schematic([(sch_master, self._tb_name)])
                self.log(f'Testbench {self._tb_name} schematic master done')

            # create netlist for tb schematic
            self.log(f'Creating testbench {self._tb_name} netlist')
            net_str = '' if dut_netlist is None else str(dut_netlist.resolve())
            sch_db.batch_schematic([(sch_master, self._tb_name)], output=self._sim.netlist_type,
                                   top_subckt=False, fname=str(tb_netlist_path),
                                   cv_info_list=dut_cv_info_list, cv_netlist=net_str)
            self.log(f'Testbench {self._tb_name} netlisting done')

        netlist_info = self.get_netlist_info()
        self._sim.create_netlist(sim_netlist_path, tb_netlist_path, netlist_info,
                                 self._precision)

    async def async_simulate(self) -> None:
        self.log(f'Simulating {self._tb_name}')
        await self._sim.async_run_simulation(self.sim_netlist_path, 'sim')
        self.log(f'Finished simulating {self._tb_name}')

    def simulate(self) -> None:
        coro = self.async_simulate()
        batch_async_task([coro])

    def load_sim_data(self) -> SimData:
        return self._sim.load_sim_data(self._work_dir, 'sim')

    def log(self, msg: str, level: LogLevel = LogLevel.INFO) -> None:
        if self._logger is None:
            print(msg)
        else:
            self._logger.log(level, msg)

    def error(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.log(LogLevel.ERROR, msg)
        raise ValueError(msg)


def _get_env_param_value(sim_envs: Sequence[str], data_shape: Tuple[int, ...],
                         val_table: Mapping[str, float]) -> np.ndarray:
    new_shape = [1] * len(data_shape)
    new_shape[0] = len(sim_envs)
    values = np.array([val_table[env] for env in sim_envs])
    values = values.reshape(new_shape)
    return np.broadcast_to(values, data_shape)


class MeasurementManager(abc.ABC):
    """A class that handles circuit performance measurement.

    This class handles all the steps needed to measure a specific performance
    metric of the device-under-test.  This may involve creating and simulating
    multiple different testbenches, where configuration of successive testbenches
    depends on previous simulation results. This class reduces the potentially
    complex measurement tasks into a few simple abstract methods that designers
    simply have to implement.

    Parameters
    ----------
    sim : SimAccess
        the simulator interface object.
    dir_path : Path
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    wrapper_lookup : Dict[str, str]
        the DUT wrapper cell name lookup table.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    precision : int
        numeric precision in simulation netlist generation.
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        self._sim = sim
        self._dir_path = dir_path.resolve()
        self._meas_name = meas_name
        self._impl_lib = impl_lib
        self._specs = specs
        self._wrapper_lookup = wrapper_lookup
        self._sim_view_list = sim_view_list
        self._env_list = env_list
        self._precision = precision

        self._dir_path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return ''

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Get information about the next testbench.

        Override this method to perform more complex operations.

        Parameters
        ----------
        state : str
            the current FSM state.
        prev_output : Optional[Dict[str, Any]]
            the previous post-processing output.

        Returns
        -------
        tb_name : str
            cell name of the next testbench.  Should incorporate self.meas_name to avoid
            collision with testbench for other designs.
        tb_type : str
            the next testbench type.
        tb_specs : Dict[str, Any]
            the testbench specification dictionary.
        tb_params : Optional[Dict[str, Any]]
            the next testbench schematic parameters.  If we are reusing an existing
            testbench, this should be None.
        """
        tb_type = state
        tb_name = self.get_testbench_name(tb_type)
        tb_specs = self.get_testbench_specs(tb_type).copy()
        tb_params = self.get_default_tb_sch_params(tb_type)

        return tb_name, tb_type, tb_specs, tb_params

    @abc.abstractmethod
    def process_output(self, state: str, data: SimData, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process simulation output data.

        Parameters
        ----------
        state : str
            the current FSM state
        data : SimData
            simulation data dictionary.
        tb_manager : TestbenchManager
            the testbench manager object.

        Returns
        -------
        done : bool
            True if this measurement is finished.
        next_state : str
            the next FSM state.
        output : Dict[str, Any]
            a dictionary containing post-processed data.
        """
        return False, '', {}

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @property
    def data_dir(self) -> Path:
        return self._dir_path

    @property
    def sim_envs(self) -> Sequence[str]:
        return self._env_list

    def get_testbench_name(self, tb_type: str) -> str:
        """Returns a default testbench name given testbench type."""
        return f'{self._meas_name}_TB_{tb_type}'

    async def async_measure_performance(self, sch_db: Optional[ModuleDB], dut_cvi_list: List[Any],
                                        dut_netlist: Optional[Path], load_from_file: bool = False,
                                        gen_sch: bool = True) -> Dict[str, Any]:
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        sch_db : Optional[ModuleDB]
            the schematic database.

            if load_from_file is True, this can be None. as it will not be used unless necessary.
        dut_cvi_list : List[str]
            cv_info for DUT cell netlist

            if load_from_file is True, this will not be used unless necessary.
        dut_netlist : Optional[Path]
            netlist of DUT cell

            if load_from_file is True, this will not be used unless necessary.
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.
        gen_sch : bool
            True to create testbench schematics.

        Returns
        -------
        output : Dict[str, Any]
            the last dictionary returned by process_output().
        """
        cur_state = self.get_initial_state()
        prev_output = None
        done = False

        while not done:
            # create and setup testbench
            tb_name, tb_type, tb_specs, tb_sch_params = self.get_testbench_info(cur_state,
                                                                                prev_output)

            tb_package = tb_specs['tb_package']
            tb_cls_name = tb_specs['tb_class']
            tb_module = importlib.import_module(tb_package)
            tb_cls = getattr(tb_module, tb_cls_name)
            work_dir = self._dir_path / cur_state
            tb_manager: TestbenchManager = tb_cls(self._sim, work_dir, tb_name, self._impl_lib,
                                                  tb_specs, self._sim_view_list, self._env_list,
                                                  precision=self._precision)

            if load_from_file:
                print(f'Measurement {self._meas_name} in state {cur_state}, '
                      'load sim data from file.')
                try:
                    cur_results = tb_manager.load_sim_data()
                except FileNotFoundError:
                    print('Cannot find data file, simulating...')
                    if sch_db is None or not dut_cvi_list or dut_netlist is None:
                        raise ValueError('Cannot create testbench as DUT netlist not given.')

                    tb_manager.setup(sch_db, tb_sch_params, dut_cv_info_list=dut_cvi_list,
                                     dut_netlist=dut_netlist, gen_sch=gen_sch)
                    await tb_manager.async_simulate()
                    cur_results = tb_manager.load_sim_data()
            else:
                tb_manager.setup(sch_db, tb_sch_params, dut_cv_info_list=dut_cvi_list,
                                 dut_netlist=dut_netlist, gen_sch=gen_sch)
                await tb_manager.async_simulate()
                cur_results = tb_manager.load_sim_data()

            # process and save simulation data
            print(f'Measurement {self._meas_name} in state {cur_state}, '
                  f'processing data from {tb_type}')
            done, next_state, prev_output = self.process_output(cur_state, cur_results, tb_manager)
            write_yaml(self._dir_path / f'{cur_state}.yaml', prev_output)

            cur_state = next_state

        write_yaml(self._dir_path / f'{self._meas_name}.yaml', prev_output)
        return prev_output

    def measure_performance(self, sch_db: Optional[ModuleDB], dut_cvi_list: List[Any],
                            dut_netlist: Optional[Path], load_from_file: bool = False,
                            gen_sch: bool = True) -> Dict[str, Any]:
        coro = self.async_measure_performance(sch_db, dut_cvi_list, dut_netlist,
                                              load_from_file=load_from_file,
                                              gen_sch=gen_sch)
        return batch_async_task([coro])[0]

    def get_state_output(self, state: str) -> Dict[str, Any]:
        """Get the post-processed output of the given state."""
        return read_yaml(self._dir_path / f'{state}.yaml')

    def get_testbench_specs(self, tb_type: str) -> Dict[str, Any]:
        """Helper method to get testbench specifications."""
        return self._specs['testbenches'][tb_type]

    def get_default_tb_sch_params(self, tb_type: str) -> Dict[str, Any]:
        """Helper method to return a default testbench schematic parameters dictionary.

        This method loads default values from specification file, the fill in dut_lib
        and dut_cell for you.

        Parameters
        ----------
        tb_type : str
            the testbench type.

        Returns
        -------
        sch_params : Dict[str, Any]
            the default schematic parameters dictionary.
        """
        tb_specs = self.get_testbench_specs(tb_type)
        wrapper_type = tb_specs.get('wrapper_type', '')

        if 'sch_params' in tb_specs:
            tb_params = tb_specs['sch_params'].copy()
        else:
            tb_params = {}

        tb_params['dut_lib'] = self._impl_lib
        tb_params['dut_cell'] = self._wrapper_lookup[wrapper_type]
        return tb_params


class DesignSpecs:
    """A class that parses the design specification file."""

    def __init__(self, spec_file: str, spec_dict: Optional[Dict[str, Any]] = None) -> None:
        if spec_dict:
            self._specs = spec_dict
            self._root_dir: Path = Path(self._specs['root_dir']).resolve()
        elif spec_file:
            spec_path = Path(spec_file).resolve()
            if spec_path.is_file():
                self._specs = read_yaml(spec_path)
                self._root_dir: Path = Path(self._specs['root_dir']).resolve()
            elif spec_path.is_dir():
                self._root_dir: Path = spec_path
                self._specs = read_yaml(self._root_dir / 'specs.yaml')
            else:
                raise ValueError(f'{spec_path} is neither data directory or specification file.')
        else:
            raise ValueError('spec_file is empty.')

        cls_package = self._specs['layout_package']
        cls_name = self._specs['layout_class']
        self._create_layout = cls_package and cls_name

        self._swp_var_list: ImmutableList[str] = ImmutableList(
            sorted(self._specs['sweep_params'].keys()))
        self._sweep_params = self._specs['sweep_params']
        self._params = self._specs['layout_params' if self._create_layout else 'schematic_params']

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def swp_var_list(self) -> ImmutableList[str]:
        return self._swp_var_list

    @property
    def dut_lib(self) -> str:
        return self._specs['dut_lib']

    @property
    def dut_cell(self) -> str:
        return self._specs['dut_cell']

    @property
    def impl_lib(self) -> str:
        return self._specs['impl_lib']

    @property
    def env_list(self) -> List[str]:
        return self._specs['env_list']

    @property
    def view_name(self) -> str:
        return self._specs['view_name']

    @property
    def dsn_basename(self) -> str:
        return self._specs['dsn_basename']

    @property
    def summary_fname(self) -> str:
        return self._specs['summary_fname']

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @property
    def create_layout(self) -> bool:
        return self._create_layout

    @property
    def first_params(self) -> Dict[str, Any]:
        combo = [self._sweep_params[key][0] for key in self._swp_var_list]
        return self._get_params(combo)

    def get_data_dir(self, dsn_name: str, meas_type: str) -> Path:
        """Returns the data directory path for the given measurement."""
        return self._root_dir.joinpath(dsn_name, meas_type)

    def get_swp_values(self, var: str) -> List[Any]:
        """Returns a list of valid sweep variable values.

        Parameter
        ---------
        var : str
            the sweep variable name.

        Returns
        -------
        val_list : List[Any]
            the sweep values of the given variable.
        """
        return self._sweep_params[var]

    def swp_combo_iter(self) -> Iterable[Tuple[Any, ...]]:
        """Returns an iterator of schematic parameter combinations we sweep over.

        Returns
        -------
        combo_iter : Iterable[Tuple[Any, ...]]
            an iterator of tuples of schematic parameters values that we sweep over.
        """
        return itertools.product(*(self._sweep_params[var] for var in self._swp_var_list))

    def dsn_param_iter(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Returns an iterator of design cell name and the parameter dictionary."""
        for combo in self.swp_combo_iter():
            yield self.get_design_name(combo), self._get_params(combo)

    def dsn_name_iter(self) -> Iterable[str]:
        return (self.get_design_name(combo) for combo in self.swp_combo_iter())

    def get_layout_class(self) -> Type[TemplateBase]:
        cls_package = self._specs['layout_package']
        cls_name = self._specs['layout_class']
        lay_module = importlib.import_module(cls_package)
        return getattr(lay_module, cls_name)

    def get_design_name(self, combo_list: Sequence[Any]) -> str:
        name = self.dsn_basename
        for var, val in zip(self.swp_var_list, combo_list):
            if isinstance(val, str) or isinstance(val, int):
                name += f'_{var}_{val}'
            elif isinstance(val, float):
                name += f'_{var}_{float_to_si_string(val)}'
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name

    def _get_params(self, combo_list: Sequence[Any]) -> Dict[str, Any]:
        params = self._params.copy()
        for var, val in zip(self._swp_var_list, combo_list):
            params[var] = val
        return params


class DesignManager:
    """A class that manages instantiating design instances and running simulations.

    This class provides various methods to allow you to sweep design parameters
    and generate multiple instances at once.  It also provides methods for running
    simulations and helps you interface with TestbenchManager instances.

    Parameters
    ----------
    prj : BagProject
        The BagProject instance.
    spec_file : str
        the specification file name or the data directory.
    """

    def __init__(self, prj: BagProject, spec_file: str = '',
                 spec_dict: Optional[Dict[str, Any]] = None, sch_db: Optional[ModuleDB] = None,
                 lay_db: Optional[TemplateDB] = None) -> None:
        self._prj = prj
        self._info = DesignSpecs(spec_file, spec_dict)

        impl_lib = self._info.impl_lib
        if sch_db is None:
            self._sch_db = ModuleDB(prj.tech_info, impl_lib, prj=prj)
        else:
            self._sch_db = sch_db

        if lay_db is None:
            self._lay_db = TemplateDB(prj.grid, impl_lib, prj=prj)
        else:
            self._lay_db = lay_db

    @classmethod
    def load_state(cls, prj: BagProject, root_dir: str) -> DesignManager:
        """Create the DesignManager instance corresponding to data in the given directory."""
        return cls(prj, root_dir)

    @classmethod
    def get_wrapper_name(cls, dut_name: str, wrapper_name: str) -> str:
        """Returns the wrapper cell name corresponding to the given DUT."""
        return f'{dut_name}_WRAPPER_{wrapper_name}'

    @property
    def info(self) -> DesignSpecs:
        """Return the specification dictionary."""
        return self._info

    async def extract_design(self, lib_name: str, dsn_name: str,
                             rcx_params: Optional[Dict[str, Any]], netlist: Optional[Path]) -> Path:
        """A coroutine that runs LVS/RCX on a given design.

        Parameters
        ----------
        lib_name : str
            library name.
        dsn_name : str
            design cell name.
        rcx_params : Optional[Dict[str, Any]]
            extraction parameters dictionary.
        netlist: Path
            CDL netlist path

        Returns
        -------
        rcx_netlist : Path
            extracted netlist path
        """
        print(f'Running LVS on {dsn_name}')
        lvs_passed, lvs_log = await self._prj.async_run_lvs(lib_name, dsn_name, netlist=netlist,
                                                            run_rcx=True)
        if not lvs_passed:
            raise ValueError('LVS failed for %s.  Log file: %s' % (dsn_name, lvs_log))

        print(f'LVS passed on {dsn_name}')
        print(f'Running RCX on {dsn_name}')
        rcx_netlist, rcx_log = await self._prj.async_run_rcx(lib_name, dsn_name, params=rcx_params)
        if not rcx_netlist:
            raise ValueError(f'RCX failed for {dsn_name}.  Log file: {rcx_log}')
        print(f'RCX passed on {dsn_name}')
        return Path(rcx_netlist)

    async def verify_design(self, lib_name: str, dsn_name: str,
                            dut_cvi_list: List[Any], dut_netlist: Path,
                            load_from_file: bool = False, gen_sch: bool = True) -> None:
        """Run all measurements on the given design.

        Parameters
        ----------
        lib_name : str
            library name.
        dsn_name : str
            design cell name.
        dut_cvi_list : List[str]
            cv_info for DUT cell netlist
        dut_netlist : Path
            netlist of DUT cell
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.
        gen_sch : bool
            True to create testbench schematics.
        """
        root_dir = self._info.root_dir
        env_list = self._info.env_list
        view_name = self._info.view_name
        summary_fname = self._info.summary_fname
        meas_list = self._info.specs['measurements']
        wrapper_list = self._info.specs['dut_wrappers']

        wrapper_lookup = {'': dsn_name}
        for wrapper_config in wrapper_list:
            wrapper_type = wrapper_config['name']
            wrapper_lookup[wrapper_type] = self.get_wrapper_name(dsn_name, wrapper_type)

        result_summary = {}
        dsn_data_dir = root_dir / dsn_name
        for meas_specs in meas_list:
            meas_type = meas_specs['meas_type']
            meas_package = meas_specs['meas_package']
            meas_cls_name = meas_specs['meas_class']
            out_fname = meas_specs['out_fname']
            data_dir = self._info.get_data_dir(dsn_name, meas_type)
            meas_name = f'{dsn_name}_MEAS_{meas_type}'

            meas_module = importlib.import_module(meas_package)
            meas_cls = getattr(meas_module, meas_cls_name)

            meas_manager: MeasurementManager = meas_cls(self._prj.sim_access, data_dir, meas_name,
                                                        lib_name, meas_specs, wrapper_lookup,
                                                        [(dsn_name, view_name)], env_list)
            print(f'Performing measurement {meas_type} on {dsn_name}')
            meas_res = await meas_manager.async_measure_performance(self._sch_db,
                                                                    dut_cvi_list=dut_cvi_list,
                                                                    dut_netlist=dut_netlist,
                                                                    load_from_file=load_from_file,
                                                                    gen_sch=gen_sch)
            print(f'Measurement {meas_type} finished on {dsn_name}')

            write_yaml(data_dir / out_fname, meas_res)
            result_summary[meas_type] = meas_res

        write_yaml(dsn_data_dir / summary_fname, result_summary)

    async def main_task(self, lib_name: str, dsn_name: str, rcx_params: Optional[Dict[str, Any]],
                        dut_cv_info_list: List[str], dut_cdl_netlist: Path, dut_sim_netlist: Path,
                        extract: bool = True, measure: bool = True, load_from_file: bool = False,
                        gen_sch: bool = True, mismatch: bool = False) -> None:
        """The main coroutine."""
        if extract:
            dut_sim_netlist = await self.extract_design(lib_name, dsn_name, rcx_params,
                                                        netlist=dut_cdl_netlist)
        if measure:
            # TODO: fix mismatch for extracted designs
            if mismatch:
                add_mismatch_offsets(dut_sim_netlist, dut_sim_netlist)
            await self.verify_design(lib_name, dsn_name, load_from_file=load_from_file,
                                     dut_cvi_list=dut_cv_info_list, dut_netlist=dut_sim_netlist,
                                     gen_sch=gen_sch)

    def characterize_designs(self, generate: bool = True, measure: bool = True,
                             load_from_file: bool = False, gen_sch: bool = True,
                             mismatch: bool = False) -> None:
        """Sweep all designs and characterize them.

        Parameters
        ----------
        generate : bool
            If True, create schematic/layout and run LVS/RCX.
        measure : bool
            If True, run all measurements.
        load_from_file : bool
            If True, measurements will load existing simulation data
            instead of running simulations.
        gen_sch : bool
            If True, schematics will be generated.
        mismatch: bool
            If True, add mismatch offset voltage sources to netlist
        """
        impl_lib = self._info.impl_lib
        rcx_params = self._info.specs.get('rcx_params', None)

        extract = generate and self._info.view_name != 'schematic'
        flat_sch_netlist = mismatch and not extract

        dut_info_list = self.create_designs(gen_sch=gen_sch, flat_sch_netlist=flat_sch_netlist)

        coro_list = [self.main_task(impl_lib, dsn_name, rcx_params, extract=extract,
                                    measure=measure, load_from_file=load_from_file,
                                    dut_cv_info_list=cv_info_list,
                                    dut_cdl_netlist=netlist_cdl,
                                    dut_sim_netlist=netlist_sim, gen_sch=gen_sch,
                                    mismatch=mismatch)
                     for dsn_name, cv_info_list, netlist_cdl, netlist_sim in dut_info_list]

        results = batch_async_task(coro_list)
        if results is not None:
            for val in results:
                if isinstance(val, Exception):
                    raise val

    def get_result(self, dsn_name: str) -> Dict[str, Any]:
        """Returns the measurement result summary dictionary.

        Parameters
        ----------
        dsn_name : str
            the design name.

        Returns
        -------
        result : Dict[str, Any]
            the result dictionary.
        """
        return read_yaml(self._info.root_dir / dsn_name / self._info.summary_fname)

    def test_layout(self, gen_sch: bool = True) -> None:
        """Create a test schematic and layout for debugging purposes"""

        lay_params = self._info.first_params
        dsn_name = self._info.dsn_basename + '_TEST'

        print('create test layout')
        sch_name_param_list = self.create_dut_layouts([(dsn_name, lay_params)])

        if gen_sch:
            print('create test schematic')
            self.create_dut_schematics(sch_name_param_list, gen_wrappers=False)
        print('done')

    def create_designs(self, gen_sch: bool = True, flat_sch_netlist: bool = False
                       ) -> List[Tuple[str, List[Any], Path, Path]]:
        """Create DUT schematics/layouts.
        """
        dsn_param_iter = self._info.dsn_param_iter()
        if self._info.create_layout:
            print('creating all layouts.')
            dsn_param_iter = self.create_dut_layouts(dsn_param_iter)
            print('layout creation done.')

        return self.create_dut_schematics(dsn_param_iter, gen_wrappers=True, gen_sch=gen_sch,
                                          flat_sch_netlist=flat_sch_netlist)

    def create_dut_schematics(self, name_param_iter: Iterable[Tuple[str, Dict[str, Any]]],
                              gen_wrappers: bool = True, gen_sch: bool = True,
                              flat_sch_netlist: bool = False) -> List[Tuple[str, List[Any], Path,
                                                                            Path]]:
        root_dir = self._info.root_dir
        dut_lib = self._info.dut_lib
        dut_cell = self._info.dut_cell
        impl_lib = self._info.impl_lib
        wrapper_list = self._info.specs['dut_wrappers']

        netlist_type = self._prj.sim_access.netlist_type
        ext = netlist_type.extension
        dir_path = root_dir / 'designs' / impl_lib
        dir_path.mkdir(parents=True, exist_ok=True)

        results: List[Tuple[str, List[Any], Path, Path]] = []
        tot_info_list = []
        print('Generating DUT schematics and netlists')
        for cur_name, sch_params in name_param_iter:
            gen_cls = cast(Type[ModuleType], ModuleDB.get_schematic_class(dut_lib, dut_cell))
            sch_master = self._sch_db.new_master(gen_cls, sch_params)
            cur_info_list = [(sch_master, cur_name)]
            if gen_wrappers:
                for wrapper_config in wrapper_list:
                    wrapper_name = wrapper_config['name']
                    wrapper_lib = wrapper_config['lib']
                    wrapper_cell = wrapper_config['cell']
                    wrapper_params = wrapper_config['params'].copy()
                    wrapper_params['dut_lib'] = dut_lib
                    wrapper_params['dut_cell'] = dut_cell
                    wrapper_params['dut_params'] = sch_params
                    gen_cls_wrap = cast(Type[ModuleType],
                                        ModuleDB.get_schematic_class(wrapper_lib, wrapper_cell))
                    sch_master_wrap = self._sch_db.new_master(gen_cls_wrap, wrapper_params)
                    cur_info_list.append((sch_master_wrap,
                                          self.get_wrapper_name(cur_name, wrapper_name)))

            dut_netlist_sim = dir_path / f'{cur_name}.{ext}'
            dut_netlist_cdl = dir_path / f'{cur_name}.cdl'

            dut_cv_info_list = []
            self._sch_db.batch_schematic(cur_info_list, output=netlist_type, top_subckt=True,
                                         cv_info_out=dut_cv_info_list, fname=str(dut_netlist_sim),
                                         flat=flat_sch_netlist)
            self._sch_db.batch_schematic(cur_info_list, output=DesignOutput.CDL,
                                         fname=str(dut_netlist_cdl), flat=flat_sch_netlist)

            tot_info_list.extend(cur_info_list)
            results.append((cur_name, dut_cv_info_list, dut_netlist_cdl, dut_netlist_sim))

        print('DUT generation and netlisting done.')
        if gen_sch:
            print('Creating schematics...')
            self._sch_db.batch_schematic(tot_info_list)
            print('schematic creation done.')

        return results

    def create_dut_layouts(self, name_param_iter: Iterable[Tuple[str, Dict[str, Any]]]
                           ) -> Sequence[Tuple[str, Dict[str, Any]]]:
        """Create multiple layouts"""
        temp_cls = self._info.get_layout_class()

        info_list, sch_name_param_list = [], []
        for cell_name, lay_params in name_param_iter:
            template = self._lay_db.new_template(params=lay_params, temp_cls=temp_cls, debug=False)
            info_list.append((template, cell_name))
            sch_name_param_list.append((cell_name, template.sch_params))
        self._lay_db.batch_layout(info_list)
        return sch_name_param_list
