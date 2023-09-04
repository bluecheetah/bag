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

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Mapping, Union, Optional, Type, cast, Sequence

import abc
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

from pybag.enum import LogLevel

from ..util.logging import LoggingBase
from ..util.importlib import import_class
from ..io.file import write_yaml
from ..concurrent.core import batch_async_task

from .core import TestbenchManager

if TYPE_CHECKING:
    from .cache import SimulationDB, DesignInstance, SimResults, MeasureResult


@dataclass
class MeasInfo:
    state: str
    prev_results: Mapping[str, Any]


class MeasurementManager(LoggingBase, abc.ABC):
    """A class that handles circuit performance measurement.

    This class handles all the steps needed to measure a specific performance
    metric of the device-under-test.  This may involve creating and simulating
    multiple different testbenches, where configuration of successive testbenches
    depends on previous simulation results. This class reduces the potentially
    complex measurement tasks into a few simple abstract methods that designers
    simply have to implement.
    """

    def __init__(self, meas_specs: Mapping[str, Any], log_file: str,
                 log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        LoggingBase.__init__(self, self.__class__.__name__, log_file, log_level=log_level)

        self._specs: Mapping[str, Any] = {k: deepcopy(v) for k, v in meas_specs.items()}
        self._precision = precision
        self.commit()

    @property
    def specs(self) -> Mapping[str, Any]:
        return self._specs

    @property
    def precision(self) -> int:
        return self._precision

    def commit(self) -> None:
        """Commit changes to specs dictionary.  Perform necessary initialization."""
        pass

    def make_tbm(self, tbm_cls: Union[Type[TestbenchManager], str], tbm_specs: Mapping[str, Any],
                 ) -> TestbenchManager:
        obj_cls = cast(Type[TestbenchManager], import_class(tbm_cls))
        return obj_cls(None, Path(), '', '', tbm_specs, None, None,
                       precision=self._precision, logger=self.logger)

    def make_mm(self, mm_cls: Union[Type[MeasurementManager], str], mm_specs: Mapping[str, Any]
                ) -> MeasurementManager:
        obj_cls = cast(Type[MeasurementManager], import_class(mm_cls))
        return obj_cls(mm_specs, self.log_file, log_level=self.log_level, precision=self._precision)

    @abc.abstractmethod
    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        """A coroutine that performs measurement.

        Parameters
        ----------
        name : str
            name of this measurement.
        sim_dir : Path
            simulation directory.
        sim_db : SimulationDB
            the simulation database object.
        dut : Optional[DesignInstance]
            the DUT to measure.
        harnesses : Optional[Sequence[DesignInstance]]
            the list of DUT and harnesses to measure.

        Returns
        -------
        output : Mapping[str, Any]
            the measurement results.
        """
        pass

    def measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                            dut: Optional[DesignInstance], harnesses: Optional[Sequence[DesignInstance]] = None
                            ) -> Mapping[str, Any]:
        if harnesses:
            coro = self.async_measure_performance(name, sim_dir, sim_db, dut, harnesses)
        else:  # for backwards compatibility
            coro = self.async_measure_performance(name, sim_dir, sim_db, dut)
        results = batch_async_task([coro])
        if results is None:
            return {}

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans


class MeasurementManagerFSM(MeasurementManager, abc.ABC):
    """A class that handles circuit performance measurement in an FSM-like fashion.

    Any subclass of MeasurementManagerFSM will need to implement the following methods:
        initialize, process_output, get_sim_info.

    Refer to async_measure_performance to see how the above methods are integrated together into the measurement process.
    """

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        name : str
            name of this measurement.
        sim_dir : Path
            simulation directory.
        sim_db : SimulationDB
            the simulation database object.
        dut : Optional[DesignInstance]
            the DUT to measure.
        harnesses : Optional[Sequence[DesignInstance]]
            the list of DUT and harnesses to measure.

        Returns
        -------
        output : Mapping[str, Any]
            the last dictionary returned by process_output().
        """
        if harnesses:
            done, cur_info = self.initialize(sim_db, dut, harnesses)
        else:  # for backwards compatibility
            done, cur_info = self.initialize(sim_db, dut)
        while not done:
            cur_state = cur_info.state
            self.log(f'Measurement {name}, state {cur_state}')
            sim_id = f'{name}_{cur_state}'

            # create and setup testbench
            if harnesses:
                sim_object, use_dut = self.get_sim_info(sim_db, dut, cur_info, harnesses)
            else:  # for backwards compatibility
                sim_object, use_dut = self.get_sim_info(sim_db, dut, cur_info)
            cur_dut = dut if use_dut else None
            if isinstance(sim_object, MeasurementManager):
                sim_results = await sim_db.async_simulate_mm_obj(sim_id, sim_dir / cur_state,
                                                                 cur_dut, sim_object, harnesses)
            else:
                tbm, tb_params = sim_object
                sim_results = await sim_db.async_simulate_tbm_obj(cur_state, sim_dir / cur_state,
                                                                  cur_dut, tbm, tb_params,
                                                                  tb_name=sim_id, harnesses=harnesses)

            self.log(f'Processing output of {name}, state {cur_state}')
            done, next_info = self.process_output(cur_info, sim_results)
            write_yaml(sim_dir / f'{cur_state}.yaml', next_info.prev_results)
            cur_info = next_info

        self.log(f'Measurement {name} done, recording results.')
        result = cur_info.prev_results
        write_yaml(sim_dir / f'{name}.yaml', cur_info.prev_results)
        return result

    @abc.abstractmethod
    def initialize(self, sim_db: SimulationDB, dut: DesignInstance,
                   harnesses: Optional[Sequence[DesignInstance]] = None) -> Tuple[bool, MeasInfo]:
        """Initialize this MeasurementManager to get ready for measurement.

        Parameters
        ----------
        sim_db : SimulationDB
            the simulation database object.
        dut : DesignInstance
            the design instance.
        harnesses : Optional[Sequence[DesignInstance]]
            the list of harness instances.

        Returns
        -------
        done : bool
            If True, then do not run measurement.
        info : MeasInfo
            the initial MeasInfo object.
        """
        pass

    @abc.abstractmethod
    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        """Process simulation output data.

        Parameters
        ----------
        cur_info : MeasInfo
            the MeasInfo object representing the current measurement state.
        sim_results : Union[SimResults, MeasureResult]
            the simulation results object.

        Returns
        -------
        done : bool
            True if this measurement is finished.
        next_info : MeasInfo
            the updated measurement state.
        """
        pass

    @abc.abstractmethod
    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo,
                     harnesses: Optional[Sequence[DesignInstance]] = None
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        """Get the testbench manager needed for the current measurement state.

        Override to customize your testbench manager.

        Parameters
        ----------
        sim_db : SimulationDB
            the simulation database object.
        dut : DesignInstance
            the design instance.
        cur_info: MeasInfo
            the MeasInfo object representing the current measurement state.
        harnesses : Optional[Sequence[DesignInstance]]
            the list of harness instances

        Returns
        -------
        sim_object : Union[Tuple[TestbenchManager, Mapping[str, Any]], MeasurementManager]
            either a TestbenchManager/tb_params tuple, or a measurement manager instance.
        use_dut : bool
            True to run simulation with DesignInstance.
        """
        pass
