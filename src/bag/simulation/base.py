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


"""This module handles high level simulation routines.

This module defines SimAccess, which provides methods to run simulations
and retrieve results.
"""

from typing import Mapping, Any, Tuple, Union, Sequence, Dict, Type

import abc
from pathlib import Path

from pybag.enum import DesignOutput
from pybag.core import get_cdba_name_bits

from ..concurrent.core import SubProcessManager, batch_async_task
from ..util.importlib import import_class
from .data import SimNetlistInfo, SimData


def get_corner_temp(env_str: str) -> Tuple[str, int]:
    idx = env_str.rfind('_')
    if idx < 0:
        raise ValueError(f'Invalid environment string: {env_str}')
    return env_str[:idx], int(env_str[idx + 1:].replace('m', '-'))


def setup_corner(corner_str: str, temp: int) -> str:
    # Inverse of get_corner_temp
    # Useful to setup strings that can properly be parsed by it
    return corner_str + '_' + str(temp).replace('-', 'm')


def get_bit_list(pin: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(pin, str):
        return get_cdba_name_bits(pin) if pin else []
    else:
        return [val for p_ in pin for val in get_cdba_name_bits(p_)]


class SimAccess(abc.ABC):
    """A class that interacts with a simulator.

    Parameters
    ----------
    parent : str
        parent directory for SimAccess.
    sim_config : Mapping[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, parent: str, sim_config: Mapping[str, Any]) -> None:
        self._config = sim_config
        self._dir_path = (Path(parent) / "simulations").resolve()

    @property
    @abc.abstractmethod
    def netlist_type(self) -> DesignOutput:
        return DesignOutput.CDL

    @abc.abstractmethod
    def create_netlist(self, output_path: Path, sch_netlist: Path, info: SimNetlistInfo,
                       precision: int = 6) -> None:
        pass

    @abc.abstractmethod
    def get_sim_file(self, dir_path: Path, sim_tag: str) -> Path:
        """Returns path to the simulation file."""
        pass

    @abc.abstractmethod
    def load_sim_data(self, dir_path: Path, sim_tag: str) -> SimData:
        """Load simulation results.

        Parameters
        ----------
        dir_path : Path
            the working directory path.
        sim_tag : str
            optional simulation name.  Empty for default.

        Returns
        -------
        data : Dict[str, Any]
            the simulation data dictionary.
        """
        pass

    @abc.abstractmethod
    async def async_run_simulation(self, netlist: Path, sim_tag: str) -> None:
        """A coroutine for simulation a testbench.

        Parameters
        ----------
        netlist : Path
            the netlist file name.
        sim_tag : str
            optional simulation name.  Empty for default.
        """
        pass

    @property
    def dir_path(self) -> Path:
        """Path: the directory for simulation files."""
        return self._dir_path

    @property
    def config(self) -> Mapping[str, Any]:
        """Dict[str, Any]: simulation configurations."""
        return self._config

    def run_simulation(self, netlist: Path, sim_tag: str) -> None:
        coro = self.async_run_simulation(netlist, sim_tag)
        batch_async_task([coro])


class SimProcessManager(SimAccess, abc.ABC):
    """An implementation of :class:`SimAccess` using :class:`SubProcessManager`.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir: str, sim_config: Mapping[str, Any]) -> None:
        SimAccess.__init__(self, tmp_dir, sim_config)

        mgr_class: Type[SubProcessManager] = import_class(sim_config.get('mgr_class', SubProcessManager))
        mgr_kwargs: Dict[str, Any] = sim_config.get('mgr_kwargs', {})

        cancel_timeout = sim_config.get('cancel_timeout_ms', 10000) / 1e3

        self._manager: SubProcessManager = mgr_class(max_workers=sim_config.get('max_workers', 0),
                                                     cancel_timeout=cancel_timeout, **mgr_kwargs)

    @property
    def manager(self) -> SubProcessManager:
        return self._manager


class EmSimAccess(abc.ABC):
    """A class that interacts with an EM simulator.

    Parameters
    ----------
    parent : str
        parent directory for EmSimAccess.
    sim_config : Mapping[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, parent: str, sim_config: Mapping[str, Any]) -> None:
        self._config = sim_config
        self._dir_path = (Path(parent) / "em_simulations").resolve()

    @property
    def dir_path(self) -> Path:
        """Path: the directory for simulation files."""
        return self._dir_path

    @property
    def config(self) -> Mapping[str, Any]:
        """Mapping[str, Any]: simulation configurations."""
        return self._config

    @staticmethod
    def _get_em_base_path(root_path: Path) -> Path:
        return root_path.resolve() / 'em_meas'

    def get_log_path(self, root_path: Path) -> Path:
        """Path: the directory for simulation files."""
        return self._get_em_base_path(root_path) / 'bag_em.log'

    @abc.abstractmethod
    async def async_gen_nport(self, cell_name: str, gds_file: Path, params: Mapping[str, Any], root_path: Path,
                              run_sim: bool = False) -> Path:
        """A coroutine for running EM sim to generate nport for the current module.

        Parameters
        ----------
        cell_name : str
            Name of the cell
        gds_file : Path
            location of the gds file of the cell
        params : Mapping[str, Any]
            various EM parameters
        root_path : Path
            Root path for running sims and storing results
        run_sim : bool
            True to run EM sim; False by default

        Returns
        -------
        sp_file: Path
            location of generated s parameter file
        """
        pass

    def run_simulation(self, cell_name: str, gds_file: Path, params: Mapping[str, Any], root_path: Path) -> None:
        coro = self.async_gen_nport(cell_name, gds_file, params, root_path, run_sim=True)
        batch_async_task([coro])

    @abc.abstractmethod
    def process_output(self, cell_name: str, params: Mapping[str, Any], root_path: Path) -> None:
        pass


class EmSimProcessManager(EmSimAccess, abc.ABC):
    """An implementation of :class:`EmSimAccess` using :class:`SubProcessManager`.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for EmSimAccess.
    sim_config : Mapping[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir: str, sim_config: Mapping[str, Any]) -> None:
        EmSimAccess.__init__(self, tmp_dir, sim_config)

        mgr_class: Type[SubProcessManager] = import_class(sim_config.get('mgr_class', SubProcessManager))
        mgr_kwargs: Dict[str, Any] = sim_config.get('mgr_kwargs', {})

        cancel_timeout = sim_config.get('cancel_timeout_ms', 10000) / 1e3

        self._manager: SubProcessManager = mgr_class(max_workers=sim_config.get('max_workers', 0),
                                                     cancel_timeout=cancel_timeout, **mgr_kwargs)

    @property
    def manager(self) -> SubProcessManager:
        return self._manager
