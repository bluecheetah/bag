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

"""This module defines DbAccess, the base class for CAD database manipulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Sequence, Any, Mapping

import abc
import importlib
import traceback
from pathlib import Path

from ..io.file import make_temp_dir, open_file, read_file
from ..verification import make_checker
from ..layout.routing.grid import RoutingGrid
from ..concurrent.core import batch_async_task
from .base import InterfaceBase

if TYPE_CHECKING:
    from .zmqwrapper import ZMQDealer
    from ..verification import Checker


def dict_to_item_list(table: Dict[str, Any]) -> List[List[str]]:
    """Given a Python dictionary, convert to sorted item list.

    Parameters
    ----------
    table :
        a Python dictionary where the keys are strings.

    Returns
    -------
    assoc_list :
        the sorted item list representation of the given dictionary.
    """
    return [[key, table[key]] for key in sorted(table.keys())]


def format_inst_map(inst_map: Dict[str, Any]) -> List[List[Any]]:
    """Given instance map from DesignModule, format it for database changes.

    Parameters
    ----------
    inst_map :
        the instance map created by DesignModule.

    Returns
    -------
    ans :
        the database change instance map.
    """
    ans = []
    for old_inst_name, rinst_list in inst_map.items():
        new_rinst_list = [dict(name=rinst['name'],
                               lib_name=rinst['lib_name'],
                               cell_name=rinst['cell_name'],
                               params=dict_to_item_list(rinst['params']),
                               term_mapping=dict_to_item_list(rinst['term_mapping']),
                               ) for rinst in rinst_list]
        ans.append([old_inst_name, new_rinst_list])
    return ans


class DbAccess(InterfaceBase, abc.ABC):
    """A class that manipulates the CAD database.

    Parameters
    ----------
    dealer : Optional[ZMQDealer]
        an optional socket that can be used to communicate with the CAD database.
    tmp_dir : str
        temporary file directory for DbAccess.
    db_config : Dict[str, Any]
        the database configuration dictionary.
    lib_defs_file : str
        name of the file that contains generator library names.
    """

    def __init__(self, dealer: ZMQDealer, tmp_dir: str, db_config: Dict[str, Any],
                 lib_defs_file: str) -> None:
        InterfaceBase.__init__(self)

        self.handler: ZMQDealer = dealer
        self.tmp_dir: str = make_temp_dir('dbTmp', parent_dir=tmp_dir)
        self.db_config: Dict[str, Any] = db_config
        # noinspection PyBroadException
        try:
            check_kwargs = self.db_config['checker'].copy()
            check_kwargs['tmp_dir'] = self.tmp_dir
            self.checker: Optional[Checker] = make_checker(**check_kwargs)
        except Exception:
            stack_trace = traceback.format_exc()
            print('*WARNING* error creating Checker:\n%s' % stack_trace)
            print('*WARNING* LVS/RCX will be disabled.')
            self.checker: Optional[Checker] = None

        # set default lib path
        self._default_lib_path: str = self.get_default_lib_path(db_config)

        # get yaml path mapping
        self.lib_path_map: Dict[str, str] = {}
        with open_file(lib_defs_file, 'r') as f:
            for line in f:
                lib_name = line.strip()
                self.add_sch_library(lib_name)

        self._close_all_cv: bool = db_config.get('close_all_cellviews', True)

    @classmethod
    def get_default_lib_path(cls, db_config: Dict[str, Any]) -> str:
        default_lib_path = Path(db_config.get('default_lib_path', '.'))

        if not default_lib_path.is_dir():
            default_lib_path = Path.cwd()

        return str(default_lib_path.resolve())

    @property
    def default_lib_path(self) -> str:
        """str: The default directory to create new libraries in.
        """
        return self._default_lib_path

    @property
    def has_bag_server(self) -> bool:
        """bool: True if the BAG server is up."""
        return self.handler is not None

    @abc.abstractmethod
    def get_exit_object(self) -> Any:
        """Returns an object to send to the server to shut it down.

        Return None if this option is not supported.
        """
        return None

    @abc.abstractmethod
    def get_cells_in_library(self, lib_name: str) -> List[str]:
        """Get a list of cells in the given library.

        Returns an empty list if the given library does not exist.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        cell_list : List[str]
            a list of cells in the library
        """
        return []

    @abc.abstractmethod
    def create_library(self, lib_name: str, lib_path: str = '') -> None:
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : str
            the library name.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        pass

    @abc.abstractmethod
    def configure_testbench(self, tb_lib: str, tb_cell: str
                            ) -> Tuple[str, List[str], Dict[str, str], Dict[str, str]]:
        """Update testbench state for the given testbench.

        This method fill in process-specific information for the given testbench.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.

        Returns
        -------
        cur_env : str
            the current simulation environment.
        envs : List[str]
            a list of available simulation environments.
        parameters : Dict[str, str]
            a list of testbench parameter values, represented as string.
        outputs : Dict[str, str]
            a dictionary of output expressions
        """
        return "", [], {}, {}

    @abc.abstractmethod
    def get_testbench_info(self, tb_lib: str, tb_cell: str
                           ) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
        """Returns information about an existing testbench.

        Parameters
        ----------
        tb_lib : str
            testbench library.
        tb_cell : str
            testbench cell.

        Returns
        -------
        cur_envs : List[str]
            the current simulation environments.
        envs : List[str]
            a list of available simulation environments.
        parameters : Dict[str, str]
            a list of testbench parameter values, represented as string.
        outputs : Dict[str, str]
            a list of testbench output expressions.
        """
        return [], [], {}, {}

    @abc.abstractmethod
    def update_testbench(self, lib: str, cell: str, parameters: Dict[str, str],
                         sim_envs: Sequence[str], config_rules: Sequence[List[str]],
                         env_parameters: Sequence[List[Tuple[str, str]]]) -> None:
        """Update the given testbench configuration.

        Parameters
        ----------
        lib : str
            testbench library.
        cell : str
            testbench cell.
        parameters : Dict[str, str]
            testbench parameters.
        sim_envs : Sequence[str]
            list of enabled simulation environments.
        config_rules : Sequence[List[str]]
            config view mapping rules, list of (lib, cell, view) rules.
        env_parameters : Sequence[List[Tuple[str, str]]]
            list of param/value list for each simulation environment.
        """
        pass

    @abc.abstractmethod
    def instantiate_layout_pcell(self, lib_name: str, cell_name: str, view_name: str,
                                 inst_lib: str, inst_cell: str, params: Dict[str, Any],
                                 pin_mapping: Dict[str, str]) -> None:
        """Create a layout cell with a single pcell instance.

        Parameters
        ----------
        lib_name : str
            layout library name.
        cell_name : str
            layout cell name.
        view_name : str
            layout view name, default is "layout".
        inst_lib : str
            pcell library name.
        inst_cell : str
            pcell cell name.
        params : Dict[str, Any]
            the parameter dictionary.
        pin_mapping: Dict[str, str]
            the pin mapping dictionary.
        """
        pass

    @abc.abstractmethod
    def create_schematics(self, lib_name: str, sch_view: str, sym_view: str,
                          content_list: Sequence[Any]) -> None:
        """Create the given schematics in CAD database.

        Precondition: the library already exists, all cellviews are writable (i.e. they have been
        closed already).

        Parameters
        ----------
        lib_name : str
            name of the new library to put the concrete schematics.
        sch_view : str
            schematic view name.
        sym_view : str
            symbol view name.
        content_list : Sequence[Any]
            list of schematics to create.
        """
        pass

    @abc.abstractmethod
    def create_layouts(self, lib_name: str, view: str, content_list: Sequence[Any]) -> None:
        """Create the given layouts in CAD database.

        Precondition: the library already exists, all cellviews are writable (i.e. they have been
        closed already).

        Parameters
        ----------
        lib_name : str
            name of the new library to put the concrete schematics.
        view : str
            layout view name.
        content_list : Sequence[Any]
            list of layouts to create.
        """
        pass

    @abc.abstractmethod
    def close_all_cellviews(self) -> None:
        """Close all currently opened cellviews in the database."""
        pass

    @abc.abstractmethod
    def release_write_locks(self, lib_name: str, cell_view_list: Sequence[Tuple[str, str]]) -> None:
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        pass

    @abc.abstractmethod
    def refresh_cellviews(self, lib_name: str, cell_view_list: Sequence[Tuple[str, str]]) -> None:
        """Refresh the given cellviews in the database.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        pass

    @abc.abstractmethod
    def perform_checks_on_cell(self, lib_name: str, cell_name: str, view_name: str) -> None:
        """Perform checks on the given cell.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        view_name : str
            the view name.
        """
        pass

    @abc.abstractmethod
    def create_schematic_from_netlist(self, netlist: str, lib_name: str, cell_name: str,
                                      sch_view: str = '', **kwargs: Any) -> None:
        """Create a schematic from a netlist.

        This is mainly used to create extracted schematic from an extracted netlist.

        Parameters
        ----------
        netlist : str
            the netlist file name.
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : str
            schematic view name.  The default value is implemendation dependent.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        pass

    @abc.abstractmethod
    def create_verilog_view(self, verilog_file: str, lib_name: str, cell_name: str, **kwargs: Any
                            ) -> None:
        """Create a verilog view for mix-signal simulation.

        Parameters
        ----------
        verilog_file : str
            the verilog file name.
        lib_name : str
            library name.
        cell_name : str
            cell name.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        pass

    @abc.abstractmethod
    def import_sch_cellview(self, lib_name: str, cell_name: str, view_name: str) -> None:
        """Recursively import the given schematic and symbol cellview.

        Parameters
        ----------
        lib_name : str
        library name.
        cell_name : str
            cell name.
        view_name : str
            view name.
        """
        pass

    @abc.abstractmethod
    def import_design_library(self, lib_name: str, view_name: str) -> None:
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        view_name : str
            the view name to import from the library.
        """
        pass

    @abc.abstractmethod
    def import_gds_file(self, gds_fname: str, lib_name: str, layer_map: str, obj_map: str,
                        grid: RoutingGrid) -> None:
        pass

    def send(self, obj: Any) -> Any:
        """Send the given Python object to the server, and return result."""
        if self.handler is None:
            raise Exception('BAG Server is not set up.')

        self.handler.send_obj(obj)
        reply = self.handler.recv_obj()
        return reply

    def close(self) -> None:
        """Terminate the database server gracefully.
        """
        if self.handler is not None:
            exit_obj = self.get_exit_object()
            if exit_obj is not None:
                self.handler.send(exit_obj)
            self.handler.close()
            self.handler = None

    def get_python_template(self, lib_name: str, cell_name: str, primitive_table: Dict[str, str]
                            ) -> str:
        """Returns the default Python Module template for the given schematic.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        primitive_table : Dict[str, str]
            a dictionary from primitive cell name to module template file name.

        Returns
        -------
        template : str
            the default Python Module template.
        """
        param_dict = dict(lib_name=lib_name, cell_name=cell_name)
        if lib_name == 'BAG_prim':
            if cell_name in primitive_table:
                # load template from user defined file
                template = self._tmp_env.from_string(read_file(primitive_table[cell_name]))
                return template.render(**param_dict)
            else:
                if cell_name.startswith('nmos4_') or cell_name.startswith('pmos4_'):
                    # transistor template
                    module_name = 'MosModuleBase'
                elif cell_name.startswith('ndio_') or cell_name.startswith('pdio_'):
                    # diode template
                    module_name = 'DiodeModuleBase'
                elif cell_name.startswith('res_metal_'):
                    module_name = 'ResMetalModule'
                elif cell_name.startswith('res_'):
                    # physical resistor template
                    module_name = 'ResPhysicalModuleBase'
                elif cell_name.startswith('esd_'):
                    # static esd template
                    module_name = 'ESDModuleBase'
                else:
                    raise Exception('Unknown primitive cell: %s' % cell_name)

                param_dict['module_name'] = module_name
                return self.render_file_template('PrimModule.pyi', param_dict)
        else:
            # use default empty template.
            return self.render_file_template('Module.pyi', param_dict)

    def instantiate_schematic(self, lib_name: str, content_list: Sequence[Any], lib_path: str = '',
                              sch_view: str = 'schematic', sym_view: str = 'symbol') -> None:
        """Create the given schematics in CAD database.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the concrete schematics.
        content_list : Sequence[Any]
            list of schematics to create.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        sch_view : str
            schematic view name.
        sym_view : str
            symbol view name.
        """
        cell_view_list = []
        if self._close_all_cv:
            self.close_all_cellviews()
        else:
            for cell_name, _ in content_list:
                cell_view_list.append((cell_name, sch_view))
                cell_view_list.append((cell_name, sym_view))
            self.release_write_locks(lib_name, cell_view_list)

        self.create_library(lib_name, lib_path=lib_path)
        self.create_schematics(lib_name, sch_view, sym_view, content_list)

        if cell_view_list:
            self.refresh_cellviews(lib_name, cell_view_list)

    def instantiate_layout(self, lib_name: str, content_list: Sequence[Any], lib_path: str = '',
                           view: str = 'layout') -> None:
        """Create a batch of layouts.

        Parameters
        ----------
        lib_name : str
            layout library name.
        content_list : Sequence[Any]
            list of layouts to create
        lib_path : str
            the path to create the library in.  If empty, use default location.
        view : str
            layout view name.
        """
        cell_view_list = []
        if self._close_all_cv:
            self.close_all_cellviews()
        else:
            for cell_name, _ in content_list:
                cell_view_list.append((cell_name, view))
            self.release_write_locks(lib_name, cell_view_list)

        self.create_library(lib_name, lib_path=lib_path)
        self.create_layouts(lib_name, view, content_list)

        if cell_view_list:
            self.refresh_cellviews(lib_name, cell_view_list)

    def run_drc(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """Run DRC on the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : bool
            True if DRC succeeds.
        log_fname : str
            name of the DRC log file.
        """
        coro = self.async_run_drc(lib_name, cell_name, **kwargs)
        results = batch_async_task([coro])
        if results is None:
            return False, ''

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    def run_lvs(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """Run LVS on the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        coro = self.async_run_lvs(lib_name, cell_name, **kwargs)
        results = batch_async_task([coro])
        if results is None:
            return False, ''

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    def run_rcx(self, lib_name: str, cell_name: str,
                params: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
        """run RC extraction on the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        params : Optional[Dict[str, Any]]
            optional RCX parameter values.

        Returns
        -------
        netlist : str
            The RCX netlist file name.  empty if RCX failed.
        log_fname : str
            RCX log file name.
        """
        coro = self.async_run_rcx(lib_name, cell_name, params=params)
        results = batch_async_task([coro])
        if results is None:
            return '', ''

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    def export_layout(self, lib_name: str, cell_name: str, out_file: str, **kwargs: Any) -> str:
        """Export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        log_fname : str
            log file name.  Empty if task cancelled.
        """
        coro = self.async_export_layout(lib_name, cell_name, out_file, **kwargs)
        results = batch_async_task([coro])
        if results is None:
            return ''

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    def export_schematic(self, lib_name: str, cell_name: str, out_file: str, **kwargs: Any) -> str:
        """Export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        log_fname : str
            log file name.  Empty if task cancelled.
        """
        coro = self.async_export_schematic(lib_name, cell_name, out_file, **kwargs)
        results = batch_async_task([coro])
        if results is None:
            return ''

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    async def async_run_drc(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """A coroutine for running DRC.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        value : bool
            True if DRC succeeds
        log_fname : str
            name of the DRC log file.
        """
        if self.checker is None:
            raise Exception('DRC/LVS/RCX is disabled.')
        return await self.checker.async_run_drc(lib_name, cell_name, **kwargs)

    async def async_run_lvs(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """A coroutine for running LVS.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        if self.checker is None:
            raise Exception('DRC/LVS/RCX is disabled.')
        return await self.checker.async_run_lvs(lib_name, cell_name, **kwargs)

    async def async_run_rcx(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[str, str]:
        """A coroutine for running RCX.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        netlist : str
            The RCX netlist file name.  empty if RCX failed.
        log_fname : str
            RCX log file name.
        """
        if self.checker is None:
            raise Exception('DRC/LVS/RCX is disabled.')
        return await self.checker.async_run_rcx(lib_name, cell_name, **kwargs)

    async def async_export_layout(self, lib_name: str, cell_name: str,
                                  out_file: str, **kwargs: Any) -> str:
        """Export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        log_fname : str
            log file name.  Empty if task cancelled.
        """
        if self.checker is None:
            raise Exception('layout export is disabled.')

        return await self.checker.async_export_layout(lib_name, cell_name, out_file, **kwargs)

    async def async_export_schematic(self, lib_name: str, cell_name: str,
                                     out_file: str, **kwargs: Any) -> str:
        if self.checker is None:
            raise Exception('schematic export is disabled.')

        return await self.checker.async_export_schematic(lib_name, cell_name, out_file, **kwargs)

    def add_sch_library(self, lib_name: str) -> Path:
        try:
            lib_module = importlib.import_module(lib_name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f'Cannot find python package {lib_name}.  '
                                      'You can only add schematic library if the corresponding '
                                      'package is on your PYTHONPATH')
        if not hasattr(lib_module, '__file__'):
            raise ImportError(f'{lib_name} is not a normal python package '
                              '(no __file__ attribute). Did you create a proper python '
                              'schematic library?')
        if lib_module.__file__ is None:
            raise ImportError(f'{lib_name} has None __file__ attribute.  Did you create a proper '
                              'python schematic library?')

        lib_path: Path = Path(lib_module.__file__).parent
        sch_lib_path = lib_path / 'schematic'
        if not sch_lib_path.is_dir():
            sch_lib_path.mkdir()
            init_file = sch_lib_path / '__init__.py'
            with open_file(init_file, 'w'):
                pass

        netlist_info_path = sch_lib_path / 'netlist_info'
        if not netlist_info_path.is_dir():
            netlist_info_path.mkdir()

        sch_lib_path = sch_lib_path.resolve()
        self.lib_path_map[lib_name] = str(sch_lib_path)
        return sch_lib_path

    def exclude_model(self, lib_name: str, cell_name: str) -> bool:
        """True to exclude the given schematic generator when generating behavioral models."""
        sch_config = self.db_config['schematic']
        lib_list = sch_config.get('model_exclude_libraries', None)
        lib_cell_dict = sch_config.get('model_exclude_cells', None)

        if lib_list and lib_name in lib_list:
            return True
        if lib_cell_dict:
            cell_list = lib_cell_dict.get(lib_name, None)
            return cell_list and cell_name in cell_list
        return False
