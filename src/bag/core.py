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

"""This is the core bag module.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING, Dict, Any, Tuple, Optional, Type, Sequence, Union, Mapping, cast, List
)

import os
import shutil
import pprint
from pathlib import Path

from pybag.enum import DesignOutput, SupplyWrapMode, LogLevel
from pybag.core import PySchCellViewInfo

from .io.file import write_yaml, read_yaml
from .interface import ZMQDealer
from .interface.lef import LEFInterface
from .design.netlist import add_mismatch_offsets
from .design.database import ModuleDB
from .design.module import Module
from .layout.routing.grid import RoutingGrid
from .layout.template import TemplateDB, TemplateBase
from .layout.tech import TechInfo
from .concurrent.core import batch_async_task
from .env import (
    get_port_number, get_bag_config, get_bag_work_dir, create_routing_grid, get_bag_tmp_dir,
    get_gds_layer_map, get_gds_object_map
)
from .util.importlib import import_class
from .simulation.data import netlist_info_from_dict
from .simulation.hdf5 import load_sim_data_hdf5
from .simulation.core import TestbenchManager
from .simulation.core import MeasurementManager as MeasurementManagerOld
from .simulation.measure import MeasurementManager
from .simulation.cache import SimulationDB, DesignDB

if TYPE_CHECKING:
    from .simulation.base import SimAccess


class BagProject:
    """The main bag controller class.

    This class mainly stores all the user configurations, and issue
    high level bag commands.

    Attributes
    ----------
    bag_config : Dict[str, Any]
        the BAG configuration parameters dictionary.
    """

    def __init__(self) -> None:
        self.bag_config = get_bag_config()

        bag_tmp_dir = get_bag_tmp_dir()
        bag_work_dir = get_bag_work_dir()

        # get port files
        port, msg = get_port_number(bag_config=self.bag_config)
        if msg:
            print(f'*WARNING* {msg}.  Operating without Virtuoso.')

        # create ZMQDealer object
        dealer_kwargs = {}
        dealer_kwargs.update(self.bag_config['socket'])
        del dealer_kwargs['port_file']

        # create TechInfo instance
        self._grid = create_routing_grid()

        if port >= 0:
            # make DbAccess instance.
            dealer = ZMQDealer(port, **dealer_kwargs)
        else:
            dealer = None

        # create database interface object
        try:
            lib_defs_file = os.path.join(bag_work_dir, self.bag_config['lib_defs'])
        except ValueError:
            lib_defs_file = ''
        db_cls = cast(Type['DbAccess'], import_class(self.bag_config['database']['class']))
        self.impl_db = db_cls(dealer, bag_tmp_dir, self.bag_config['database'], lib_defs_file)
        self._default_lib_path = self.impl_db.default_lib_path

        # make SimAccess instance.
        sim_cls = cast(Type['SimAccess'], import_class(self.bag_config['simulation']['class']))
        self._sim = sim_cls(bag_tmp_dir, self.bag_config['simulation'])

        # make LEFInterface instance
        self._lef: Optional[LEFInterface] = None
        lef_config = self.bag_config.get('lef', None)
        if lef_config is not None:
            lef_cls = cast(Type[LEFInterface], import_class(lef_config['class']))
            self._lef = lef_cls(lef_config)

    @property
    def tech_info(self) -> TechInfo:
        """TechInfo: the TechInfo object."""
        return self._grid.tech_info

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: the global routing grid object."""
        return self._grid

    @property
    def default_lib_path(self) -> str:
        return self._default_lib_path

    @property
    def sim_access(self) -> SimAccess:
        return self._sim

    def close_bag_server(self) -> None:
        """Close the BAG database server."""
        self.impl_db.close()
        self.impl_db = None

    def import_sch_cellview(self, lib_name: str, cell_name: str,
                            view_name: str = 'schematic') -> None:
        """Import the given schematic and symbol template into Python.

        This import process is done recursively.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        view_name : str
            view name.
        """
        self.impl_db.import_sch_cellview(lib_name, cell_name, view_name)

    def import_design_library(self, lib_name, view_name='schematic'):
        # type: (str, str) -> None
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        view_name : str
            the view name to import from the library.
        """
        self.impl_db.import_design_library(lib_name, view_name)

    def import_gds_file(self, gds_fname: str, lib_name: str) -> None:
        lay_map = get_gds_layer_map()
        obj_map = get_gds_object_map()
        self.impl_db.create_library(lib_name)
        self.impl_db.import_gds_file(gds_fname, lib_name, lay_map, obj_map, self.grid)

    def get_cells_in_library(self, lib_name):
        # type: (str) -> Sequence[str]
        """Get a list of cells in the given library.

        Returns an empty list if the given library does not exist.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        cell_list : Sequence[str]
            a list of cells in the library
        """
        return self.impl_db.get_cells_in_library(lib_name)

    def make_template_db(self, impl_lib: str, **kwargs: Any) -> TemplateDB:
        """Create and return a new TemplateDB instance.

        Parameters
        ----------
        impl_lib : str
            the library name to put generated layouts in.
        **kwargs : Any
            optional TemplateDB parameters.
        """
        return TemplateDB(self.grid, impl_lib, prj=self, **kwargs)

    def make_module_db(self, impl_lib: str, **kwargs: Any) -> ModuleDB:
        """Create and return a new ModuleDB instance.

        Parameters
        ----------
        impl_lib : str
            the library name to put generated layouts in.
        **kwargs : Any
            optional ModuleDB parameters.
        """
        return ModuleDB(self.tech_info, impl_lib, prj=self, **kwargs)

    def make_dsn_db(self, root_dir: Path, log_file: str, impl_lib: str,
                    sch_db: Optional[ModuleDB] = None, lay_db: Optional[TemplateDB] = None,
                    **kwargs: Any) -> DesignDB:
        if sch_db is None:
            sch_db = self.make_module_db(impl_lib)
        if lay_db is None:
            lay_db = self.make_template_db(impl_lib)

        dsn_db = DesignDB(root_dir, log_file, self.impl_db, self.sim_access.netlist_type,
                          sch_db, lay_db, **kwargs)
        return dsn_db

    def make_sim_db(self, dsn_dir: Path, log_file: str, impl_lib: str,
                    dsn_options: Optional[Mapping[str, Any]] = None,
                    **kwargs: Any) -> SimulationDB:
        if dsn_options is None:
            dsn_options = {}

        dsn_db = self.make_dsn_db(dsn_dir, log_file, impl_lib, **dsn_options)
        sim_db = SimulationDB(log_file, dsn_db, **kwargs)
        return sim_db

    def generate_cell(self, specs: Dict[str, Any],
                      raw: bool = False,
                      gen_lay: bool = True,
                      gen_sch: bool = True,
                      run_drc: bool = False,
                      run_lvs: bool = False,
                      run_rcx: bool = False,
                      lay_db: Optional[TemplateDB] = None,
                      sch_db: Optional[ModuleDB] = None,
                      gen_lef: bool = False,
                      cv_info_out: Optional[List[PySchCellViewInfo]] = None,
                      sim_netlist: bool = False,
                      flat: bool = False,
                      gen_hier: bool = False,
                      gen_model: bool = False,
                      mismatch: bool = False,
                      gen_shell: bool = False,
                      export_lay: bool = False,
                      gen_netlist: bool = False) -> str:
        """Generate layout/schematic of a given cell from specification file.

        Parameters
        ----------
        specs : Dict[str, Any]
            the specification dictionary.  Some non-obvious parameters:

            params : Dict[str, Any]
                If layout generator is given, this is the layout parameters.  Otherwise this
                is the schematic parameters.
            netlist_file : str
                If not empty, we will create a netlist file with this name (even if raw = False).
                if sim_netlist is True, this will be a simulation netlist.
            model_file : str
                the behavioral model filename if gen_model = True.
            gds_file : str
                override the default GDS layout file name.  Note that specifying this entry does
                not mean a GDS file will be created, you must set raw = True or gen_gds = True.

        raw : bool
            True to generate GDS and netlist files instead of OA cellviews.
        gen_lay : bool
            True to generate layout.
        gen_sch : bool
            True to generate schematics.
        run_drc : bool
            True to run DRC.
        run_lvs : bool
            True to run LVS.
        run_rcx : bool
            True to run RCX.
        lay_db : Optional[TemplateDB]
            the layout database.
        sch_db : Optional[ModuleDB]
            the schematic database.
        gen_lef : bool
            True to create LEF file.
        cv_info_out : Optional[List[PySchCellViewInfo]] = None
            If given cellview information objects will be appended to this list.
        sim_netlist : bool
            True to return a simulation netlist.
        flat : bool
            True to generate flat netlist.
        gen_hier: bool
            True to write the system verilog modeling hierarchy in a yaml file.
        gen_model: bool
            True to generate behavioral models
        mismatch : bool
            True to add mismatch voltages
        gen_shell: bool
            True to generate verilog shell file.
        export_lay: bool
            True to export layout file even in non-raw mode.
        gen_netlist : bool
            True to generate netlist even in non-raw mode.
        Returns
        -------
        rcx_netlist : str
            the extraction netlist.  Empty on error or if extraction is not run.
        """
        root_dir: Union[str, Path] = specs.get('root_dir', '')
        dut_str: Union[str, Type[TemplateBase], Type[Module]] = specs.get('dut_class') or specs.get('lay_class', '')
        sch_str: Union[str, Type[Module]] = specs.get('sch_class', '')
        dut_cls = import_class(dut_str)
        if issubclass(dut_cls, TemplateBase):
            lay_cls = dut_cls
            has_lay = True
        else:
            lay_cls = None
            sch_str = dut_cls
            has_lay = False
        impl_lib: str = specs['impl_lib']
        impl_cell: str = specs['impl_cell']
        params: Optional[Mapping[str, Any]] = specs.get('params', None)
        netlist_file_override: str = specs.get('netlist_file', '')
        model_file: str = specs.get('model_file', '')
        yaml_file: str = specs.get('yaml_file', '')
        layout_file_override: str = specs.get('layout_file', '')
        leaves: Optional[Mapping[str, List[str]]] = specs.get('leaf_cells', None)
        mod_type_str: str = specs.get('model_type', 'SYSVERILOG')
        default_model_view: str = specs.get('default_model_view', '')
        hierarchy_file: str = specs.get('hierarchy_file', '')
        model_params: Mapping[str, Any] = specs.get('model_params', {})
        sup_wrap_mode: str = specs.get('model_supply_wrap_mode', 'NONE')
        lef_config: Mapping[str, Any] = specs.get('lef_config', {})
        name_prefix: str = specs.get('name_prefix', '')
        name_suffix: str = specs.get('name_suffix', '')
        exact_cell_names_list: List[str] = specs.get('exact_cell_names', [])
        square_bracket: bool = specs.get('square_bracket', False)
        lay_type_specs: Union[str, List[str]] = specs.get('layout_type', 'GDS')
        mod_type: DesignOutput = DesignOutput[mod_type_str]
        sup_wrap_type: SupplyWrapMode = SupplyWrapMode[sup_wrap_mode]
        exact_cell_names = set(exact_cell_names_list)

        if isinstance(lay_type_specs, str):
            lay_type_list: List[DesignOutput] = [DesignOutput[lay_type_specs]]
        else:
            lay_type_list: List[DesignOutput] = [DesignOutput[v] for v in lay_type_specs]

        if isinstance(root_dir, str):
            root_path = Path(root_dir)
        else:
            root_path = root_dir

        gen_lay = gen_lay and has_lay
        gen_model = gen_model and model_params
        run_drc = run_drc and gen_lay

        verilog_shell_path = root_path / f'{impl_cell}_shell.v' if gen_lef or gen_shell else None
        sch_cls = None
        layout_ext = lay_type_list[0].extension
        layout_file = ''
        lef_options = {}
        if layout_file_override and Path(layout_file_override).suffix[1:] != layout_ext:
            raise ValueError('Conflict between layout file type and layout file name.')
        if has_lay:
            if lay_db is None:
                lay_db = self.make_template_db(impl_lib, name_prefix=name_prefix,
                                               name_suffix=name_suffix)

            print('computing layout...')
            lay_master: TemplateBase = lay_db.new_template(lay_cls, params=params)
            lay_master.get_lef_options(lef_options, lef_config)
            # try getting schematic class from instance, if possible
            sch_cls = lay_master.get_schematic_class_inst()
            dut_list = [(lay_master, impl_cell)]
            print('computation done.')

            if gen_lay:
                print('creating layout...')
                if not raw:
                    lay_db.batch_layout(dut_list, output=DesignOutput.LAYOUT,
                                        exact_cell_names=exact_cell_names)
                else:
                    layout_file = (layout_file_override or
                                   str(root_path / f'{impl_cell}.{layout_ext}'))
                    lay_db.batch_layout(dut_list, output=lay_type_list[0], fname=layout_file,
                                        exact_cell_names=exact_cell_names,
                                        square_bracket=square_bracket)
                    for out_type in lay_type_list[1:]:
                        cur_file = str(root_path / f'{impl_cell}.{out_type.extension}')
                        lay_db.batch_layout(dut_list, output=out_type, fname=cur_file,
                                            exact_cell_names=exact_cell_names,
                                            square_bracket=square_bracket)

                print('layout done.')

            sch_params = lay_master.sch_params
        else:
            sch_params = params

        if export_lay and not raw:
            print('exporting layout')
            layout_file = (layout_file_override or
                           str(root_path / f'{impl_cell}.{layout_ext}'))
            export_params = dict(square_bracket=square_bracket,
                                 output_type=lay_type_list[0])
            self.impl_db.export_layout(impl_lib, impl_cell, layout_file,
                                       params=export_params)
            for out_type in lay_type_list[1:]:
                export_params['output_type'] = out_type
                cur_file = str(root_path / f'{impl_cell}.{out_type.extension}')
                self.impl_db.export_layout(impl_lib, impl_cell, cur_file,
                                           params=export_params)

        if sch_cls is None:
            if isinstance(sch_str, str):
                if sch_str:
                    # no schematic class from layout, try get it from string
                    sch_cls = cast(Type[Module], import_class(sch_str))
            else:
                sch_cls = sch_str
        has_sch = sch_cls is not None

        run_lvs = (run_lvs or run_rcx) and gen_lay and has_sch
        run_rcx = run_rcx and gen_lay and has_sch
        gen_sch = (gen_sch or gen_hier or gen_model or run_lvs or run_rcx) and has_sch
        flat = flat or (mismatch and not run_rcx)

        final_netlist = ''
        final_netlist_type = DesignOutput.CDL
        lvs_netlist = ''
        netlist_file = netlist_file_override
        if (gen_netlist or raw) and not netlist_file:
            if sim_netlist:
                ext = self._sim.netlist_type.extension
            else:
                ext = DesignOutput.CDL.extension
            netlist_file = str(root_path / f'{impl_cell}.{ext}')

        if gen_sch:
            if sch_db is None:
                sch_db = self.make_module_db(impl_lib, name_prefix=name_prefix,
                                             name_suffix=name_suffix)

            print('computing schematic...')
            sch_master: Module = sch_db.new_master(sch_cls, params=sch_params)
            sch_master.get_lef_options(lef_options, lef_config)
            dut_list = [(sch_master, impl_cell)]
            print('computation done.')

            if not raw:
                print('creating schematic...')
                sch_db.batch_schematic(dut_list, exact_cell_names=exact_cell_names)
                print('schematic done.')

            if yaml_file:
                sch_db.batch_schematic(dut_list, output=DesignOutput.YAML, fname=yaml_file,
                                       exact_cell_names=exact_cell_names)

            if netlist_file:
                print('creating netlist...')
                final_netlist = netlist_file
                if sim_netlist:
                    final_netlist_type = self._sim.netlist_type
                    if run_lvs:
                        lvs_netlist = str(root_path / f'{impl_cell}.{DesignOutput.CDL.extension}')
                        sch_db.batch_schematic(dut_list, output=DesignOutput.CDL,
                                               fname=lvs_netlist, cv_info_out=cv_info_out,
                                               flat=flat, exact_cell_names=exact_cell_names,
                                               square_bracket=square_bracket)
                        sch_db.batch_schematic(dut_list, output=final_netlist_type,
                                               fname=netlist_file, cv_info_out=cv_info_out,
                                               flat=flat, exact_cell_names=exact_cell_names)
                    else:
                        sch_db.batch_schematic(dut_list, output=final_netlist_type,
                                               fname=netlist_file, cv_info_out=cv_info_out,
                                               flat=flat, exact_cell_names=exact_cell_names)
                else:
                    final_netlist_type = DesignOutput.CDL
                    lvs_netlist = netlist_file
                    sch_db.batch_schematic(dut_list, output=final_netlist_type, fname=netlist_file,
                                           cv_info_out=cv_info_out, flat=flat,
                                           exact_cell_names=exact_cell_names,
                                           square_bracket=square_bracket)
                print('netlisting done.')

            if verilog_shell_path is not None:
                sch_db.batch_schematic(dut_list, output=DesignOutput.VERILOG, shell=True,
                                       fname=str(verilog_shell_path),
                                       exact_cell_names=exact_cell_names)
                print(f'verilog shell file created at {verilog_shell_path}')

            if gen_hier:
                print('creating hierarchy...')
                if not hierarchy_file:
                    hierarchy_file = str(root_path / 'hierarchy.yaml')
                write_yaml(hierarchy_file,
                           sch_master.get_instance_hierarchy(mod_type, leaves, default_model_view))
                print(f'hierarchy done. File is {hierarchy_file}')

            if gen_model:
                if not model_file:
                    model_file = str(root_path / f'{impl_cell}.{mod_type.extension}')
                print('creating behavioral model...')
                sch_db.batch_model([(sch_master, impl_cell, model_params)],
                                   output=mod_type, fname=model_file,
                                   supply_wrap_mode=sup_wrap_type,
                                   exact_cell_names=exact_cell_names)
                print(f'behavioral model done. File is {model_file}')
        elif netlist_file:
            if sim_netlist:
                raise ValueError('Cannot generate simulation netlist from custom cellview')

            print('exporting netlist')
            self.impl_db.export_schematic(impl_lib, impl_cell, netlist_file)

        if impl_cell in exact_cell_names:
            gen_cell_name = impl_cell
        else:
            gen_cell_name = name_prefix + impl_cell + name_suffix

        if run_drc:
            print('running DRC...')
            drc_passed, drc_log = self.run_drc(impl_lib, gen_cell_name, layout=layout_file)
            if drc_passed:
                print('DRC passed!')
            else:
                print(f'DRC failed... log file: {drc_log}')

        lvs_passed = False
        if run_lvs:
            print('running LVS...')
            lvs_passed, lvs_log = self.run_lvs(impl_lib, gen_cell_name, run_rcx=run_rcx,
                                               layout=layout_file, netlist=lvs_netlist)
            if lvs_passed:
                print('LVS passed!')
            else:
                raise ValueError(f'LVS failed... log file: {lvs_log}')

        if lvs_passed and run_rcx:
            print('running RCX...')
            final_netlist, rcx_log = self.run_rcx(impl_lib, gen_cell_name)
            final_netlist_type = DesignOutput.CDL
            if final_netlist:
                print('RCX passed!')
                if not raw:
                    root_path.mkdir(parents=True, exist_ok=True)
                if isinstance(final_netlist, list):
                    for f in final_netlist:
                        to_file = str(root_path / Path(f).name)
                        shutil.copy(f, to_file)
                        final_netlist = to_file
                else:
                    to_file = str(root_path / Path(final_netlist).name)
                    shutil.copy(final_netlist, to_file)
                    final_netlist = to_file
            else:
                raise ValueError(f'RCX failed... log file: {rcx_log}')

        if gen_lef:
            if not verilog_shell_path.is_file():
                raise ValueError(f'Missing verilog shell file: {verilog_shell_path}')

            lef_options = lef_config.get('lef_options_override', lef_options)
            print('generating LEF...')
            lef_path = root_path / f'{impl_cell}.lef'
            success = self.generate_lef(impl_lib, impl_cell, verilog_shell_path, lef_path,
                                        root_path, lef_options)
            if success:
                print(f'LEF generation done, file at {lef_path}')
            else:
                raise ValueError('LEF generation failed... '
                                 f'check log files in run directory: {root_path}')

        if mismatch:
            add_mismatch_offsets(final_netlist, final_netlist, final_netlist_type)

        return final_netlist

    def replace_dut_in_wrapper(self, params: Mapping[str, Any], dut_lib: str,
                               dut_cell: str) -> Mapping[str, Any]:
        # helper function that replaces dut_lib and dut_cell in the wrapper recursively in
        # dut_params
        ans = {k: v for k, v in params.items()}
        dut_params: Optional[Mapping[str, Any]] = params.get('dut_params', None)
        if dut_params is None:
            ans['dut_lib'] = dut_lib
            ans['dut_cell'] = dut_cell
        else:
            ans['dut_params'] = self.replace_dut_in_wrapper(dut_params, dut_lib, dut_cell)
        return ans

    def simulate_cell(self, specs: Dict[str, Any],
                      extract: bool = True,
                      gen_tb: bool = True,
                      simulate: bool = True,
                      mismatch: bool = False,
                      raw: bool = True,
                      lay_db: Optional[TemplateDB] = None,
                      sch_db: Optional[ModuleDB] = None,
                      ) -> str:
        """Generate and simulate a single design.

        This method only works for simulating a single cell (or a wrapper around a single cell).
        If you need to simulate multiple designs together, use simulate_config().

        Parameters
        ----------
        specs : Dict[str, Any]
            the specification dictionary.  Important entries are:

            use_netlist : str
                If specified, use this netlist file as the DUT netlist, and
                only generate the testbench and simulation netlists.
                If specified but the netlist does not exist, or the PySchCellViewInfo yaml file
                does not exist in the same directory, we will still generate the DUT,
                but the resulting netlist/PySchCellViewInfo object will be saved to this location

        extract : bool
            True to generate extracted netlist.
        gen_tb : bool
            True to generate the DUT/testbench/simulation netlists.

            If False, we will simply grab the final simulation netlist and simulate it.
            This means you can quickly simulate a previously generated netlist with manual
            modifications.
        simulate : bool
            True to run simulation.

            If False, we will only generate the netlists.
        mismatch: bool
            If True mismatch voltage sources are added to the netlist and simulation is done with
            those in place
        raw: bool
            True to generate GDS and netlist files instead of OA cellviews.
        lay_db : Optional[TemplateDB]
            the layout database.
        sch_db : Optional[ModuleDB]
            the schematic database.

        Returns
        -------
        sim_result : str
            simulation result file name.
        """
        root_dir: Union[str, Path] = specs['root_dir']
        impl_lib: str = specs['impl_lib']
        impl_cell: str = specs['impl_cell']
        use_netlist: str = specs.get('use_netlist', '')
        precision: int = specs.get('precision', 6)
        tb_params: Dict[str, Any] = specs.get('tb_params', {}).copy()
        wrapper_lib: str = specs.get('wrapper_lib', '')
        if wrapper_lib:
            wrapper_cell: str = specs['wrapper_cell']
            wrapper_params: Mapping[str, Any] = specs['wrapper_params']
            wrapper_params = self.replace_dut_in_wrapper(wrapper_params, impl_lib, impl_cell)
            tb_params['dut_params'] = wrapper_params
            tb_params['dut_lib'] = wrapper_lib
            tb_params['dut_cell'] = wrapper_cell
        else:
            tb_params['dut_lib'] = impl_lib
            tb_params['dut_cell'] = impl_cell

        if isinstance(root_dir, str):
            root_path = Path(root_dir).resolve()
        else:
            root_path = root_dir

        netlist_type = self._sim.netlist_type
        tb_netlist_path = root_path / f'tb.{netlist_type.extension}'
        sim_netlist_path = root_path / f'sim.{netlist_type.extension}'

        root_path.mkdir(parents=True, exist_ok=True)
        if gen_tb:
            if not impl_cell:
                raise ValueError('impl_cell is empty.')

            if sch_db is None:
                sch_db = self.make_module_db(impl_lib)

            if use_netlist:
                use_netlist_path = Path(use_netlist)
                netlist_dir: Path = use_netlist_path.parent
                cv_info_path = netlist_dir / (use_netlist_path.stem + '.cvinfo.yaml')
                if use_netlist_path.is_file():
                    if cv_info_path.is_file():
                        # both files exist, load from file
                        cvinfo = PySchCellViewInfo(str(cv_info_path))
                        cv_info_list = [cvinfo]
                    else:
                        # no cv_info, still need to generate
                        cv_info_list = []
                else:
                    # need to save netlist and cv_info_list
                    cv_info_list = []
            else:
                # no need to save
                use_netlist_path = None
                cv_info_path = None
                cv_info_list = []
            has_netlist = use_netlist_path is not None and use_netlist_path.is_file()
            extract = extract and not has_netlist
            if not cv_info_list:
                gen_netlist = self.generate_cell(specs, raw=raw, gen_lay=extract, gen_sch=True,
                                                 run_lvs=extract, run_rcx=extract,
                                                 sim_netlist=True, sch_db=sch_db, lay_db=lay_db,
                                                 cv_info_out=cv_info_list, mismatch=mismatch)
                if use_netlist_path is None:
                    use_netlist_path = Path(gen_netlist)
                else:
                    # save netlist and cvinfo
                    use_netlist_path.parent.mkdir(parents=True, exist_ok=True)
                    if not use_netlist_path.is_file():
                        shutil.copy(gen_netlist, str(use_netlist_path))
                    for cv_info in reversed(cv_info_list):
                        print(cv_info.lib_name, cv_info.cell_name)
                        if cv_info.lib_name == impl_lib and cv_info.cell_name == impl_cell:
                            cv_info.to_file(str(cv_info_path))
                            break

            tbm_str: Union[str, Type[TestbenchManager]] = specs.get('tbm_class', '')
            if isinstance(tbm_str, str):
                if tbm_str:
                    tbm_cls = cast(Type[TestbenchManager], import_class(tbm_str))
                else:
                    tbm_cls = None
            else:
                tbm_cls = tbm_str

            if tbm_cls is not None:
                # setup testbench using TestbenchManager
                tbm_specs: Dict[str, Any] = specs['tbm_specs']
                sim_envs: List[str] = tbm_specs['sim_envs']

                tbm = tbm_cls(self._sim, root_path, 'tb_sim', impl_lib,
                              tbm_specs, [], sim_envs, precision=precision)
                tbm.setup(sch_db, tb_params, cv_info_list, use_netlist_path, gen_sch=not raw)
            else:
                # setup testbench using spec file
                tb_lib: str = specs['tb_lib']
                tb_cell: str = specs['tb_cell']
                sim_info_dict: Dict[str, Any] = specs['sim_info']
                impl_cell_tb = f'{tb_cell.upper()}_{impl_cell}' if impl_cell else tb_cell.upper()

                tb_cls = sch_db.get_schematic_class(tb_lib, tb_cell)
                # noinspection PyTypeChecker
                tb_master = sch_db.new_master(tb_cls, params=tb_params)
                dut_list = [(tb_master, impl_cell_tb)]

                fname = '' if use_netlist_path is None else str(use_netlist_path)
                sch_db.batch_schematic(dut_list, output=netlist_type, top_subckt=False,
                                       fname=str(tb_netlist_path), cv_info_list=cv_info_list,
                                       cv_netlist=fname)
                if not raw:
                    sch_db.batch_schematic(dut_list, output=DesignOutput.SCHEMATIC)
                sim_info = netlist_info_from_dict(sim_info_dict)
                self._sim.create_netlist(sim_netlist_path, tb_netlist_path, sim_info, precision)
                tbm = None
        else:
            tbm = None

        sim_result = ''
        if simulate:
            if not tb_netlist_path.is_file():
                raise ValueError(f'Cannot find testbench netlist: {tb_netlist_path}')
            if not sim_netlist_path.is_file():
                raise ValueError(f'Cannot find simulation netlist: {sim_netlist_path}')

            sim_tag = 'sim'
            print(f'simulation netlist: {sim_netlist_path}')
            self._sim.run_simulation(sim_netlist_path, sim_tag)
            print(f'Finished simulating {sim_netlist_path}')
            sim_path = self._sim.get_sim_file(sim_netlist_path.parent, sim_tag)
            sim_result = str(sim_path)
            print(f'Simulation result in {sim_result}')
            if tbm is not None and specs.get('tbm_print', False):
                tbm.print_results(load_sim_data_hdf5(sim_path))

        return sim_result

    def measure_cell(self, specs: Mapping[str, Any], extract: bool = False,
                     force_sim: bool = False, force_extract: bool = False, gen_sch: bool = False,
                     fake: bool = False, log_level: LogLevel = LogLevel.DEBUG) -> None:
        meas_str: Union[str, Type[MeasurementManager]] = specs['meas_class']
        meas_name: str = specs['meas_name']
        meas_params: Dict[str, Any] = specs['meas_params']
        precision: int = specs.get('precision', 6)

        gen_specs_file: str = specs.get('gen_specs_file', '')
        if gen_specs_file:
            gen_specs: Mapping[str, Any] = read_yaml(gen_specs_file)
            dut_str: Union[str, Type[TemplateBase]] = gen_specs.get('dut_class') or gen_specs['lay_class']
            impl_lib: str = gen_specs['impl_lib']
            impl_cell: str = gen_specs['impl_cell']
            dut_params: Mapping[str, Any] = gen_specs['params']
            root_dir: Union[str, Path] = gen_specs['root_dir']
            meas_rel_dir: str = specs.get('meas_rel_dir', '')
        else:
            dut_str: Union[str, Type[TemplateBase]] = specs.get('dut_class') or specs['lay_class']
            impl_lib: str = specs['impl_lib']
            impl_cell: str = specs['impl_cell']
            dut_params: Mapping[str, Any] = specs['dut_params']
            root_dir: Union[str, Path] = specs['root_dir']
            meas_rel_dir: str = specs.get('meas_rel_dir', '')

        meas_cls = cast(Type[MeasurementManager], import_class(meas_str))
        dut_cls = import_class(dut_str)
        if isinstance(root_dir, str):
            root_path = Path(root_dir)
        else:
            root_path = root_dir
        if meas_rel_dir:
            meas_path = root_path / meas_rel_dir
        else:
            meas_path = root_path

        dsn_options = dict(
            extract=extract,
            force_extract=force_extract,
            gen_sch=gen_sch,
            log_level=log_level,
        )
        log_file = str(meas_path / 'meas.log')
        sim_db: SimulationDB = self.make_sim_db(root_path / 'dsn', log_file, impl_lib,
                                                dsn_options=dsn_options, force_sim=force_sim,
                                                precision=precision, log_level=log_level)

        dut = sim_db.new_design(impl_cell, dut_cls, dut_params, extract=extract)
        meas_params['fake'] = fake
        mm = sim_db.make_mm(meas_cls, meas_params)
        result = sim_db.simulate_mm_obj(meas_name, meas_path / meas_name, dut, mm)
        pprint.pprint(result.data)

    def measure_cell_old(self, specs: Dict[str, Any],
                         gen_dut: bool = True,
                         load_from_file: bool = False,
                         extract: bool = True,
                         mismatch: bool = False,
                         sch_db: Optional[ModuleDB] = None,
                         cv_info_list: Optional[List[PySchCellViewInfo]] = None,
                         ) -> Dict[str, Any]:
        """Generate and simulate a single design.

        This method only works for simulating a single cell (or a wrapper around a single cell).
        If you need to simulate multiple designs together, use simulate_config().

        Parameters
        ----------
        specs : Dict[str, Any]
           the specification dictionary.  Important entries are:

           use_netlist : str
               If specified, use this netlist file as the DUT netlist, and
               only generate the testbench and simulation netlists.

        gen_dut : bool
           True to generate DUT.
        load_from_file : bool
           True to load from file.
        extract : bool
           True to run extracted simulation.
        mismatch: bool
           If True mismatch voltage sources are added to the netlist and simulation is done with
           those in place
        sch_db : Optional[ModuleDB]
            the schematic database.
        cv_info_list: Optional[List[PySchCellViewInfo]]
           Optional cellview information objects.

        Returns
        -------
        meas_result : Dict[str, Any]
           measurement results dictionary.
        """
        root_dir: str = specs['root_dir']
        impl_lib: str = specs['impl_lib']
        impl_cell: str = specs['impl_cell']
        precision: int = specs.get('precision', 6)
        use_netlist: str = specs.get('use_netlist', None)
        mm_name: str = specs['meas_name']
        mm_str: Union[str, Type[MeasurementManagerOld]] = specs['meas_class']
        mm_specs: Dict[str, Any] = specs['meas_specs']

        gen_dut = (gen_dut or not load_from_file) and not use_netlist

        root_path = Path(root_dir).resolve()

        root_path.mkdir(parents=True, exist_ok=True)
        wrapper_lookup = {'': impl_cell}

        if cv_info_list is None:
            cv_info_list = []

        if gen_dut:
            if sch_db is None:
                sch_db = self.make_module_db(impl_lib)
            netlist = Path(self.generate_cell(specs, raw=True, gen_lay=extract, gen_sch=True,
                                              run_lvs=extract, run_rcx=extract, sim_netlist=True,
                                              sch_db=sch_db, cv_info_out=cv_info_list,
                                              mismatch=mismatch))
        else:
            netlist = use_netlist

        mm_cls = cast(Type[MeasurementManagerOld], import_class(mm_str))
        sim_envs: List[str] = mm_specs['sim_envs']
        mm = mm_cls(self._sim, root_path, mm_name, impl_lib,
                    mm_specs, wrapper_lookup, [], sim_envs, precision)

        result = mm.measure_performance(sch_db, cv_info_list, netlist,
                                        load_from_file=load_from_file, gen_sch=False)

        return result

    def create_library(self, lib_name, lib_path=''):
        # type: (str, str) -> None
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : str
            the library name.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        return self.impl_db.create_library(lib_name, lib_path=lib_path)

    def instantiate_schematic(self, lib_name, content_list, lib_path=''):
        # type: (str, Sequence[Any], str) -> None
        """Create the given schematic contents in CAD database.

        NOTE: this is BAG's internal method.  To create schematics, call batch_schematic() instead.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the schematic instances.
        content_list : Sequence[Any]
            list of schematics to create.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        """
        self.impl_db.instantiate_schematic(lib_name, content_list, lib_path=lib_path)

    def instantiate_layout_pcell(self, lib_name, cell_name, inst_lib, inst_cell, params,
                                 pin_mapping=None, view_name='layout'):
        # type: (str, str, str, str, Dict[str, Any], Optional[Dict[str, str]], str) -> None
        """Create a layout cell with a single pcell instance.

        Parameters
        ----------
        lib_name : str
            layout library name.
        cell_name : str
            layout cell name.
        inst_lib : str
            pcell library name.
        inst_cell : str
            pcell cell name.
        params : Dict[str, Any]
            the parameter dictionary.
        pin_mapping: Optional[Dict[str, str]]
            the pin renaming dictionary.
        view_name : str
            layout view name, default is "layout".
        """
        pin_mapping = pin_mapping or {}
        self.impl_db.instantiate_layout_pcell(lib_name, cell_name, view_name,
                                              inst_lib, inst_cell, params, pin_mapping)

    def instantiate_layout(self, lib_name, content_list, lib_path='', view='layout'):
        # type: (str, Sequence[Any], str, str) -> None
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
        self.impl_db.instantiate_layout(lib_name, content_list, lib_path=lib_path, view=view)

    def release_write_locks(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        self.impl_db.release_write_locks(lib_name, cell_view_list)

    def refresh_cellviews(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        """Refresh the given cellviews in the database.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        self.impl_db.refresh_cellviews(lib_name, cell_view_list)

    def perform_checks_on_cell(self, lib_name, cell_name, view_name):
        # type: (str, str, str) -> None
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
        self.impl_db.perform_checks_on_cell(lib_name, cell_name, view_name)

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
        return self.impl_db.run_drc(lib_name, cell_name, **kwargs)

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
        return self.impl_db.run_lvs(lib_name, cell_name, **kwargs)

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
        return self.impl_db.run_rcx(lib_name, cell_name, params=params)

    def generate_lef(self, impl_lib: str, impl_cell: str, verilog_path: Path,
                     lef_path: Path, run_path: Path, options: Dict[str, Any]) -> bool:
        if self._lef is None:
            raise ValueError('LEF generation interface not defined in bag_config.yaml')
        else:
            return self._lef.generate_lef(impl_lib, impl_cell, verilog_path, lef_path, run_path,
                                          **options)

    def export_layout(self, lib_name: str, cell_name: str, out_file: str, **kwargs: Any) -> str:
        """export layout.

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
        return self.impl_db.export_layout(lib_name, cell_name, out_file, **kwargs)

    def batch_export_layout(self, info_list):
        # type: (Sequence[Tuple[Any, ...]]) -> Optional[Sequence[str]]
        """Export layout of all given cells

        Parameters
        ----------
        info_list:
            list of cell information.  Each element is a tuple of:

            lib_name : str
                library name.
            cell_name : str
                cell name.
            out_file : str
                layout output file name.
            view_name : str
                layout view name.  Optional.
            params : Optional[Dict[str, Any]]
                optional export parameter values.

        Returns
        -------
        results : Optional[Sequence[str]]
            If task is cancelled, return None.  Otherwise, this is a
            list of log file names.
        """
        coro_list = [self.impl_db.async_export_layout(*info) for info in info_list]
        temp_results = batch_async_task(coro_list)
        if temp_results is None:
            return None
        return ['' if isinstance(val, Exception) else val for val in temp_results]

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
            LVS parameters should be specified as lvs_params.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        return await self.impl_db.async_run_lvs(lib_name, cell_name, **kwargs)

    async def async_run_rcx(self, lib_name: str, cell_name: str,
                            params: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """A coroutine for running RCX.

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
        return await self.impl_db.async_run_rcx(lib_name, cell_name, params=params)

    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view=None, **kwargs):
        # type: (str, str, str, Optional[str], **Any) -> None
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
        sch_view : Optional[str]
            schematic view name.  The default value is implemendation dependent.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        return self.impl_db.create_schematic_from_netlist(netlist, lib_name, cell_name,
                                                          sch_view=sch_view, **kwargs)

    def create_verilog_view(self, verilog_file, lib_name, cell_name, **kwargs):
        # type: (str, str, str, **Any) -> None
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
        verilog_file = os.path.abspath(verilog_file)
        if not os.path.isfile(verilog_file):
            raise ValueError('%s is not a file.' % verilog_file)

        return self.impl_db.create_verilog_view(verilog_file, lib_name, cell_name, **kwargs)

    def exclude_model(self, lib_name: str, cell_name: str) -> bool:
        """True to exclude the given schematic generator when generating behavioral models."""
        return self.impl_db.exclude_model(lib_name, cell_name)
