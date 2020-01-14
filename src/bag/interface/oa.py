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

"""This module implements all CAD database manipulations using OpenAccess plugins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, List, Dict, Optional, Any, Tuple

import os
import shutil
from pathlib import Path

from pybag.core import PyOADatabase, make_tr_colors

from ..io.file import write_file
from ..layout.routing.grid import RoutingGrid
from .database import DbAccess
from .skill import handle_reply

if TYPE_CHECKING:
    from .zmqwrapper import ZMQDealer


class OAInterface(DbAccess):
    """OpenAccess interface between bag and Virtuoso.
    """

    def __init__(self, dealer: ZMQDealer, tmp_dir: str, db_config: Dict[str, Any],
                 lib_defs_file: str) -> None:

        # Create PyOADatabase object before calling super constructor,
        # So that schematic library yaml path is added correctly.
        cds_lib_path: str = db_config.get('lib_def_path', '')
        if not cds_lib_path:
            cds_lib_path = str((Path(os.environ.get('CDSLIBPATH', '')) / 'cds.lib').resolve())

        self._oa_db = PyOADatabase(cds_lib_path)
        for lib_name in db_config['schematic']['exclude_libraries']:
            self._oa_db.add_primitive_lib(lib_name)
        # BAG_prim is always excluded
        self._oa_db.add_primitive_lib('BAG_prim')

        DbAccess.__init__(self, dealer, tmp_dir, db_config, lib_defs_file)

    def add_sch_library(self, lib_name: str) -> None:
        """Override; register yaml path in PyOADatabase too."""
        lib_path = DbAccess.add_sch_library(self, lib_name)
        self._oa_db.add_yaml_path(lib_name, str(lib_path / 'netlist_info'))

    def _eval_skill(self, expr: str, input_files: Optional[Dict[str, Any]] = None,
                    out_file: Optional[str] = None) -> str:
        """Send a request to evaluate the given skill expression.

        Because Virtuoso has a limit on the input/output data (< 4096 bytes),
        if your input is large, you need to write it to a file and have
        Virtuoso open the file to parse it.  Similarly, if you expect a
        large output, you need to make Virtuoso write the result to the
        file, then read it yourself.  The parameters input_files and
        out_file help you achieve this functionality.

        For example, if you need to evaluate "skill_fun(arg fname)", where
        arg is a file containing the list [1 2 3], and fname is the output
        file name, you will call this function with:

        expr = "skill_fun({arg} {fname})"
        input_files = { "arg": [1 2 3] }
        out_file = "fname"

        the bag server will then a temporary file for arg and fname, write
        the list [1 2 3] into the file for arg, call Virtuoso, then read
        the output file fname and return the result.

        Parameters
        ----------
        expr :
            the skill expression to evaluate.
        input_files :
            A dictionary of input files content.
        out_file :
            the output file name argument in expr.

        Returns
        -------
        result :
            a string representation of the result.

        Raises
        ------
        VirtuosoException :
            if virtuoso encounters errors while evaluating the expression.
        """
        request = dict(
            type='skill',
            expr=expr,
            input_files=input_files,
            out_file=out_file,
        )

        reply = self.send(request)
        return handle_reply(reply)

    def close(self) -> None:
        DbAccess.close(self)
        if self._oa_db is not None:
            self._oa_db.close()
            self._oa_db = None

    def get_exit_object(self) -> Any:
        return {'type': 'exit'}

    def get_cells_in_library(self, lib_name: str) -> List[str]:
        return self._oa_db.get_cells_in_lib(lib_name)

    def create_library(self, lib_name: str, lib_path: str = '') -> None:
        lib_path = lib_path or self.default_lib_path
        tech_lib = self.db_config['schematic']['tech_lib']
        self._oa_db.create_lib(lib_name, lib_path, tech_lib)

    def configure_testbench(self, tb_lib: str, tb_cell: str
                            ) -> Tuple[str, List[str], Dict[str, str], Dict[str, str]]:
        raise NotImplementedError('Not implemented yet.')

    def get_testbench_info(self, tb_lib: str, tb_cell: str
                           ) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
        raise NotImplementedError('Not implemented yet.')

    def update_testbench(self, lib: str, cell: str, parameters: Dict[str, str],
                         sim_envs: Sequence[str], config_rules: Sequence[List[str]],
                         env_parameters: Sequence[List[Tuple[str, str]]]) -> None:
        raise NotImplementedError('Not implemented yet.')

    def instantiate_layout_pcell(self, lib_name: str, cell_name: str, view_name: str,
                                 inst_lib: str, inst_cell: str, params: Dict[str, Any],
                                 pin_mapping: Dict[str, str]) -> None:
        raise NotImplementedError('Not implemented yet.')

    def create_schematics(self, lib_name: str, sch_view: str, sym_view: str,
                          content_list: Sequence[Any]) -> None:
        self._oa_db.implement_sch_list(lib_name, sch_view, sym_view, content_list)

    def create_layouts(self, lib_name: str, view: str, content_list: Sequence[Any]) -> None:
        self._oa_db.implement_lay_list(lib_name, view, content_list)

    def close_all_cellviews(self) -> None:
        if self.has_bag_server:
            self._eval_skill('close_all_cellviews()')

    def release_write_locks(self, lib_name: str, cell_view_list: Sequence[Tuple[str, str]]) -> None:
        if self.has_bag_server:
            cmd = 'release_write_locks( "%s" {cell_view_list} )' % lib_name
            in_files = {'cell_view_list': cell_view_list}
            self._eval_skill(cmd, input_files=in_files)

    def refresh_cellviews(self, lib_name: str, cell_view_list: Sequence[Tuple[str, str]]) -> None:
        if self.has_bag_server:
            cmd = 'refresh_cellviews( "%s" {cell_view_list} )' % lib_name
            in_files = {'cell_view_list': cell_view_list}
            self._eval_skill(cmd, input_files=in_files)

    def perform_checks_on_cell(self, lib_name: str, cell_name: str, view_name: str) -> None:
        self._eval_skill(
            'check_and_save_cell( "{}" "{}" "{}" )'.format(lib_name, cell_name, view_name))

    def create_schematic_from_netlist(self, netlist: str, lib_name: str, cell_name: str,
                                      sch_view: str = '', **kwargs: Any) -> None:
        # get netlists to copy
        netlist_dir: Path = Path(netlist).parent
        netlist_files = self.checker.get_rcx_netlists(lib_name, cell_name)
        if not netlist_files:
            # some error checking.  Shouldn't be needed but just in case
            raise ValueError('RCX did not generate any netlists')

        # copy netlists to a "netlist" subfolder in the CAD database
        cell_dir: Path = Path(self.get_cell_directory(lib_name, cell_name))
        targ_dir = cell_dir / 'netlist'
        targ_dir.mkdir(parents=True, exist_ok=True)
        for fname in netlist_files:
            # TODO: pycharm type-hint bug
            # noinspection PyTypeChecker
            shutil.copy(netlist_dir / fname, targ_dir)

        # create symbolic link as aliases
        symlink = targ_dir / 'netlist'
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(netlist_files[0])

    def get_cell_directory(self, lib_name: str, cell_name: str) -> str:
        """Returns the directory name of the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.

        Returns
        -------
        cell_dir : str
            path to the cell directory.
        """
        return str(Path(self._oa_db.get_lib_path(lib_name)) / cell_name)

    def create_verilog_view(self, verilog_file: str, lib_name: str, cell_name: str, **kwargs: Any
                            ) -> None:
        # delete old verilog view
        cmd = 'delete_cellview( "%s" "%s" "verilog" )' % (lib_name, cell_name)
        self._eval_skill(cmd)
        cmd = 'schInstallHDL("%s" "%s" "verilog" "%s" t)' % (lib_name, cell_name, verilog_file)
        self._eval_skill(cmd)

    def import_sch_cellview(self, lib_name: str, cell_name: str, view_name: str) -> None:
        if lib_name not in self.lib_path_map:
            self.add_sch_library(lib_name)

        # read schematic information
        cell_list = self._oa_db.read_sch_recursive(lib_name, cell_name, view_name)

        # create python templates
        self._create_sch_templates(cell_list)

    def import_design_library(self, lib_name: str, view_name: str) -> None:
        if lib_name not in self.lib_path_map:
            self.add_sch_library(lib_name)

        if lib_name == 'BAG_prim':
            # reading BAG primitives library, don't need to parse YAML files,
            # just get the cell list
            cell_list = [(lib_name, cell) for cell in self.get_cells_in_library(lib_name)]
        else:
            # read schematic information
            cell_list = self._oa_db.read_library(lib_name, view_name)

        # create python templates
        self._create_sch_templates(cell_list)

    def import_gds_file(self, gds_fname: str, lib_name: str, layer_map: str, obj_map: str,
                        grid: RoutingGrid) -> None:
        tr_colors = make_tr_colors(grid.tech_info)
        self._oa_db.import_gds(gds_fname, lib_name, layer_map, obj_map, grid, tr_colors)

    def _create_sch_templates(self, cell_list: List[Tuple[str, str]]) -> None:
        for lib, cell in cell_list:
            python_file = Path(self.lib_path_map[lib]) / (cell + '.py')
            if not python_file.exists():
                content = self.get_python_template(lib, cell,
                                                   self.db_config.get('prim_table', {}))
                write_file(python_file, content, mkdir=False)
