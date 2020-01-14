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

"""This module implements all CAD database manipulations using skill commands.
"""

from typing import TYPE_CHECKING, Sequence, List, Dict, Optional, Any, Tuple, Set

import os
import shutil

from ..io.common import get_encoding, fix_string
from ..io.file import open_temp, read_yaml
from ..io.string import read_yaml_str
from .database import DbAccess

try:
    import cybagoa
except ImportError:
    cybagoa = None

if TYPE_CHECKING:
    from .zmqwrapper import ZMQDealer


def _dict_to_pcell_params(table):
    """Convert given parameter dictionary to pcell parameter list format.

    Parameters
    ----------
    table : dict[str, any]
        the parameter dictionary.

    Returns
    -------
    param_list : list[any]
        the Pcell parameter list
    """
    param_list = []
    for key, val in table.items():
        # python 2/3 compatibility: convert raw bytes to string.
        val = fix_string(val)
        if isinstance(val, float):
            param_list.append([key, "float", val])
        elif isinstance(val, str):
            # unicode string
            param_list.append([key, "string", val])
        elif isinstance(val, int):
            param_list.append([key, "int", val])
        elif isinstance(val, bool):
            param_list.append([key, "bool", val])
        else:
            raise Exception('Unsupported parameter %s with type: %s' % (key, type(val)))

    return param_list


def to_skill_list_str(pylist):
    """Convert given python list to a skill list string.

    Parameters
    ----------
    pylist : list[str]
        a list of string.

    Returns
    -------
    ans : str
        a string representation of the equivalent skill list.

    """
    content = ' '.join(('"%s"' % val for val in pylist))
    return "'( %s )" % content


def handle_reply(reply):
    """Process the given reply."""
    if isinstance(reply, dict):
        if reply.get('type') == 'error':
            if 'data' not in reply:
                raise Exception('Unknown reply format: %s' % reply)
            raise VirtuosoException(reply['data'])
        else:
            try:
                return reply['data']
            except Exception:
                raise Exception('Unknown reply format: %s' % reply)
    else:
        raise Exception('Unknown reply format: %s' % reply)


class VirtuosoException(Exception):
    """Exception raised when Virtuoso returns an error."""

    def __init__(self, *args, **kwargs):
        # noinspection PyArgumentList
        Exception.__init__(self, *args, **kwargs)


class SkillInterface(DbAccess):
    """Skill interface between bag and Virtuoso.

    This class sends all bag's database and simulation operations to
    an external Virtuoso process, then get the result from it.
    """

    def __init__(self, dealer, tmp_dir, db_config, lib_defs_file):
        # type: (ZMQDealer, str, Dict[str, Any], str) -> None
        DbAccess.__init__(self, dealer, tmp_dir, db_config, lib_defs_file)
        self.exc_libs = set(db_config['schematic']['exclude_libraries'])
        # BAG_prim is always excluded
        self.exc_libs.add('BAG_prim')

    def _eval_skill(self, expr, input_files=None, out_file=None):
        # type: (str, Optional[Dict[str, Any]], Optional[str]) -> str
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
        expr : string
            the skill expression to evaluate.
        input_files : dict[string, any] or None
            A dictionary of input files content.
        out_file : string or None
            the output file name argument in expr.

        Returns
        -------
        result : str
            a string representation of the result.

        Raises
        ------
        :class: `.VirtuosoException` :
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

    def get_exit_object(self):
        # type: () -> Any
        return {'type': 'exit'}

    def get_cells_in_library(self, lib_name):
        # type: (str) -> List[str]
        cmd = 'get_cells_in_library_file( "%s" {cell_file} )' % lib_name
        return self._eval_skill(cmd, out_file='cell_file').split()

    def create_library(self, lib_name, lib_path=''):
        # type: (str, str) -> None
        lib_path = lib_path or self.default_lib_path
        tech_lib = self.db_config['schematic']['tech_lib']
        self._eval_skill(
            'create_or_erase_library("%s" "%s" "%s" nil)' % (lib_name, tech_lib, lib_path))

    def create_implementation(self, lib_name, template_list, change_list, lib_path=''):
        # type: (str, Sequence[Any], Sequence[Any], str) -> None
        lib_path = lib_path or self.default_lib_path
        tech_lib = self.db_config['schematic']['tech_lib']

        if cybagoa is not None and self.db_config['schematic'].get('use_cybagoa', False):
            cds_lib_path = os.environ.get('CDS_LIB_PATH', './cds.lib')
            sch_name = 'schematic'
            sym_name = 'symbol'
            encoding = get_encoding()
            # release write locks
            cell_view_list = []
            for _, _, cell_name in template_list:
                cell_view_list.append((cell_name, sch_name))
                cell_view_list.append((cell_name, sym_name))
            self.release_write_locks(lib_name, cell_view_list)

            # create library in case it doesn't exist
            self.create_library(lib_name, lib_path)

            # write schematic
            with cybagoa.PyOASchematicWriter(cds_lib_path, lib_name, encoding) as writer:
                for temp_info, change_info in zip(template_list, change_list):
                    sch_cell = cybagoa.PySchCell(temp_info[0], temp_info[1], temp_info[2], encoding)
                    for old_pin, new_pin in change_info['pin_map']:
                        sch_cell.rename_pin(old_pin, new_pin)
                    for inst_name, rinst_list in change_info['inst_list']:
                        sch_cell.add_inst(inst_name, lib_name, rinst_list)
                    writer.add_sch_cell(sch_cell)
                writer.create_schematics(sch_name, sym_name)

            copy = 'nil'
        else:
            copy = "'t"

        in_files = {'template_list': template_list,
                    'change_list': change_list}
        sympin = to_skill_list_str(self.db_config['schematic']['sympin'])
        ipin = to_skill_list_str(self.db_config['schematic']['ipin'])
        opin = to_skill_list_str(self.db_config['schematic']['opin'])
        iopin = to_skill_list_str(self.db_config['schematic']['iopin'])
        simulators = to_skill_list_str(self.db_config['schematic']['simulators'])
        cmd = ('create_concrete_schematic( "%s" "%s" "%s" {template_list} '
               '{change_list} %s %s %s %s %s %s)' % (lib_name, tech_lib, lib_path,
                                                     sympin, ipin, opin, iopin, simulators, copy))

        self._eval_skill(cmd, input_files=in_files)

    def configure_testbench(self, tb_lib, tb_cell):
        # type: (str, str) -> Tuple[str, List[str], Dict[str, str], Dict[str, str]]

        tb_config = self.db_config['testbench']

        cmd = ('instantiate_testbench("{tb_cell}" "{targ_lib}" ' +
               '"{config_libs}" "{config_views}" "{config_stops}" ' +
               '"{default_corner}" "{corner_file}" {def_files} ' +
               '"{tech_lib}" {result_file})')
        cmd = cmd.format(tb_cell=tb_cell,
                         targ_lib=tb_lib,
                         config_libs=tb_config['config_libs'],
                         config_views=tb_config['config_views'],
                         config_stops=tb_config['config_stops'],
                         default_corner=tb_config['default_env'],
                         corner_file=tb_config['env_file'],
                         def_files=to_skill_list_str(tb_config['def_files']),
                         tech_lib=self.db_config['schematic']['tech_lib'],
                         result_file='{result_file}')
        output = read_yaml(self._eval_skill(cmd, out_file='result_file'))
        return tb_config['default_env'], output['corners'], output['parameters'], output['outputs']

    def get_testbench_info(self, tb_lib, tb_cell):
        # type: (str, str) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]
        cmd = 'get_testbench_info("{tb_lib}" "{tb_cell}" {result_file})'
        cmd = cmd.format(tb_lib=tb_lib,
                         tb_cell=tb_cell,
                         result_file='{result_file}')
        output = read_yaml(self._eval_skill(cmd, out_file='result_file'))
        return output['enabled_corners'], output['corners'], output['parameters'], output['outputs']

    def update_testbench(self,  # type: SkillInterface
                         lib,  # type: str
                         cell,  # type: str
                         parameters,  # type: Dict[str, str]
                         sim_envs,  # type: List[str]
                         config_rules,  # type: List[List[str]]
                         env_parameters,  # type: List[List[Tuple[str, str]]]
                         ):
        # type: (...) -> None
        cmd = 'modify_testbench("%s" "%s" {conf_rules} ' \
              '{run_opts} {sim_envs} {params} {env_params})' % (lib, cell)
        in_files = {'conf_rules': config_rules,
                    'run_opts': [],
                    'sim_envs': sim_envs,
                    'params': list(parameters.items()),
                    'env_params': list(zip(sim_envs, env_parameters)),
                    }
        self._eval_skill(cmd, input_files=in_files)

    def instantiate_schematic(self, lib_name, content_list, lib_path='',
                              sch_view='schematic', sym_view='symbol'):
        # type: (str, Sequence[Any], str, str, str) -> None
        raise NotImplementedError('Not implemented yet.')

    def instantiate_layout_pcell(self, lib_name, cell_name, view_name,
                                 inst_lib, inst_cell, params, pin_mapping):
        # type: (str, str, str, str, str, Dict[str, Any], Dict[str, str]) -> None
        # create library in case it doesn't exist
        self.create_library(lib_name)

        # convert parameter dictionary to pcell params list format
        param_list = _dict_to_pcell_params(params)

        cmd = ('create_layout_with_pcell( "%s" "%s" "%s" "%s" "%s"'
               '{params} {pin_mapping} )' % (lib_name, cell_name,
                                             view_name, inst_lib, inst_cell))
        in_files = {'params': param_list, 'pin_mapping': list(pin_mapping.items())}
        self._eval_skill(cmd, input_files=in_files)

    def instantiate_layout(self, lib_name, content_list, lib_path='', view='layout'):
        # type: (str, Sequence[Any], str, str) -> None
        # create library in case it doesn't exist
        self.create_library(lib_name)

        # convert parameter dictionary to pcell params list format
        new_layout_list = []
        for info_list in content_list:
            new_inst_list = []
            for inst in info_list[1]:
                if 'params' in inst:
                    inst = inst.copy()
                    inst['params'] = _dict_to_pcell_params(inst['params'])
                new_inst_list.append(inst)

            new_info_list = info_list[:]
            new_info_list[1] = new_inst_list
            new_layout_list.append(new_info_list)

        tech_lib = self.db_config['schematic']['tech_lib']
        cmd = 'create_layout( "%s" "%s" "%s" {layout_list} )' % (lib_name, view, tech_lib)
        in_files = {'layout_list': new_layout_list}
        self._eval_skill(cmd, input_files=in_files)

    def release_write_locks(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        cmd = 'release_write_locks( "%s" {cell_view_list} )' % lib_name
        in_files = {'cell_view_list': cell_view_list}
        self._eval_skill(cmd, input_files=in_files)

    def refresh_cellviews(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        cmd = 'refresh_cellviews( "%s" {cell_view_list} )' % lib_name
        in_files = {'cell_view_list': cell_view_list}
        self._eval_skill(cmd, input_files=in_files)

    def perform_checks_on_cell(self, lib_name, cell_name, view_name):
        # type: (str, str, str) -> None
        self._eval_skill(
            'check_and_save_cell( "{}" "{}" "{}" )'.format(lib_name, cell_name, view_name))

    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view='', **kwargs):
        # type: (str, str, str, str, **Any) -> None
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
        calview_config = self.db_config.get('calibreview', None)
        use_calibreview = self.db_config.get('use_calibreview', True)
        if calview_config is not None and use_calibreview:
            # create calibre view from extraction netlist
            cell_map = calview_config['cell_map']
            sch_view = sch_view or calview_config['view_name']

            # create calibre view config file
            tmp_params = dict(
                netlist_file=netlist,
                lib_name=lib_name,
                cell_name=cell_name,
                calibre_cellmap=cell_map,
                view_name=sch_view,
            )
            content = self.render_file_template('calibreview_setup.txt', tmp_params)
            with open_temp(prefix='calview', dir=self.tmp_dir, delete=False) as f:
                fname = f.name
                f.write(content)

            # delete old calibre view
            cmd = 'delete_cellview( "%s" "%s" "%s" )' % (lib_name, cell_name, sch_view)
            self._eval_skill(cmd)
            # make extracted schematic
            cmd = 'mgc_rve_load_setup_file( "%s" )' % fname
            self._eval_skill(cmd)
        else:
            # get netlists to copy
            netlist_dir = os.path.dirname(netlist)
            netlist_files = self.checker.get_rcx_netlists(lib_name, cell_name)
            if not netlist_files:
                # some error checking.  Shouldn't be needed but just in case
                raise ValueError('RCX did not generate any netlists')

            # copy netlists to a "netlist" subfolder in the CAD database
            cell_dir = self.get_cell_directory(lib_name, cell_name)
            targ_dir = os.path.join(cell_dir, 'netlist')
            os.makedirs(targ_dir, exist_ok=True)
            for fname in netlist_files:
                shutil.copy(os.path.join(netlist_dir, fname), targ_dir)

            # create symbolic link as aliases
            symlink = os.path.join(targ_dir, 'netlist')
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            os.symlink(netlist_files[0], symlink)

    def get_cell_directory(self, lib_name, cell_name):
        # type: (str, str) -> str
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
        # use yaml.load to remove outermost quotation marks
        lib_dir = read_yaml_str(self._eval_skill('get_lib_directory( "%s" )' % lib_name))  # type: str
        if not lib_dir:
            raise ValueError('Library %s not found.' % lib_name)
        return os.path.join(lib_dir, cell_name)

    def create_verilog_view(self, verilog_file, lib_name, cell_name, **kwargs):
        # type: (str, str, str, **Any) -> None
        # delete old verilog view
        cmd = 'delete_cellview( "%s" "%s" "verilog" )' % (lib_name, cell_name)
        self._eval_skill(cmd)
        cmd = 'schInstallHDL("%s" "%s" "verilog" "%s" t)' % (lib_name, cell_name, verilog_file)
        self._eval_skill(cmd)

    def import_sch_cellview(self, lib_name, cell_name, view_name):
        # type: (str, str, str) -> None
        self._import_design(lib_name, cell_name, view_name, set())

    def import_design_library(self, lib_name, view_name):
        # type: (str, str) -> None
        imported_cells = set()
        for cell_name in self.get_cells_in_library(lib_name):
            self._import_design(lib_name, cell_name, view_name, imported_cells)

    def _import_design(self, lib_name, cell_name, view_name, imported_cells):
        # type: (str, str, str, Set[str]) -> None
        """Recursive helper for import_design_library.
        """
        # check if we already imported this schematic
        key = '%s__%s' % (lib_name, cell_name)
        if key in imported_cells:
            return
        imported_cells.add(key)

        # create root directory if missing
        root_path = dsn_db.get_library_path(lib_name)
        if not root_path:
            root_path = new_lib_path
            dsn_db.append_library(lib_name, new_lib_path)

        package_path = os.path.join(root_path, lib_name)
        python_file = os.path.join(package_path, '%s.py' % cell_name)
        yaml_file = os.path.join(package_path, 'netlist_info', '%s.yaml' % cell_name)
        yaml_dir = os.path.dirname(yaml_file)
        if not os.path.exists(yaml_dir):
            os.makedirs(yaml_dir)
            bag.io.write_file(os.path.join(package_path, '__init__.py'), '\n',
                              mkdir=False)

        # update netlist file
        content = self.parse_schematic_template(lib_name, cell_name)
        sch_info = read_yaml_str(content)
        try:
            bag.io.write_file(yaml_file, content)
        except IOError:
            print('Warning: cannot write to %s.' % yaml_file)

        # generate new design module file if necessary.
        if not os.path.exists(python_file):
            content = self.get_python_template(lib_name, cell_name,
                                               self.db_config.get('prim_table', {}))
            bag.io.write_file(python_file, content + '\n', mkdir=False)

        # recursively import all children
        for inst_name, inst_attrs in sch_info['instances'].items():
            inst_lib_name = inst_attrs['lib_name']
            if inst_lib_name not in self.exc_libs:
                inst_cell_name = inst_attrs['cell_name']
                self._import_design(inst_lib_name, inst_cell_name, imported_cells, dsn_db,
                                    new_lib_path)

    def parse_schematic_template(self, lib_name, cell_name):
        # type: (str, str) -> str
        """Parse the given schematic template.

        Parameters
        ----------
        lib_name : str
            name of the library.
        cell_name : str
            name of the cell.

        Returns
        -------
        template : str
            the content of the netlist structure file.
        """
        cmd = 'parse_cad_sch( "%s" "%s" {netlist_info} )' % (lib_name, cell_name)
        return self._eval_skill(cmd, out_file='netlist_info')
