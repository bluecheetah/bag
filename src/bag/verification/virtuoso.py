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

"""This module handles exporting schematic/layout from Virtuoso.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple, Union

import os
import shutil
from abc import ABC
from pathlib import Path

from jinja2 import Template

from pybag.enum import DesignOutput

from ..io import write_file
from ..io.template import new_template_env_fs
from ..env import get_bag_work_dir, get_gds_layer_map, get_gds_object_map
from .base import SubProcessChecker, get_flow_config

if TYPE_CHECKING:
    from .base import ProcInfo, FlowInfo


class VirtuosoChecker(SubProcessChecker, ABC):
    """the base Checker class for Virtuoso.

    This class implement layout/schematic export and import procedures.

    Parameters
    ----------
    tmp_dir : str
        temporary directory to save files in.
    root_dir : Dict[str, str]
        dictionary of root run directories.
    template : Dict[str, str]
        dictionary of SVRF jinja template files.
    env_vars: Dict[str, Dict[str, str]]
        dictionary of environment variables.
    params : Dict[str, Dict[str, Any]]
        dictionary of default flow parameters.
    max_workers : int
        maximum number of sub-processes BAG can launch.
    source_added_file : str
        the Calibre source.added file location.  Environment variable is supported.
        If empty (default), this is not configured.
    import_ref_lib : str
        the import reference libraries list file location.  Environment variable is supported.
        If empty (default), this is not configured.
    cancel_timeout_ms : int
        cancel timeout in milliseconds.
    enable_color : bool
        True to enable coloring in GDS export.
    """

    def __init__(self, tmp_dir: str, root_dir: Dict[str, str], template: Dict[str, str],
                 env_vars: Dict[str, Dict[str, str]], link_files: Dict[str, List[str]],
                 params: Dict[str, Dict[str, Any]], max_workers: int = 0,
                 source_added_file: str = '', import_ref_lib: str = '', cancel_timeout_ms: int = 10000,
                 enable_color: bool = False, **kwargs: Dict[str, Any]) -> None:

        cancel_timeout = cancel_timeout_ms / 1e3
        SubProcessChecker.__init__(self, tmp_dir, max_workers, cancel_timeout, **kwargs)

        self._flow_config = get_flow_config(root_dir, template, env_vars, link_files, params)
        self._source_added_file = source_added_file
        self._bag_work_dir = get_bag_work_dir()
        self._strm_params = dict(enable_color=enable_color)
        self._temp_env_ctrl = new_template_env_fs()
        self._import_ref_lib = import_ref_lib

    @property
    def bag_work_dir(self) -> str:
        return self._bag_work_dir

    def get_config(self, mode: str) -> Dict[str, Any]:
        return self._flow_config[mode]

    def get_control_template(self, mode: str) -> Template:
        template: str = self.get_config(mode)['template']
        return self._temp_env_ctrl.get_template(template)

    def setup_job(self, mode: str, lib_name: str, cell_name: str,
                  layout: Optional[str], netlist: Optional[str], lay_view: str, sch_view: str,
                  user_params: Optional[Dict[str, Any]], run_dir_override: Union[str, Path]
                  ) -> Tuple[List[FlowInfo], Path, Optional[Dict[str, str]],
                             Dict[str, Any], Dict[str, Any]]:
        config = self.get_config(mode)
        root_dir: Path = config['root_dir']
        link_files: List[Tuple[Path, Path]] = config['link_files']
        env_vars: Dict[str, str] = config['env_vars']
        params: Dict[str, Any] = config['params']

        if isinstance(run_dir_override, str):
            if run_dir_override:
                run_dir = Path(run_dir_override).resolve()
            else:
                run_dir = root_dir.joinpath(lib_name, cell_name)
        else:
            run_dir = run_dir_override.resolve()

        run_dir.mkdir(parents=True, exist_ok=True)

        for fpath, basename in link_files:
            link = run_dir / basename
            if not link.exists():
                link.symlink_to(fpath)

        flow_list = []
        sch_path = run_dir / 'netlist.cdl'
        ctl_params = dict(cell_name=cell_name)
        if layout is not None:
            if layout:
                ext = Path(layout).suffix[1:]
                lay_path = run_dir / f'layout.{ext}'
                shutil.copy(layout, str(lay_path))
            else:
                ext = 'gds'
                lay_path = run_dir / 'layout.gds'
                info = self.setup_export_layout(lib_name, cell_name, str(lay_path), lay_view)
                flow_list.append((info[0], info[1], info[2], info[3], all_pass_callback))

            if ext == DesignOutput.GDS.extension:
                ctl_params['layout_type'] = 'GDSII'
            elif ext == DesignOutput.OASIS.extension:
                ctl_params['layout_type'] = 'OASIS'
            else:
                raise ValueError(f'Cannot determine layout type from layout file name: {lay_path}')
            ctl_params['layout_file'] = str(lay_path)

        if netlist is not None:
            if netlist:
                shutil.copy(netlist, str(sch_path))
            else:
                info = self.setup_export_schematic(lib_name, cell_name, str(sch_path), sch_view)
                flow_list.append((info[0], info[1], info[2], info[3], all_pass_callback))

        ctl_params['netlist_file'] = str(sch_path)

        params_actual = params.copy()
        if user_params is not None:
            params_actual.update(user_params)
        ctl_params.update(params_actual)

        if env_vars:
            run_env = dict(**os.environ)
            run_env.update(env_vars)
        else:
            run_env = None

        return flow_list, run_dir, run_env, params_actual, ctl_params

    def setup_export_layout(self, lib_name: str, cell_name: str, out_file: str,
                            view_name: str = 'layout', params: Optional[Dict[str, Any]] = None
                            ) -> ProcInfo:
        if params is None:
            params = {}

        enable_color: bool = params.get('enable_color',
                                        self._strm_params.get('enable_color', False))
        square_bracket: bool = params.get('square_bracket', False)

        out_path = Path(out_file).resolve()
        run_dir = out_path.parent
        out_name = out_path.name

        output_type: DesignOutput = params.get('output_type')
        if not output_type:
            # figure out output_type from out_file extension
            if out_path.suffix.lower() == '.gds':
                output_type = DesignOutput.GDS
            elif out_path.suffix.lower() in ['.oas', '.oasis']:
                output_type = DesignOutput.OASIS
            else:
                raise ValueError(f'Unknown layout export format: {out_path.suffix}')

        if output_type is DesignOutput.GDS:
            template_name = 'gds_export_config.txt'
            cmd_str = 'strmout'
        elif output_type is DesignOutput.OASIS:
            template_name = 'oasis_export_config.txt'
            cmd_str = 'oasisout'
        else:
            raise ValueError(f'Unknown layout export format: {output_type.name}')

        log_file = str(run_dir / 'layout_export.log')

        run_dir.mkdir(parents=True, exist_ok=True)

        # fill in stream out configuration file.
        content = self.render_file_template(template_name,
                                            dict(lib_name=lib_name,
                                                 cell_name=cell_name,
                                                 view_name=view_name,
                                                 output_name=out_name,
                                                 run_dir=str(run_dir),
                                                 enable_color=str(enable_color).lower(),
                                                 square_bracket=str(square_bracket).lower(),
                                                 ))
        # run strmOut
        ctrl_file = run_dir / 'stream_template'
        write_file(ctrl_file, content)
        cmd = [cmd_str, '-templateFile', str(ctrl_file)]
        return cmd, log_file, None, self._bag_work_dir

    def setup_import_layout(self, in_file: str, lib_name: str, cell_name: str,
                            view_name: str = 'layout', params: Optional[Dict[str, Any]] = None
                            ) -> ProcInfo:
        if params is None:
            params = {}

        enable_color: bool = params.get('enable_color',
                                        self._strm_params.get('enable_color', False))
        square_bracket: bool = params.get('square_bracket', False)

        in_path = Path(in_file).resolve()
        run_dir = in_path.parent
        in_name = in_path.name

        input_type: DesignOutput = params.get('input_type')
        if not input_type:
            # figure out input_type from in_file extension
            if in_path.suffix.lower() == '.gds':
                input_type = DesignOutput.GDS
            elif in_path.suffix.lower() in ['.oas', '.oasis']:
                input_type = DesignOutput.OASIS
            else:
                raise ValueError(f'Unknown layout import format: {in_path.suffix}')

        if input_type is DesignOutput.GDS:
            template_name = 'gds_import_config.txt'
            cmd_str = 'strmin'
        elif input_type is DesignOutput.OASIS:
            template_name = 'oasis_import_config.txt'
            cmd_str = 'oasisin'
        else:
            raise ValueError(f'Unknown layout import format: {input_type.name}')

        log_file = str(run_dir / 'layout_import.log')

        run_dir.mkdir(parents=True, exist_ok=True)

        # fill in stream in configuration file.
        content = self.render_file_template(template_name,
                                            dict(lib_name=lib_name,
                                                 cell_name=cell_name,
                                                 view_name=view_name,
                                                 input_name=in_name,
                                                 run_dir=str(run_dir),
                                                 enable_color=str(enable_color).lower(),
                                                 square_bracket=str(square_bracket).lower(),
                                                 import_ref_lib=self._import_ref_lib,
                                                 layer_map=get_gds_layer_map(),
                                                 object_map=get_gds_object_map(),
                                                 ))
        # run strmOut
        ctrl_file = run_dir / 'stream_template'
        write_file(ctrl_file, content)
        cmd = [cmd_str, '-templateFile', str(ctrl_file)]
        return cmd, log_file, None, self._bag_work_dir

    def setup_export_schematic(self, lib_name: str, cell_name: str, out_file: str,
                               view_name: str = 'schematic',
                               params: Optional[Dict[str, Any]] = None) -> ProcInfo:
        out_path = Path(out_file).resolve()
        run_dir = out_path.parent
        out_name = out_path.name
        log_file = str(run_dir / 'schematic_export.log')

        run_dir.mkdir(parents=True, exist_ok=True)

        # fill in stream out configuration file.
        content = self.render_file_template('si_env.txt',
                                            dict(
                                                lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                source_added_file=self._source_added_file,
                                                run_dir=run_dir,
                                            ))

        # run command
        write_file(run_dir / 'si.env', content)
        cmd = ['si', str(run_dir), '-batch', '-command', 'netlist']

        return cmd, log_file, None, self._bag_work_dir


# noinspection PyUnusedLocal
def all_pass_callback(retcode: int, log_file: str) -> bool:
    return True
