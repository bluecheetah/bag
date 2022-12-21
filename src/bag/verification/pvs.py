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

"""This module implements LVS/RCX using PVS/QRC and stream out from Virtuoso.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Sequence, Tuple, Union

from pathlib import Path

from ..io import read_file, write_file

from .virtuoso import VirtuosoChecker

if TYPE_CHECKING:
    from .base import FlowInfo


class PVS(VirtuosoChecker):
    """A subclass of VirtuosoChecker that uses PVS/QRC for verification.

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
    lvs_cmd : str
        the lvs command.
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
                 params: Dict[str, Dict[str, Any]],
                 lvs_cmd: str = 'pvs', max_workers: int = 0, source_added_file: str = '', import_ref_lib: str = '',
                 cancel_timeout_ms: int = 10000, enable_color: bool = False, **kwargs: Dict[str, Any]) -> None:
        VirtuosoChecker.__init__(self, tmp_dir, root_dir, template, env_vars, link_files,
                                 params, max_workers, source_added_file, import_ref_lib, cancel_timeout_ms,
                                 enable_color, **kwargs)

        self._lvs_cmd = lvs_cmd

    def get_rcx_netlists(self, lib_name: str, cell_name: str) -> List[str]:
        # PVS generate schematic cellviews directly.
        return [f'{cell_name}.spf']

    def setup_drc_flow(self, lib_name: str, cell_name: str, lay_view: str = 'layout',
                       layout: str = '', params: Optional[Dict[str, Any]] = None,
                       run_dir: Union[str, Path] = '') -> Sequence[FlowInfo]:
        raise NotImplementedError('Not supported yet.')

    def setup_lvs_flow(self, lib_name: str, cell_name: str, sch_view: str = 'schematic',
                       lay_view: str = 'layout', layout: str = '', netlist: str = '',
                       params: Optional[Dict[str, Any]] = None, run_rcx: bool = False,
                       run_dir: Union[str, Path] = '') -> Sequence[FlowInfo]:
        mode = 'lvs_rcx' if run_rcx else 'lvs'

        tmp = self.setup_job(mode, lib_name, cell_name, layout, netlist, lay_view,
                             sch_view, params, run_dir)
        flow_list, run_dir_path, run_env, params, ctl_params = tmp

        if ctl_params['layout_type'] != 'GDSII':
            raise ValueError('Only LVS with gds file is supported.')

        # generate new control file
        ctl_path = run_dir_path / f'bag_{mode}.ctrl'
        temp = self.get_control_template(mode)
        content = temp.render(**ctl_params)
        write_file(ctl_path, content)

        cmd = [self._lvs_cmd, '-perc', '-lvs', '-control', str(ctl_path),
               '-gds', ctl_params['layout_file'], '-layout_top_cell', cell_name,
               '-source_cdl', ctl_params['netlist_file'], '-source_top_cell', cell_name,
               'pvs_rules']
        if run_rcx:
            cmd.insert(3, '-qrc_data')

        log_path = run_dir_path / f'bag_{mode}.log'
        flow_list.append((cmd, str(log_path), run_env, str(run_dir_path), _lvs_passed_check))

        return flow_list

    def setup_rcx_flow(self, lib_name: str, cell_name: str, sch_view: str = 'schematic',
                       lay_view: str = 'layout', layout: str = '', netlist: str = '',
                       params: Optional[Dict[str, Any]] = None, run_dir: Union[str, Path] = ''
                       ) -> Sequence[FlowInfo]:
        # noinspection PyUnusedLocal
        def _rcx_passed_check(retcode: int, log_file: str) -> Tuple[str, str]:
            out_file = Path(log_file).parent.resolve()
            out_file = out_file.joinpath(f'{cell_name}.spf')
            if not out_file.is_file():
                return '', ''

            return str(out_file), str(log_file)

        mode = 'rcx'
        tmp = self.setup_job(mode, lib_name, cell_name, layout, netlist, lay_view, sch_view, params, run_dir)
        flow_list, run_dir, run_env, params, ctl_params = tmp

        ctl_path = run_dir / f'bag_{mode}.ctrl'
        ctl_template = self.get_control_template(mode)
        content = ctl_template.render(**ctl_params)
        write_file(ctl_path, content)

        # generate new control file
        run_cmd = ['qrc', '-64', '-cmd', str(ctl_path)]
        log_path = run_dir / f'bag_{mode}.log'

        flow_list.append((run_cmd, str(log_path), run_env, str(run_dir), _rcx_passed_check))
        return flow_list


# noinspection PyUnusedLocal
def _lvs_passed_check(retcode: int, log_file: str) -> Tuple[bool, str]:
    """Check if LVS passed

    Parameters
    ----------
    retcode : int
        return code of the LVS subprocess.
    log_file : str
        log file name.

    Returns
    -------
    success : bool
        True if LVS passed.
    log_file : str
        the log file name.
    """
    fpath = Path(log_file)
    if not fpath.is_file():
        return False, ''

    cmd_output = read_file(fpath)
    test_str = '# Run Result             : MATCH'
    return test_str in cmd_output, log_file
