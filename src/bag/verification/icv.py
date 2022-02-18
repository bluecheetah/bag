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

"""This module implements LVS/RCX using ICV and stream out from Virtuoso.
"""

from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Any, Sequence

import os

from .virtuoso import VirtuosoChecker
from ..io import read_file, open_temp

if TYPE_CHECKING:
    from .base import FlowInfo


# noinspection PyUnusedLocal
def _all_pass(retcode: int, log_file: str) -> bool:
    return True


# noinspection PyUnusedLocal
def lvs_passed(retcode: int, log_file: str) -> Tuple[bool, str]:
    """Check if LVS passed

    Parameters
    ----------
    retcode : int
        return code of the LVS process.
    log_file : str
        log file name.

    Returns
    -------
    success : bool
        True if LVS passed.
    log_file : str
        the log file name.
    """
    dirname = os.path.dirname(log_file)
    cell_name = os.path.basename(dirname)
    lvs_error_file = os.path.join(dirname, cell_name + '.LVS_ERRORS')

    # append error file at the end of log file
    with open(log_file, 'a') as logf:
        with open(lvs_error_file, 'r') as errf:
            for line in errf:
                logf.write(line)

    if not os.path.isfile(log_file):
        return False, ''

    cmd_output = read_file(log_file)
    test_str = 'Final comparison result:PASS'

    return test_str in cmd_output, log_file


class ICV(VirtuosoChecker):
    """A subclass of VirtuosoChecker that uses ICV for verification.

    Parameters
    ----------
    tmp_dir : string
        temporary directory to save files in.
    lvs_run_dir : str
        the LVS run directory.
    lvs_runset : str
        the LVS runset filename.
    rcx_run_dir : str
        the RCX run directory.
    rcx_runset : str
        the RCX runset filename.
    source_added_file : str
        the source.added file location.  Environment variable is supported.
        Default value is '$DK/Calibre/lvs/source.added'.
    rcx_mode : str
        the RC extraction mode.  Defaults to 'starrc'.
    """

    def __init__(self, tmp_dir: str, lvs_run_dir: str, lvs_runset: str, rcx_run_dir: str,
                 rcx_runset: str, source_added_file: str = '$DK/Calibre/lvs/source.added',
                 rcx_mode: str = 'pex', **kwargs):

        max_workers = kwargs.get('max_workers', None)
        cancel_timeout = kwargs.get('cancel_timeout_ms', None)
        rcx_params = kwargs.get('rcx_params', {})
        lvs_params = kwargs.get('lvs_params', {})
        rcx_link_files = kwargs.get('rcx_link_files', None)
        lvs_link_files = kwargs.get('lvs_link_files', None)

        if cancel_timeout is not None:
            cancel_timeout /= 1e3

        VirtuosoChecker.__init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file, **kwargs)

        self.default_rcx_params = rcx_params
        self.default_lvs_params = lvs_params
        self.lvs_run_dir = os.path.abspath(lvs_run_dir)
        self.lvs_runset = lvs_runset
        self.lvs_link_files = lvs_link_files
        self.rcx_run_dir = os.path.abspath(rcx_run_dir)
        self.rcx_runset = rcx_runset
        self.rcx_link_files = rcx_link_files
        self.rcx_mode = rcx_mode

    def get_rcx_netlists(self, lib_name: str, cell_name: str) -> List[str]:
        """Returns a list of generated extraction netlist file names.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name

        Returns
        -------
        netlists : List[str]
            a list of generated extraction netlist file names.  The first index is the main netlist.
        """
        # PVS generate schematic cellviews directly.
        if self.rcx_mode == 'starrc':
            return ['%s.spf' % cell_name]
        else:
            pass

    def setup_lvs_flow(self, lib_name: str, cell_name: str, sch_view: str = 'schematic',
                       lay_view: str = 'layout', gds: str = '', netlist = '',
                       params: Optional[Dict[str, Any]] = None) -> Sequence[FlowInfo]:

        if netlist:
            netlist = os.path.abspath(netlist)

        run_dir = os.path.join(self.lvs_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        lay_file, sch_file = self._get_lay_sch_files(run_dir, netlist)

        # add schematic/layout export to flow
        flow_list = []
        if not gds:
            cmd, log, env, cwd = self.setup_export_layout(lib_name, cell_name, lay_file, lay_view,
                                                          None)
            flow_list.append((cmd, log, env, cwd, _all_pass))
        if not netlist:
            cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file,
                                                             sch_view, None)
            flow_list.append((cmd, log, env, cwd, _all_pass))

        lvs_params_actual = self.default_lvs_params.copy()
        if params is not None:
            lvs_params_actual.update(params)

        with open_temp(prefix='lvsLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name

        # set _drPROCESS
        dr_process_str = '_drPROCESS=' + lvs_params_actual['_drPROCESS']

        cmd = ['icv', '-D', dr_process_str, '-i', lay_file, '-s', sch_file, '-sf', 'SPICE',
               '-f', 'GDSII', '-c', cell_name, '-vue', '-I']
        for f in self.lvs_link_files:
            cmd.append(f)

        flow_list.append((cmd, log_file, None, run_dir, lvs_passed))
        return flow_list

    def setup_rcx_flow(self, lib_name: str, cell_name: str, sch_view: str = 'schematic',
                       lay_view: str = 'layout', gds: str = '', netlist: str = '',
                       params: Optional[Dict[str, Any]] = None) -> Sequence[FlowInfo]:

        # update default RCX parameters.
        rcx_params_actual = self.default_rcx_params.copy()
        if params is not None:
            rcx_params_actual.update(params)

        run_dir = os.path.join(self.rcx_run_dir, lib_name, cell_name)
        os.makedirs(run_dir, exist_ok=True)

        lay_file, sch_file = self._get_lay_sch_files(run_dir, netlist)
        with open_temp(prefix='rcxLog', dir=run_dir, delete=False) as logf:
            log_file = logf.name
        flow_list = []
        if not gds:
            cmd, log, env, cwd = self.setup_export_layout(lib_name, cell_name, lay_file, lay_view,
                                                          None)
            flow_list.append((cmd, log, env, cwd, _all_pass))
        if not netlist:
            cmd, log, env, cwd = self.setup_export_schematic(lib_name, cell_name, sch_file,
                                                             sch_view, None)
            flow_list.append((cmd, log, env, cwd, _all_pass))

        if self.rcx_mode == 'starrc':
            # first: run Extraction LVS
            lvs_params_actual = self.default_lvs_params.copy()

            dr_process_str = '_drPROCESS=' + lvs_params_actual['_drPROCESS']

            cmd = ['icv', '-D', '_drRCextract', '-D', dr_process_str, '-D', '_drICFOAlayers',
                   '-i', lay_file, '-s', sch_file, '-sf', 'SPICE', '-f', 'GDSII',
                   '-c', cell_name, '-I']
            for f in self.lvs_link_files:
                cmd.append(f)

            # hack the environment variables to make sure $PWD is the same as current working directory
            env_copy = os.environ.copy()
            env_copy['PWD'] = run_dir
            flow_list.append((cmd, log_file, env_copy, run_dir, lvs_passed))

            # second: setup CCP
            # make symlinks
            if self.rcx_link_files:
                for source_file in self.rcx_link_files:
                    targ_file = os.path.join(run_dir, os.path.basename(source_file))
                    if not os.path.exists(targ_file):
                        os.symlink(source_file, targ_file)

            # generate new cmd for StarXtract
            cmd_content, result = self.modify_starrc_cmd(run_dir, lib_name, cell_name,
                                                         rcx_params_actual, sch_file)

            # save cmd for StarXtract
            with open_temp(dir=run_dir, delete=False) as cmd_file:
                cmd_fname = cmd_file.name
                cmd_file.write(cmd_content)

            cmd = ['StarXtract', '-clean', cmd_fname]
        else:
            cmd = []

        # noinspection PyUnusedLocal
        def rcx_passed(retcode, log_fname):
            dirname = os.path.dirname(log_fname)
            cell_name = os.path.basename(dirname)
            results_file = os.path.join(dirname, cell_name + '.RESULTS')

            # append error file at the end of log file
            with open(log_fname, 'a') as logf:
                with open(results_file, 'r') as errf:
                    for line in errf:
                        logf.write(line)

            if not os.path.isfile(log_fname):
                return None, ''

            cmd_output = read_file(log_fname)
            test_str = 'DRC and Extraction Results: CLEAN'

            if test_str in cmd_output:
                return results_file, log_fname
            else:
                return None, log_fname

        flow_list.append((cmd, log_file, None, run_dir, rcx_passed))
        return flow_list

    @classmethod
    def _get_lay_sch_files(cls, run_dir, netlist=''):
        lay_file = os.path.join(run_dir, 'layout.gds')
        sch_file = netlist if netlist else os.path.join(run_dir, 'schematic.net')
        return lay_file, sch_file

    def modify_starrc_cmd(self, run_dir, lib_name, cell_name, starrc_params, sch_file):
        # type: (str, str, str, Dict[str, Any], str) -> Tuple[str, str]
        """Modify the cmd file.

        Parameters
        ----------
        run_dir : str
            the run directory.
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        starrc_params : Dict[str, Any]
            override StarRC parameters.
        sch_file : str
            the schematic netlist

        Returns
        -------
        starrc_cmd : str
            the new StarXtract cmd file.
        output_name : str
            the extracted netlist file.
        """
        output_name = '%s.spf' % cell_name
        if 'CDSLIBPATH' in os.environ:
            cds_lib_path = os.path.abspath(os.path.join(os.environ['CDSLIBPATH'], 'cds.lib'))
        else:
            cds_lib_path = os.path.abspath('./cds.lib')
        content = self.render_string_template(read_file(self.rcx_runset),
                                              dict(
                                                  cell_name=cell_name,
                                                  extract_type=starrc_params['extract'].get('type'),
                                                  netlist_format=starrc_params.get('netlist_format',
                                                                                   'SPF'),
                                                  sch_file=sch_file,
                                                  cds_lib=cds_lib_path,
                                                  lib_name=lib_name,
                                                  run_dir=run_dir,
                                              ))
        return content, os.path.join(run_dir, output_name)
