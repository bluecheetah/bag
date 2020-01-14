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

from typing import Dict, Any

import re
from pathlib import Path

from pybag.enum import DesignOutput
from pybag.core import read_gds

from bag.env import get_bag_work_dir, get_gds_layer_map, get_gds_object_map
from bag.io.file import read_file, read_yaml
from bag.core import BagProject


def check_netlist(output_type: DesignOutput, actual: str, expect: str) -> None:
    if output_type == DesignOutput.CDL:
        inc_line = '^\\.INCLUDE (.*)$'
    elif (output_type == DesignOutput.VERILOG or
          output_type == DesignOutput.SYSVERILOG or
          output_type == DesignOutput.SPECTRE):
        inc_line = '^include "(.*)"$'
    else:
        inc_line = ''

    if not inc_line:
        assert actual == expect
    else:
        bag_work_dir = get_bag_work_dir()
        pattern = re.compile(inc_line)
        actual_lines = actual.splitlines()
        expect_lines = expect.splitlines()
        for al, el in zip(actual_lines, expect_lines):
            am = pattern.match(al)
            if am is None:
                assert al == el
            else:
                em = pattern.match(el)
                if em is None:
                    assert al == el
                else:
                    # both are include statements
                    apath = am.group(1)
                    epath = em.group(1)
                    arel = Path(apath).relative_to(bag_work_dir)
                    assert epath.endswith(str(arel))


def test_dsn(bag_project: BagProject, dsn_specs: Dict[str, Any], gen_output: bool, run_lvs: bool
             ) -> None:
    impl_lib: str = dsn_specs['impl_lib']
    root_dir: str = dsn_specs['root_dir']
    lay_str: str = dsn_specs.get('lay_class', '')
    pytest_info: Dict[str, Path] = dsn_specs['pytest']
    model_type: str = dsn_specs.get('model_type', 'SYSVERILOG')
    root_path = Path(root_dir)
    mod_type: DesignOutput = DesignOutput[model_type]

    lay_db = bag_project.make_template_db(impl_lib)
    bag_project.generate_cell(dsn_specs, raw=True, gen_lay=bool(lay_str), gen_sch=True,
                              run_drc=False, run_lvs=run_lvs, run_rcx=False, lay_db=lay_db,
                              gen_model=True)

    if not gen_output:
        for key, expect_path in pytest_info.items():
            if key == 'test_id':
                continue

            out_path = root_path / key.replace('_', '.')
            if not out_path.is_file():
                raise ValueError(f'Cannot find output file: {out_path}')
            if key.endswith('yaml'):
                actual_dict = read_yaml(out_path)
                expect_dict = read_yaml(expect_path)
                assert actual_dict == expect_dict
            elif key.endswith('gds'):
                lay_map = get_gds_layer_map()
                obj_map = get_gds_object_map()
                grid = lay_db.grid
                tr_colors = lay_db.tr_colors
                expect_cv_list = read_gds(str(expect_path), lay_map, obj_map, grid, tr_colors)
                actual_cv_list = read_gds(str(out_path), lay_map, obj_map, grid, tr_colors)
                assert expect_cv_list == actual_cv_list
            else:
                if key.endswith('netlist'):
                    output_type = DesignOutput.CDL
                else:
                    output_type = mod_type

                actual = read_file(out_path)
                expect = read_file(expect_path)
                check_netlist(output_type, actual, expect)
