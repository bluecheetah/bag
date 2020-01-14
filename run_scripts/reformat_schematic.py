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

"""Reformat BAG2 schematic generator files to BAG3.

NOTE: This is an alpha script, please double check your results.
"""

from typing import Tuple

import os
import glob
import argparse

repl_header = r'''# -*- coding: utf-8 -*-

from typing import Dict, Any

import os
import pkg_resources

from bag.util.immutable import Param
from bag.design.module import Module
from bag.design.database import ModuleDB


# noinspection PyPep8Naming
class {lib_name}__{cell_name}(Module):
    """Module for library {lib_name} cell {cell_name}.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             '{cell_name}.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)
'''


def parse_options() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description='Convert BAG2 schematic generators to BAG3.')
    parser.add_argument('root_path', type=str,
                        help='path to schematic generator files.')
    parser.add_argument('lib_name', type=str,
                        help='schematic library name.')

    args = parser.parse_args()
    return args.root_path, args.lib_name


def main() -> None:
    root_path, lib_name = parse_options()
    os.chdir(root_path)
    for fname in glob.iglob('*.py'):
        if fname == '__init__.py':
            continue

        cell_name = fname[:-3]
        with open(fname, 'r') as f:
            lines = f.readlines()

        new_header = repl_header.format(lib_name=lib_name, cell_name=cell_name)
        with open(fname, 'w') as f:
            f.write(new_header)
            start_write = False
            for l in lines:
                if start_write:
                    f.write(l)
                else:
                    tmp = l.lstrip()
                    if '.__init__(' in tmp:
                        start_write = True


if __name__ == '__main__':
    main()
