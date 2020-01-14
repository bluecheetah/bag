# SPDX-License-Identifier: Apache-2.0
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

import argparse

from bag.io import read_yaml
from bag.core import BagProject
from bag.util.misc import register_pdb_hook

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulate cell from spec file.')
    parser.add_argument('specs', help='YAML specs file name.')
    parser.add_argument('-x', '--extract', dest='extract', action='store_true', default=False,
                        help='generate extracted netlist.')
    parser.add_argument('--no-tb', dest='gen_tb', action='store_false', default=True,
                        help='disable testbench generation; simulate using existing files')
    parser.add_argument('--no-sim', dest='simulate', action='store_false', default=True,
                        help='disable simulation; only generate the netlists')
    parser.add_argument('-mismatch', '--do-mismatch', dest='mismatch', action='store_true',
                        default=False, help='enables mismatch analysis')
    parser.add_argument('--gen-oa', dest='raw', action='store_false',
                        default=True, help='enables oa view generation')

    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    prj.simulate_cell(specs, extract=args.extract, gen_tb=args.gen_tb, simulate=args.simulate,
                      mismatch=args.mismatch, raw=args.raw)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    run_main(_prj, _args)
