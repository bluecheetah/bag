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

from bag.core import BagProject
from bag.util.misc import register_pdb_hook

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run DRC/LVS/RCX.')
    parser.add_argument('lib', help='library name.')
    parser.add_argument('cell', help='cell name.')
    parser.add_argument('-d', '--drc', dest='run_drc', action='store_true', default=False,
                        help='run DRC.')
    parser.add_argument('-v', '--lvs', dest='run_lvs', action='store_true', default=False,
                        help='run LVS.  Pass --rcx flag to run LVS for extraction.')
    parser.add_argument('-x', '--rcx', dest='run_rcx', action='store_true', default=False,
                        help='run RCX.')

    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    if args.run_drc:
        print('Running DRC')
        success, log = prj.run_drc(args.lib, args.cell)
        if success:
            print('DRC passed!')
        else:
            print('DRC failed...')
        print(f'log file: {log}')
    elif args.run_lvs:
        mode = 'LVS_RCX' if args.run_rcx else 'LVS'
        print(f'Running {mode}')
        success, log = prj.run_lvs(args.lib, args.cell, run_rcx=args.run_rcx)
        if success:
            print(f'{mode} passed!')
        else:
            print(f'{mode} failed...')
        print(f'log file: {log}')
    elif args.run_rcx:
        print('Running RCX')
        netlist, log = prj.run_rcx(args.lib, args.cell)
        if netlist:
            print('RCX passed!')
        else:
            print('RCX failed...')
        print(f'log file: {log}')
    else:
        print('No operation specifiied, do nothing.')


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
