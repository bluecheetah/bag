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

from typing import Mapping, Any

import argparse

from pybag.enum import LogLevel

from bag.io import read_yaml
from bag.core import BagProject
from bag.util.misc import register_pdb_hook

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run EMX on cell from spec file.')
    parser.add_argument('specs', help='Design specs file name.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='Print only warning messages or above.')
    parser.add_argument('--force_sim', action='store_true', default=False,
                        help='Force simulation even if dut is unchanged')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs: Mapping[str, Any] = read_yaml(args.specs)

    log_level = LogLevel.WARN if args.quiet else LogLevel.INFO
    prj.em_cell(specs, force_sim=args.force_sim, log_level=log_level)


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
