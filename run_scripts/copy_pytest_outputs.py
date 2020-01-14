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
from shutil import copy
from pathlib import Path

from bag.util.misc import register_pdb_hook

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Copy pytest output data.')
    parser.add_argument('data_dir', help='data directory.')
    parser.add_argument('package', help='package name.')
    parser.add_argument('--cell',dest='cell',  help='cell name.', default='')
    parser.add_argument('--lay', dest='copy_lay', action='store_true', default=False,
                        help='copy layout files.')
    args = parser.parse_args()
    return args


def run_main(args: argparse.Namespace) -> None:
    root_dir = Path(args.data_dir)
    pkg_name: str = args.package
    cell_name: str = args.cell
    copy_lay: bool = args.copy_lay

    # check data directory exists
    cur_dir = root_dir / pkg_name
    if not cur_dir.is_dir():
        raise ValueError(f'package data directory {cur_dir} is not a directory')

    src_dir = Path('pytest_output', pkg_name)
    if not src_dir.is_dir():
        raise ValueError(f'Cannot find pytest output directory {src_dir}')
    for p in src_dir.iterdir():
        if p.is_dir():
            tokens = p.name.rsplit('_', maxsplit=1)
            if not cell_name or tokens[0] == cell_name:
                dst_dir = cur_dir / p.name
                if not dst_dir.is_dir():
                    continue

                for fpath in p.iterdir():
                    if not copy_lay and fpath.name.endswith('gds'):
                        continue
                    copy(str(fpath), str(dst_dir / fpath.name))


if __name__ == '__main__':
    _args = parse_options()

    run_main(_args)
