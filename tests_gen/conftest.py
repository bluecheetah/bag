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

from typing import Dict, Any, List

import importlib
from pathlib import Path

import pytest

from bag.io.file import read_yaml
from bag.core import BagProject


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, dict) and isinstance(right, dict) and op == '==':
        return get_dict_diff_msg(left, right)


def pytest_addoption(parser):
    parser.addoption(
        '--data_root', action='store', default='', help='test data root directory',
    )
    parser.addoption(
        '--package', action='store', default='', help='generator package to test',
    )
    parser.addoption(
        '--gen_output', action='store_true', default=False,
        help='True to generate expected outputs',
    )
    parser.addoption(
        '--run_lvs', action='store_true', default=False,
        help='True to run LVS',
    )


def get_dict_diff_msg(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    ans = ['Comparing (Nested) Dictionaries:']
    get_dict_diff_msg_helper(left, right, ans, [])
    return ans


def get_dict_diff_msg_helper(left: Dict[str, Any], right: Dict[str, Any], msgs: List[str],
                             prefix: List[str]) -> None:
    keys1 = sorted(left.keys())
    keys2 = sorted(right.keys())

    idx1 = 0
    idx2 = 0
    n1 = len(keys1)
    n2 = len(keys2)
    prefix_str = ','.join(prefix)
    while idx1 < n1 and idx2 < n2:
        k1 = keys1[idx1]
        k2 = keys2[idx2]
        v1 = left[k1]
        v2 = right[k2]
        if k1 == k2:
            if v1 != v2:
                if isinstance(v1, dict) and isinstance(v2, dict):
                    next_prefix = prefix.copy()
                    next_prefix.append(k1)
                    get_dict_diff_msg_helper(v1, v2, msgs, next_prefix)
                else:
                    msgs.append(f'L[{prefix_str},{k1}]:')
                    msgs.append(f'{v1}')
                    msgs.append(f'R[{prefix_str},{k1}]:')
                    msgs.append(f'{v2}')
            idx1 += 1
            idx2 += 1
        elif k1 < k2:
            msgs.append(f'R[{prefix_str}] missing key: {k1}')
            idx1 += 1
        else:
            msgs.append(f'L[{prefix_str}] missing key: {k2}')
            idx2 += 1
    while idx1 < n1:
        msgs.append(f'R[{prefix_str}] missing key: {keys1[idx1]}')
        idx1 += 1
    while idx2 < n2:
        msgs.append(f'L[{prefix_str}] missing key: {keys2[idx2]}')
        idx2 += 1


def get_test_data_id(data: Dict[str, Any]) -> str:
    return data['pytest']['test_id']


def pytest_generate_tests(metafunc):
    if 'dsn_specs' not in metafunc.fixturenames:
        return

    pkg_name = metafunc.config.getoption('package')
    root_dir = Path(metafunc.config.getoption('--data_root'))

    # get list of packages
    if pkg_name:
        # check package is importable
        try:
            importlib.import_module(pkg_name)
        except ImportError:
            raise ImportError(f"Cannot find python package {pkg_name}, "
                              "make sure it's on your PYTHONPATH")

        # check data directory exists
        tmp = root_dir / pkg_name
        if not tmp.is_dir():
            raise ValueError(f'package data directory {tmp} is not a directory')
        pkg_iter = [pkg_name]
    else:
        pkg_iter = (d.name for d in root_dir.iterdir() if d.is_dir())

    data = []

    for pkg in pkg_iter:
        cur_dir = root_dir / pkg
        if not cur_dir.is_dir():
            continue

        for p in cur_dir.iterdir():
            if p.is_dir():
                test_id = p.name
                # noinspection PyTypeChecker
                content = read_yaml(p / 'specs.yaml')
                # inject fields
                root_path = Path('pytest_output', pkg, test_id).resolve()
                content['root_dir'] = str(root_path)
                content['impl_lib'] = 'PYTEST'
                content['impl_cell'] = test_id
                content['layout_file'] = str(root_path / 'out.gds')
                content['yaml_file'] = str(root_path / 'out.yaml')
                content['netlist_file'] = str(root_path / 'out.netlist')
                content['model_file'] = str(root_path / 'out.model')
                pytest_info = dict(test_id=f'{pkg}__{test_id}')
                for fpath in p.iterdir():
                    if fpath.stem.startswith('out'):
                        pytest_info[f'{fpath.stem}_{fpath.suffix[1:]}'] = fpath.absolute()
                content['pytest'] = pytest_info

                data.append(content)
    if data:
        metafunc.parametrize('dsn_specs', data, indirect=True, ids=get_test_data_id)


@pytest.fixture(scope='session')
def root_test_dir() -> Path:
    ans = Path('pytest_output')
    ans.mkdir(parents=True, exist_ok=True)
    return ans


@pytest.fixture(scope='session')
def gen_output(request) -> bool:
    return request.config.getoption("--gen_output")


@pytest.fixture(scope='session')
def run_lvs(request) -> bool:
    return request.config.getoption("--run_lvs")


@pytest.fixture(scope='session')
def bag_project() -> BagProject:
    return BagProject()


@pytest.fixture(scope='session')
def dsn_specs(request) -> Dict[str, Any]:
    return request.param
