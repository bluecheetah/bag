#!/usr/bin/env bash
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

# crazy black magic from:
# https://unix.stackexchange.com/questions/20880/how-can-i-use-environment-variables-in-my-shebang
# this block of code is valid in both bash and python.
# this means if this script is run under bash, it'll
# call this script again using BAG_PYTHON.  If
# this script is run under Python, this block of code
# effectively does nothing.
if "true" : '''\'
then
if [[ $BAG_PYTHON ]]; then
exec ${BAG_PYTHON} "$0" "$@"
else
echo "ERROR! BAG_PYTHON environment variable is not set"
fi
exit 127
fi
'''
from typing import List, Tuple, Optional, Dict, Any

import subprocess
from pathlib import Path

from ruamel.yaml import YAML
yaml = YAML()

BAG_DIR = 'BAG_framework'


def write_to_file(fname: str, lines: List[str]) -> None:
    with open(fname, 'w') as f:
        f.writelines((l + '\n' for l in lines))
    add_git_file(fname)


def setup_python_path(module_list: List[Tuple[str, Dict[str, Any]]], repo_path: Path) -> None:
    lines = ['#!/usr/bin/env bash',
             '',
             'export PYTHONPATH="${BAG_FRAMEWORK}/src"',
             'export PYTHONPATH="${PYTHONPATH}:${BAG_FRAMEWORK}/pybag/_build/lib"',
             'export PYTHONPATH="${PYTHONPATH}:${BAG_TECH_CONFIG_DIR}/src"',
             ]
    template = 'export PYTHONPATH="${PYTHONPATH}:${BAG_WORK_DIR}/%s'
    for mod_name, _ in module_list:
        if mod_name != BAG_DIR:
            root = template % mod_name
            lines.append(root + '/src"')
            inc_path = repo_path / mod_name / 'bag_pypath'
            if inc_path.is_file():
                # noinspection PyTypeChecker
                with open(inc_path, 'r') as f:
                    for line in f:
                        cur = line.strip()
                        if cur:
                            lines.append(f'{root}/{cur}"')

    lines.append('export PYTHONPATH="${PYTHONPATH}:${PYTHONPATH_CUSTOM:-}"')
    write_to_file('.bashrc_pypath', lines)


def get_oa_libraries(mod_name: str) -> List[str]:
    root_dir = Path(mod_name, 'OA').resolve()
    if root_dir.is_dir():
        return [str(sub.stem) for sub in root_dir.iterdir() if sub.is_dir()]

    return []


def get_sch_libraries(mod_name: str) -> List[str]:
    root_dir = Path(mod_name, 'src')
    if root_dir.is_dir():
        return [str(sub.stem) for sub in root_dir.iterdir()
                if sub.is_dir() and (sub / 'schematic').is_dir()]
    return []


def setup_libs_def(module_list: List[Tuple[str, Dict[str, Any]]]) -> None:
    lines = ['BAG_prim']
    for mod_name, _ in module_list:
        for lib_name in get_sch_libraries(mod_name):
            lines.append(lib_name)

    write_to_file('bag_libs.def', lines)


def setup_cds_lib(module_list: List[Tuple[str, Dict[str, Any]]]) -> None:
    lines = ['DEFINE BAG_prim $BAG_TECH_CONFIG_DIR/OA/BAG_prim']
    template = 'DEFINE {} $BAG_WORK_DIR/{}/OA/{}'
    for mod_name, _ in module_list:
        for lib_name in get_oa_libraries(mod_name):
            lines.append(template.format(lib_name, mod_name, lib_name))

    write_to_file('cds.lib.bag', lines)


def run_command(cmd: List[str], cwd: Optional[str] = None, get_output: bool = False) -> str:
    timeout = 5
    print(f'cmd: {" ".join(cmd)}, cwd: {cwd}')
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE if get_output else None)
    output = ''
    try:
        output = proc.communicate()[0]
        if output is not None:
            output = output.decode('utf-8').strip()
    except KeyboardInterrupt:
        print('Ctrl-C detected, terminating')
        if proc.returncode is None:
            proc.terminate()
            print('terminating process...')
            try:
                proc.wait(timeout=timeout)
                print('process terminated')
            except subprocess.TimeoutExpired:
                proc.kill()
                print('process did not terminate, try killing...')
                try:
                    proc.wait(timeout=timeout)
                    print('process killed')
                except subprocess.TimeoutExpired:
                    print('cannot kill process...')

    if proc.returncode is None:
        raise ValueError('Ctrl-C detected, but cannot kill process')
    elif proc.returncode < 0:
        raise ValueError('process terminated with return code = %d' % proc.returncode)
    elif proc.returncode > 0:
        raise ValueError('command %s failed' % ' '.join(cmd))

    if get_output:
        print('output: ' + output)
    return output


def add_git_submodule(module_name: str, url: str, branch: str) -> None:
    mpath = Path(module_name)
    if not mpath.exists():
        run_command(['git', 'submodule', 'add', '-b', branch, url])


def add_git_file(fname: str) -> None:
    run_command(['git', 'add', '-f', fname])


def link_submodule(repo_path: str, module_name: str) -> None:
    mpath = Path(module_name)
    if mpath.exists():
        # skip if already exists
        return

    src = Path(repo_path, module_name)
    if not src.is_dir():
        raise ValueError('Cannot find submodule %s in %s' % (module_name, repo_path))
    mpath.symlink_to(src)
    add_git_file(module_name)


def setup_git_submodules(module_list: List[Tuple[str, Dict[str, Any]]]) -> None:
    for module_name, module_info in module_list:
        add_git_submodule(module_name, module_info['url'], module_info.get('branch', 'master'))


def setup_submodule_links(module_list: List[Tuple[str, Dict[str, Any]]], repo_path: str) -> None:
    for module_name, _ in module_list:
        link_submodule(repo_path, module_name)


def run_main() -> None:
    default_submodules = {
        BAG_DIR: {
            'url': 'git@github.com:bluecheetah/bag.git',
        },
    }

    with open('bag_submodules.yaml', 'r') as f:
        modules_info = yaml.load(f)

    # add default submodules
    for name, info in default_submodules.items():
        if name not in modules_info:
            modules_info[name] = info

    module_list = [(key, modules_info[key]) for key in sorted(modules_info.keys())]

    # error checking
    bag_dir_path = Path(BAG_DIR).resolve()
    if not bag_dir_path.is_dir():
        raise ValueError('Cannot find directory %s' % BAG_DIR)

    # get real absolute path of parent directory of BAG_framework
    repo_path = bag_dir_path.parent
    cur_path = Path('.').resolve()
    if cur_path == repo_path:
        # BAG_framework is an actual directory in this repo; add dependencies as git submodules
        setup_git_submodules(module_list)
    else:
        setup_submodule_links(module_list, repo_path)

    setup_python_path(module_list, repo_path)
    setup_libs_def(module_list)
    setup_cds_lib(module_list)


if __name__ == '__main__':
    run_main()
