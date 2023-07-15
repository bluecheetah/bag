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

"""This module defines various methods to query information about the design environment.
"""

from typing import Tuple, Dict, Any, Optional, Type, List, cast

import os
import socket
import string
from pathlib import Path

from .io.file import read_file, read_yaml_env, read_yaml
from .layout.tech import TechInfo
from .layout.routing import RoutingGrid
from .util.importlib import import_class


def get_bag_work_path() -> Path:
    """Returns the BAG working directory."""
    work_dir = os.environ.get('BAG_WORK_DIR', '')
    if not work_dir:
        raise ValueError('Environment variable BAG_WORK_DIR not defined.')
    work_path = Path(work_dir).resolve()
    if not work_path.is_dir():
        raise ValueError(f'$BAG_WORK_DIR = "{work_dir}" is not a directory')

    return work_path


def get_bag_tmp_path() -> Path:
    """Returns the BAG temporary files directory."""
    tmp_dir = os.environ.get('BAG_TEMP_DIR', '')
    if not tmp_dir:
        raise ValueError('Environment variable BAG_TEMP_DIR not defined.')
    tmp_path = Path(tmp_dir).resolve()
    tmp_path.mkdir(parents=True, exist_ok=True)
    if not tmp_path.is_dir():
        raise ValueError(f'$BAG_TEMP_DIR = "{tmp_dir}" is not a directory')

    return tmp_path


def get_tech_path() -> Path:
    """Returns the technology directory."""
    tech_dir = os.environ.get('BAG_TECH_CONFIG_DIR', '')
    if not tech_dir:
        raise ValueError('Environment variable BAG_TECH_CONFIG_DIR not defined.')
    tech_path = Path(tech_dir).resolve()
    if not tech_path.is_dir():
        raise ValueError('BAG_TECH_CONFIG_DIR = "{}" is not a directory'.format(tech_dir))

    return tech_path


def get_bag_work_dir() -> str:
    """Returns the BAG working directory."""
    return str(get_bag_work_path())


def get_bag_tmp_dir() -> str:
    """Returns the BAG temporary files directory."""
    return str(get_bag_tmp_path())


def get_tech_dir() -> str:
    """Returns the technology directory."""
    return str(get_tech_path())


def get_bag_config() -> Dict[str, Any]:
    """Returns the BAG configuration dictioanry."""
    bag_config_path = os.environ.get('BAG_CONFIG_PATH', '')
    if not bag_config_path:
        raise ValueError('Environment variable BAG_CONFIG_PATH not defined.')

    return read_yaml_env(bag_config_path)


def get_tech_params(bag_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Returns the technology parameters dictioanry.

    Parameters
    ----------
    bag_config : Optional[Dict[str, Any]]
        the BAG configuration dictionary.  If None, will try to read it from file.

    Returns
    -------
    tech_params : Dict[str, Any]
        the technology configuration dictionary.
    """
    if bag_config is None:
        bag_config = get_bag_config()

    fname = bag_config['tech_config_path']
    ans = read_yaml_env(bag_config['tech_config_path'])
    ans['tech_config_fname'] = fname
    return ans


def create_tech_info(bag_config: Optional[Dict[str, Any]] = None) -> TechInfo:
    """Create TechInfo object."""
    tech_params = get_tech_params(bag_config=bag_config)

    if 'class' in tech_params:
        tech_cls = cast(Type[TechInfo], import_class(tech_params['class']))
        tech_info = tech_cls(tech_params)
    else:
        # just make a default tech_info object as place holder.
        print('*WARNING*: No TechInfo class defined.  Using a dummy version.')
        tech_info = TechInfo(tech_params, {}, '')

    return tech_info


def create_routing_grid(tech_info: Optional[TechInfo] = None,
                        bag_config: Optional[Dict[str, Any]] = None) -> RoutingGrid:
    """Create RoutingGrid object."""
    if tech_info is None:
        tech_info = create_tech_info(bag_config=bag_config)
    return RoutingGrid(tech_info, tech_info.tech_params['tech_config_fname'])


def create_routing_grid_from_file(config_fname: str, tech_info: Optional[TechInfo] = None,
                                  bag_config: Optional[Dict[str, Any]] = None) -> RoutingGrid:
    """Create RoutingGrid object from the given config file."""
    if tech_info is None:
        tech_info = create_tech_info(bag_config=bag_config)
    return RoutingGrid(tech_info, string.Template(config_fname).substitute(os.environ))


def can_connect_to_port(port: int) -> bool:
    """Check if we can successfully connect to a port.

    Used to check if Virtuoso server is up.
    """
    s = socket.socket()
    try:
        s.connect(('localhost', port))
        return True
    except socket.error:
        return False
    finally:
        s.close()


def get_port_number(bag_config: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    """Read the port number from the port file..

    Parameters
    ----------
    bag_config : Optional[Dict[str, Any]]
        the BAG configuration dictionary.  If None, will try to read it from file.

    Returns
    -------
    port : int
        the port number.  Negative on failure.
    msg : str
        Empty string on success, the error message on failure.
    """
    if bag_config is None:
        bag_config = get_bag_config()

    port_file = get_bag_work_path() / bag_config['socket']['port_file']
    try:
        port = int(read_file(port_file))
    except ValueError as err:
        return -1, str(err)
    except FileNotFoundError as err:
        return -1, str(err)

    if can_connect_to_port(port):
        return port, ''
    return -1, f'Cannot connect to port {port}'


def get_netlist_setup_file() -> str:
    """Returns the netlist setup file path."""
    ans = get_tech_path() / 'netlist_setup' / 'netlist_setup.yaml'
    if not ans.is_file():
        raise ValueError(f'{ans} is not a file.')
    return str(ans)


def get_gds_layer_map() -> str:
    """Returns the GDS layer map file."""
    ans = get_tech_path() / 'gds_setup' / 'gds.layermap'
    if not ans.is_file():
        raise ValueError(f'{ans} is not a file.')
    return str(ans)


def get_gds_object_map() -> str:
    """Returns the GDS object map file."""
    ans = get_tech_path() / 'gds_setup' / 'gds.objectmap'
    if not ans.is_file():
        raise ValueError(f'{ans} is not a file.')
    return str(ans)


def get_bag_device_map(name: str) -> List[Tuple[str, str]]:
    config_path = get_tech_path() / 'netlist_setup' / 'gen_config.yaml'
    config = read_yaml(config_path)
    return config[name]['types']


def get_tech_global_info(prj_name: str) -> Dict[str, Any]:
    path = f'data/{prj_name}/specs_global/tech_global.yaml'
    return read_yaml(path)
