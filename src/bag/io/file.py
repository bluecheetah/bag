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

"""This module handles file related IO.
"""

from typing import TextIO, Any, Iterable, Union, Dict

import os
import time
import string
import codecs
import tempfile
import pkg_resources
from pathlib import Path
import jinja2

from ruamel.yaml import YAML

from .common import bag_encoding, bag_codec_error

yaml = YAML(typ='unsafe')


def render_yaml(fname: Union[str, Path], params: Dict[str, Any]) -> Dict[str, Any]:
    """Renders a yaml file as a jinja template.

    Parameters
    ----------
    fname: Union[str, Path]
        the yaml file name.
    params: Dict[str, Any]
        parameters to be replaced in the yaml template

    Returns
    -------
    rendered_content: Dict[str, Any]
        A dictionary with keywords replaced in the yaml file.
    """
    raw_content = read_file(fname)
    populated_content = jinja2.Template(raw_content).render(**params)
    return yaml.load(populated_content)


def open_file(fname: Union[str, Path], mode: str) -> TextIO:
    """Opens a file with the correct encoding interface.

    Use this method if you need to have a file handle.

    Parameters
    ----------
    fname : str
        the file name.
    mode : str
        the mode, either 'r', 'w', or 'a'.

    Returns
    -------
    file_obj : TextIO
        a file objects that reads/writes string with the BAG system encoding.
    """
    if mode != 'r' and mode != 'w' and mode != 'a':
        raise ValueError("Only supports 'r', 'w', or 'a' mode.")
    return open(fname, mode, encoding=bag_encoding, errors=bag_codec_error)


def read_file(fname: Union[str, Path]) -> str:
    """Read the given file and return content as string.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.

    Returns
    -------
    content : str
        the content as a unicode string.
    """
    with open_file(fname, 'r') as f:
        content = f.read()
    return content


def readlines_iter(fname: Union[str, Path]) -> Iterable[str]:
    """Iterate over lines in a file.

    Parameters
    ----------
    fname : str
        the file name.

    Yields
    ------
    line : str
        a line in the file.
    """
    with open_file(fname, 'r') as f:
        for line in f:
            yield line


def read_yaml(fname: Union[str, Path]) -> Any:
    """Read the given file using YAML.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    with open_file(fname, 'r') as f:
        content = yaml.load(f)

    return content


def read_yaml_env(fname: str) -> Any:
    """Parse YAML file with environment variable substitution.

    Parameters
    ----------
    fname : str
        yaml file name.

    Returns
    -------
    table : Any
        the object returned by YAML.
    """
    content = read_file(fname)
    # substitute environment variables
    content = string.Template(content).substitute(os.environ)
    return yaml.load(content)


def read_resource(package: str, fname: str) -> str:
    """Read the given resource file and return content as string.

    Parameters
    ----------
    package : str
        the package name.
    fname : str
        the resource file name.

    Returns
    -------
    content : str
        the content as a unicode string.
    """
    raw_content = pkg_resources.resource_string(package, fname)
    return raw_content.decode(encoding=bag_encoding, errors=bag_codec_error)


def write_file(fname: Union[str, Path], content: str, append: bool = False,
               mkdir: bool = True) -> None:
    """Writes the given content to file.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    content : str
        the unicode string to write to file.
    append : bool
        True to append instead of overwrite.
    mkdir : bool
        If True, will create parent directories if they don't exist.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open_file(fpath, 'a' if append else 'w') as f:
        f.write(content)


def write_yaml(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using YAML format.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open_file(fpath, 'w') as f:
        yaml.dump(obj, f)


def make_temp_dir(prefix: str, parent_dir: str = '') -> str:
    """Create a new temporary directory.

    Parameters
    ----------
    prefix : str
        the directory prefix.
    parent_dir : str
        the parent directory.

    Returns
    -------
    dir_name : str
        the temporary directory name.
    """
    prefix += time.strftime("_%Y%m%d_%H%M%S")
    parent_dir = parent_dir or tempfile.gettempdir()
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=parent_dir)


def open_temp(**kwargs: Any) -> TextIO:
    """Opens a new temporary file for writing with unicode interface.

    Parameters
    ----------
    **kwargs : Any
        the tempfile keyword arguments.  See documentation for
        :func:`tempfile.NamedTemporaryFile`.

    Returns
    -------
    file : TextIO
        the opened file that accepts unicode input.
    """
    timestr = time.strftime("_%Y%m%d_%H%M%S")
    if 'prefix' in kwargs:
        kwargs['prefix'] += timestr
    else:
        kwargs['prefix'] = timestr
    temp = tempfile.NamedTemporaryFile(**kwargs)
    return codecs.getwriter(bag_encoding)(temp, errors=bag_codec_error)
