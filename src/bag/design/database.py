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

"""This module defines the design database class.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Mapping, Optional, Any, Sequence, Type, Tuple

import importlib
from pathlib import Path

from jinja2 import Template

from pybag.enum import DesignOutput, LogLevel

from ..util.cache import MasterDB, Param
from ..io.template import new_template_env_fs

from .module import Module

if TYPE_CHECKING:
    from ..core import BagProject
    from ..layout.tech import TechInfo

ModuleType = TypeVar('ModuleType', bound=Module)


class ModuleDB(MasterDB):
    """A database of all modules.

    This class is a subclass of MasterDB that defines some extra properties/function
    aliases to make creating schematics easier.

    Parameters
    ----------
    tech_info : TechInfo
        the TechInfo instance.
    lib_name : str
        the cadence library to put all generated templates in.
    log_file: str
        the log file path.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated schematic name prefix.
    name_suffix : str
        generated schematic name suffix.
    log_level : LogLevel
        the logging level.
    """

    def __init__(self, tech_info: TechInfo, lib_name: str, log_file: str, prj: Optional[BagProject] = None,
                 name_prefix: str = '', name_suffix: str = '', log_level: LogLevel = LogLevel.DEBUG) -> None:
        MasterDB.__init__(self, lib_name, log_file, prj=prj, name_prefix=name_prefix, name_suffix=name_suffix,
                          log_level=log_level)

        self._tech_info = tech_info
        self._temp_env = new_template_env_fs()

    @classmethod
    def get_schematic_class(cls, lib_name: str, cell_name: str) -> Type[Module]:
        """Get the Python class object for the given schematic.

        Parameters
        ----------
        lib_name : str
            schematic library name.
        cell_name : str
            schematic cell name.

        Returns
        -------
        sch_cls : Type[Module]
            the schematic class.
        """
        module_name = lib_name + '.schematic.' + cell_name
        try:
            sch_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError('Cannot find Python module {} for schematic generator {}__{}.  '
                              'Is it on your PYTHONPATH?'.format(module_name, lib_name, cell_name))
        cls_name = lib_name + '__' + cell_name
        if not hasattr(sch_module, cls_name):
            raise ImportError('Cannot find schematic generator class {} '
                              'in module {}'.format(cls_name, module_name))
        return getattr(sch_module, cls_name)

    @property
    def tech_info(self) -> TechInfo:
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def get_model_netlist_template(self, fpath: Path) -> Template:
        return self._temp_env.get_template(str(fpath))

    def instantiate_schematic(self, design: Module, top_cell_name: str = '',
                              output: DesignOutput = DesignOutput.SCHEMATIC,
                              **kwargs: Any) -> None:
        """Alias for instantiate_master(), with default output type of SCHEMATIC.
        """
        self.instantiate_master(output, design, top_cell_name, **kwargs)

    def batch_schematic(self, info_list: Sequence[Tuple[Module, str]],
                        output: DesignOutput = DesignOutput.SCHEMATIC,
                        **kwargs: Any) -> None:
        """Alias for batch_output(), with default output type of SCHEMATIC.
        """
        self.batch_output(output, info_list, **kwargs)

    def new_model(self, master: Module, model_params: Param, **kwargs: Any) -> Module:
        """Create a new schematic master instance with behavioral model information

        Parameters
        ----------
        master : Module
            the schematic master instance.
        model_params : Param
            model parameters.
        **kwargs : Any
            optional arguments

        Returns
        -------
        master : Module
            the new master instance.
        """
        debug = kwargs.get('debug', False)

        new_params = master.params.copy(append=dict(model_params=model_params))
        key = master.compute_unique_key(new_params)
        test = self.find_master(key)
        if test is not None:
            if debug:
                print('model master cached')
            return test

        if debug:
            print('generating model master')
        new_master = master.get_copy_with(new_params)
        new_master.design_model(key)
        self.register_master(key, new_master)
        return new_master

    def instantiate_model(self, design: Module, model_params: Param, top_cell_name: str = '',
                          **kwargs: Any) -> None:
        self.batch_model([(design, top_cell_name, model_params)], **kwargs)

    def batch_model(self, info_list: Sequence[Tuple[Module, str, Mapping[str, Any]]],
                    output: DesignOutput = DesignOutput.SYSVERILOG,
                    **kwargs: Any) -> Sequence[Tuple[Module, str]]:
        new_info_list = [(self.new_model(m, Param(m_params)), name)
                         for m, name, m_params in info_list]
        self.batch_output(output, new_info_list, **kwargs)
        return new_info_list
