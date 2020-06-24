# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Blue Cheetah Analog Design Inc.
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

from __future__ import annotations

from typing import Any, Optional, cast, Type, Mapping, Dict, Union, List

import abc
import pkg_resources
from pathlib import Path

from bag.core import BagProject
from bag.io.file import read_yaml
from bag.design.database import ModuleDB
from bag.layout.template import TemplateDB


class DesignerBase(abc.ABC):

    def __init__(self, bprj: BagProject, spec_file: str = '',
                 spec_dict: Optional[Mapping[str, Any]] = None,
                 sch_db: Optional[ModuleDB] = None, lay_db: Optional[TemplateDB] = None,
                 extract: bool = False) -> None:
        if spec_dict:
            params = spec_dict
        else:
            params = read_yaml(spec_file)

        self.params = cast(Dict[str, Any], params)

        self._root_dir = Path(self.params['root_dir']).resolve()

        self._prj = bprj

        if sch_db is None:
            self._sch_db = ModuleDB(bprj.tech_info, self.params['impl_lib'], prj=bprj)
        else:
            self._sch_db = sch_db

        if lay_db is None:
            self._lay_db = TemplateDB(bprj.grid, self.params['impl_lib'], prj=bprj)
        else:
            self._lay_db = lay_db

        self.extract = extract
        self.data = {}  # a dictionary to access package resources
        self.designed_params = {}  # the parameters after design has been done
        self.designed_performance = {}  # the performance metrics that designed params satisfy

    @classmethod
    def get_schematic_class(cls):
        return None

    @classmethod
    def get_layout_class(cls):
        return None

    @property
    def sch_db(self):
        return self._sch_db

    @property
    def lay_db(self):
        return self._lay_db

    def new_designer(self, cls: Type[DesignerBase], params: Mapping[str, Any],
                     extract: bool):
        return cls(self._prj, spec_dict=params, sch_db=self._sch_db, lay_db=self._lay_db,
                   extract=extract)

    def register_resources(self, resource_names: Union[str, List[str]]) -> None:

        if isinstance(resource_names, str):
            resource_names = [resource_names]
        for name in resource_names:
            module_name = self.__module__.split('.')[-1]
            fpath = str(Path('data', module_name, f'{name}.yaml'))
            yaml_file = pkg_resources.resource_filename(self.__module__, fpath)
            self.data[name] = yaml_file

    @abc.abstractmethod
    def design(self, *args, **kwargs) -> None:
        pass
