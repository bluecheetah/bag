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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, Type, Mapping, Dict, Optional, Sequence, cast, Tuple

import abc
import pprint
from pathlib import Path
from copy import deepcopy

from pybag.enum import LogLevel

from ..core import BagProject
from ..util.importlib import import_class
from ..util.logging import LoggingBase
from ..io.file import write_yaml
from ..io import read_yaml
from ..concurrent.core import batch_async_task
from ..design.module import Module
from ..layout.tech import TechInfo
from ..layout.routing.grid import RoutingGrid
from ..layout.template import TemplateBase
from .core import TestbenchManager
from .measure import MeasurementManager
from .cache import SimulationDB, SimResults, MeasureResult, DesignInstance

from bag3_digital.measurement.liberty.io import generate_liberty


class DesignerBase(LoggingBase, abc.ABC):
    """Base class of all design scripts.

    Notes
    -----
    1. This class hides the SimulationDB object from the user.  This is because hierarchical
       designers share the same SimulationDB, and if you don't make sure to update the
       working directory every time you run

    """

    def __init__(self, root_dir: Path, sim_db: SimulationDB, dsn_specs: Mapping[str, Any]) -> None:
        cls_name = self.__class__.__name__
        super().__init__(cls_name, sim_db.log_file, log_level=sim_db.log_level)

        self._root_dir = root_dir
        self._work_dir = root_dir
        self._sim_db = sim_db
        self._dsn_specs = {k: deepcopy(v) for k, v in dsn_specs.items()}

        self.commit()

    @property
    def tech_info(self) -> TechInfo:
        return self._sim_db.prj.tech_info

    @property
    def grid(self) -> RoutingGrid:
        return self._sim_db.prj.grid

    @property
    def dsn_specs(self) -> Dict[str, Any]:
        return self._dsn_specs

    @property
    def extract(self) -> bool:
        return self._sim_db.extract

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_dut_class_info(cls, gen_specs: Mapping[str, Any]) -> Tuple[bool, Union[Type[TemplateBase], Type[Module]]]:
        """Returns information about the DUT generator class.

        Parameters
        ----------
        specs : Param
            The generator specs.

        Returns
        -------
        is_lay : bool
            True if the DUT generator is a layout generator, False if schematic generator.

        dut_cls : Union[Type[TemplateBase], Type[Module]]
            The DUT generator class.
        """
        is_lay, lay_cls, sch_cls = BagProject.get_dut_class_info(gen_specs)
        return is_lay, lay_cls or sch_cls

    @classmethod
    def design_cell(cls, prj: BagProject, specs: Mapping[str, Any], extract: bool = False,
                    force_sim: bool = False, force_extract: bool = False, gen_cell: bool = False,
                    gen_cell_dut: bool = False, gen_cell_tb: bool = False, log_level: LogLevel = LogLevel.DEBUG
                    ) -> None:
        dsn_str: Union[str, Type[DesignerBase]] = specs['dsn_class']
        root_dir: Union[str, Path] = specs['root_dir']
        impl_lib: str = specs['impl_lib']
        dsn_params: Mapping[str, Any] = specs['dsn_params']
        precision: int = specs.get('precision', 6)
        gen_cell_dut |= gen_cell
        gen_cell_tb |= gen_cell

        dsn_cls = cast(Type[DesignerBase], import_class(dsn_str))
        root_path = prj.get_root_path(root_dir)

        dsn_options = dict(
            extract=extract,
            force_extract=force_extract,
            gen_sch_dut=gen_cell_dut,
            gen_sch_tb=gen_cell_tb,
            log_level=log_level,
        )
        log_file = str(root_path / 'dsn.log')
        sim_db = prj.make_sim_db(root_path / 'dsn', log_file, impl_lib, dsn_options=dsn_options,
                                 force_sim=force_sim, precision=precision, log_level=log_level)
        designer = dsn_cls(root_path, sim_db, dsn_params)
        summary = designer.run_design()
        if 'gen_specs' in summary:
            prj.generate_cell(summary['gen_specs'], **summary['gen_args'])
            if 'gen_lib' in summary['gen_specs'] and summary['gen_specs']['gen_lib']:
                lib_specs = summary['gen_specs']['lib_specs']
                lib_args = summary['gen_specs']['lib_args']
                lib_file = lib_specs['lib_file']
                lib_file_path = Path(lib_file)
                lib_file_specs = read_yaml(lib_file_path)
                root_dir = lib_file_path.parent
                lib_config = read_yaml(root_dir / 'lib_config.yaml')
                sim_config = read_yaml(root_dir / 'sim_config.yaml')

                generate_liberty(prj, lib_config, sim_config, lib_file_specs, **lib_args)
        else:
            pprint.pprint(summary)

    def get_design_dir(self, parent_dir: Path) -> Path:
        if self.extract:
            return parent_dir / 'extract'
        else:
            return parent_dir / 'schematic'

    def commit(self) -> None:
        """Commit changes to specs dictionary.  Perform necessary initialization."""
        for k, v in self.get_default_param_values().items():
            if k not in self._dsn_specs:
                self._dsn_specs[k] = v

        self._work_dir = self.get_design_dir(self._root_dir / self.__class__.__name__)

    def design(self, **kwargs: Any) -> Mapping[str, Any]:
        coro = self.async_design(**kwargs)
        results = batch_async_task([coro])
        if results is None:
            self.error('Design script cancelled.')

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    @abc.abstractmethod
    async def async_design(self, **kwargs: Any) -> Mapping[str, Any]:
        pass

    def run_design(self) -> Mapping[str, Any]:
        return self.design(**self.dsn_specs)

    def set_dsn_specs(self, specs: Mapping[str, Any]) -> None:
        self._dsn_specs = {k: deepcopy(v) for k, v in specs.items()}
        self.commit()

    def new_designer(self, cls: Union[str, Type[DesignerBase]], dsn_specs: Mapping[str, Any]
                     ) -> DesignerBase:
        dsn_cls = cast(Type[DesignerBase], import_class(cls))
        designer = dsn_cls(self._root_dir, self._sim_db, dsn_specs)
        return designer

    def make_tbm(self, tbm_cls: Union[Type[TestbenchManager], str], tbm_specs: Mapping[str, Any],
                 ) -> TestbenchManager:
        return self._sim_db.make_tbm(tbm_cls, tbm_specs, logger=self.logger)

    def make_mm(self, mm_cls: Union[Type[MeasurementManager], str], meas_specs: Mapping[str, Any]
                ) -> MeasurementManager:
        return self._sim_db.make_mm(mm_cls, meas_specs)

    async def async_batch_dut(self, dut_specs: Sequence[Mapping[str, Any]],
                              rcx_params: Optional[Mapping[str, Any]] = None) -> Sequence[DesignInstance]:
        return await self._sim_db.async_batch_design(dut_specs, rcx_params=rcx_params)

    async def async_new_dut(self, impl_cell: str,
                            dut_cls: Union[Type[TemplateBase], Type[Module], str],
                            dut_params: Mapping[str, Any], extract: Optional[bool] = None,
                            name_prefix: str = '', name_suffix: str = '',
                            flat: bool = False, export_lay: bool = False) -> DesignInstance:
        return await self._sim_db.async_new_design(impl_cell, dut_cls, dut_params, extract=extract,
                                                   name_prefix=name_prefix, name_suffix=name_suffix,
                                                   flat=flat, export_lay=export_lay)

    async def async_new_em_dut(self, impl_cell: str, dut_cls: Union[Type[TemplateBase], str],
                               dut_params: Mapping[str, Any], name_prefix: str = '', name_suffix: str = '',
                               flat: bool = False, export_lay: bool = False) -> Tuple[DesignInstance, Path, bool]:
        return await self._sim_db.async_new_em_design(impl_cell, dut_cls, dut_params,
                                                      name_prefix=name_prefix, name_suffix=name_suffix,
                                                      flat=flat, export_lay=export_lay)

    async def async_simulate_tbm_obj(self, sim_id: str, dut: Optional[DesignInstance],
                                     tbm: TestbenchManager, tb_params: Optional[Mapping[str, Any]],
                                     tb_name: str = '') -> SimResults:
        return await self._sim_db.async_simulate_tbm_obj(sim_id, self._work_dir / sim_id, dut, tbm,
                                                         tb_params, tb_name=tb_name)

    async def async_simulate_mm_obj(self, sim_id: str, dut: Optional[DesignInstance],
                                    mm: MeasurementManager) -> MeasureResult:
        return await self._sim_db.async_simulate_mm_obj(sim_id, self._work_dir / sim_id, dut, mm)

    async def async_gen_nport(self, dut: DesignInstance, gds_file: Path, gds_cached: bool, em_params: Mapping[str, Any],
                              root_path: Path) -> Path:
        return await self._sim_db.async_gen_nport(dut, gds_file, gds_cached, em_params, root_path)
