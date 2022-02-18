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

"""This module handles abstract generation
"""

from __future__ import annotations

from typing import Dict, Any, Sequence, Type

from pathlib import Path

from ..env import get_bag_work_dir
from ..io.file import write_file
from ..io.template import new_template_env_fs
from ..concurrent.core import SubProcessManager
from ..util.importlib import import_class

from .lef import LEFInterface


class AbstractInterface(LEFInterface):
    """A class that creates LEF using the abstract generator.

    Parameters
    ----------
    config : Dict[str, Any]
        the configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        LEFInterface.__init__(self, config)

        mgr_class: Type[SubProcessManager] = import_class(config.get('mgr_class', SubProcessManager))
        mgr_kwargs: Dict[str, Any] = config.get('mgr_kwargs', {})

        self._manager: SubProcessManager = mgr_class(max_workers=1, **mgr_kwargs)
        self._temp_env_fs = new_template_env_fs()

    def generate_lef(self, impl_lib: str, impl_cell: str, verilog_path: Path, lef_path: Path,
                     run_path: Path, pwr_pins: Sequence[str], gnd_pins: Sequence[str],
                     clk_pins: Sequence[str], analog_pins: Sequence[str],
                     output_pins: Sequence[str], detailed_layers: Sequence[str],
                     cover_layers: Sequence[str],  cell_type: str, **kwargs: Any) -> bool:
        run_path.mkdir(parents=True, exist_ok=True)

        # create options file
        options_path = (run_path / 'bag_abstract.options').resolve()
        self._create_options_file(options_path, pwr_pins, gnd_pins, clk_pins, analog_pins,
                                  output_pins, detailed_layers, cover_layers, cell_type, impl_cell)

        # create replay file
        parent_dir: Path = lef_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        content = self.render_file_template('abstract.replay',
                                            dict(lib_name=impl_lib, cell_name=impl_cell,
                                                 options_file=str(options_path),
                                                 verilog_file=str(verilog_path),
                                                 lef_file=str(lef_path)))
        replay_path = run_path / 'bag_abstract.replay'
        write_file(replay_path, content)

        log_path = run_path / f'bag_abstract.log'
        log_file = str(log_path)
        cwd = get_bag_work_dir()
        pinfo_list = [(['abstract', '-replay', str(replay_path), '-nogui'], log_file, None, cwd)]
        self._manager.batch_subprocess(pinfo_list)

        return lef_path.is_file()

    def _create_options_file(self, out_file: Path, pwr_pins: Sequence[str],
                             gnd_pins: Sequence[str], clk_pins: Sequence[str],
                             analog_pins: Sequence[str], output_pins: Sequence[str],
                             detailed_layers: Sequence[str], cover_layers: Sequence[str],
                             cell_type: str, impl_cell: str) -> None:
        options_file: str = self.config['options_file']

        options_path = Path(options_file).resolve()

        # check options file exists
        if not options_path.is_file():
            raise ValueError(f'Cannot find abstract options template file: {options_path}')

        template = self._temp_env_fs.get_template(str(options_path))

        pwr_regexp = _get_pin_regexp(pwr_pins)
        gnd_regexp = _get_pin_regexp(gnd_pins)
        clk_regexp = _get_pin_regexp(clk_pins)
        ana_regexp = _get_pin_regexp(analog_pins)
        out_regexp = _get_pin_regexp(output_pins)

        detail_str = ' '.join(detailed_layers)
        cover_str = ' '.join(cover_layers)

        content = template.render(pwr_regexp=pwr_regexp, gnd_regexp=gnd_regexp,
                                  clk_regexp=clk_regexp, ana_regexp=ana_regexp,
                                  out_regexp=out_regexp, detail_blk=detail_str,
                                  cover_blk=cover_str, cell_type=cell_type, impl_cell=impl_cell)

        write_file(out_file, content)


def _get_pin_regexp(pin_list: Sequence[str]) -> str:
    if not pin_list:
        return ''
    elif len(pin_list) == 1:
        return f'^{pin_list[0]}$'

    return f'^({"|".join(pin_list)})$'
