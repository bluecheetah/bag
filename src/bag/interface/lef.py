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

from typing import Dict, Any, Sequence

import abc

from pathlib import Path

from .base import InterfaceBase


class LEFInterface(InterfaceBase, abc.ABC):
    """An abstract class that defines interface for generating LEF files.

    Parameters
    ----------
    config : Dict[str, Any]
        the configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        InterfaceBase.__init__(self)

        self._config = config

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @abc.abstractmethod
    def generate_lef(self, impl_lib: str, impl_cell: str, verilog_path: Path, lef_path: Path,
                     run_path: Path, pwr_pins: Sequence[str], gnd_pins: Sequence[str],
                     clk_pins: Sequence[str], analog_pins: Sequence[str],
                     output_pins: Sequence[str], detailed_layers: Sequence[str],
                     cover_layers: Sequence[str],  cell_type: str, **kwargs: Any) -> bool:
        """Generate the LEF file.

        Parameters
        ----------
        impl_lib : str
            the implementation library name.
        impl_cell : str
            the implementation cell name.
        verilog_path: Path
            the verilog shell file.
        lef_path : Path
            the output file path.
        run_path: Path
            the run directory.
        pwr_pins : Sequence[str]
            list of power pin names.
        gnd_pins : Sequence[str]
            list of ground pin names.
        clk_pins : Sequence[str]
            list of clock pin names.
        analog_pins : Sequence[str]
            list of analog pin names.
        output_pins : Sequence[str]
            list of output pin names.
        detailed_layers : Sequence[str]
            list of detailed layer names.
        cover_layers : Sequence[str]
            list of cover layer names.
        cell_type : str
            the cell type.
        **kwargs: Any
            Tool-specific configuration parameters.

        Returns
        -------
        success : bool
            True if LEF generation succeeded.
        """
        pass
