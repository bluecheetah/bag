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

"""This module defines various data classes used in layout.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Mapping

from dataclasses import dataclass

from pybag.enum import Orient2D, Direction2D

from ..util.immutable import Param


class TemplateEdgeInfo:
    def __init__(self, west: Param, south: Param, east: Param, north: Param):
        self._info = {
            Direction2D.WEST: west,
            Direction2D.EAST: east,
            Direction2D.SOUTH: south,
            Direction2D.NORTH: north,
        }

    def get_edge_params(self, direction: Direction2D) -> Param:
        return self._info[direction]

    def to_tuple(self) -> Tuple[Param, Param, Param, Param]:
        return (self._info[Direction2D.WEST], self._info[Direction2D.SOUTH],
                self._info[Direction2D.EAST], self._info[Direction2D.NORTH])


class MOMCapInfo:
    """Class providing convenient MOM cap information lookup"""

    def __init__(self, cap_info: Mapping[str, Any],
                 port_widths: Mapping[int, int], port_pleft: Mapping[int, bool]):
        self._bot_dir = Orient2D[cap_info['bot_dir']]
        self._cap_info: Dict[int, Tuple[int, int, int, int, int]] = cap_info['info']
        self._bot_layer = min(self._cap_info.keys())
        self._port_widths = port_widths
        self._port_pleft = port_pleft

    def get_direction(self, layer: int) -> Orient2D:
        diff = layer - self._bot_layer
        return self._bot_dir if diff & 1 == 0 else self._bot_dir.perpendicular()

    def get_port_tr_w(self, layer: int) -> int:
        port_tr_w0 = self._cap_info[layer][4]
        port_tr_w1 = self._port_widths.get(layer, 1)
        return max(port_tr_w0, port_tr_w1)

    def get_port_plow(self, layer: int) -> bool:
        return self._port_pleft.get(layer, False)

    def get_cap_specs(self, layer: int) -> Tuple[int, int, int, int]:
        return self._cap_info[layer][:-1]


@dataclass(frozen=True)
class MaxSpaceFillInfo:
    info: Tuple[int, int, int, int, float]

    def get_space(self, orient: Orient2D) -> int:
        return self.info[orient.value]

    def get_margin(self, orient: Orient2D) -> int:
        return self.info[orient.value + 2]
