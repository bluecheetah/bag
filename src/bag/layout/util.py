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

from typing import Mapping, Any, Tuple, Optional, Type, cast, Sequence

from pybag.enum import Direction2D
from pybag.core import BBox, Transform

from ..util.immutable import Param
from ..util.importlib import import_class
from ..design.module import Module


from .template import TemplateBase, TemplateDB


class BlackBoxTemplate(TemplateBase):
    """A black box template."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            lib_name='The library name.',
            cell_name='The layout cell name.',
            top_layer='The top level layer.',
            size='The width/height of the cell, in resolution units.',
            ports='The port information dictionary.',
        )

    def get_layout_basename(self) -> str:
        cell_name: str = self.params['cell_name']
        return f'BlackBox_{cell_name}'

    def draw_layout(self) -> None:
        lib_name: str = self.params['lib_name']
        cell_name: str = self.params['cell_name']
        top_layer: int = self.params['top_layer']
        size: Tuple[int, int] = self.params['size']
        ports: Mapping[str, Mapping[str, Sequence[Tuple[int, int, int, int]]]] = self.params['ports']

        show_pins = self.show_pins
        for term_name, pin_dict in ports.items():
            for lay, bbox_list in pin_dict.items():
                for xl, yb, xr, yt in bbox_list:
                    box = BBox(xl, yb, xr, yt)
                    self._register_pin(lay, term_name, box, show_pins)

        self.add_instance_primitive(lib_name, cell_name)

        self.prim_top_layer = top_layer
        self.prim_bound_box = BBox(0, 0, size[0], size[1])

        for layer in range(1, top_layer + 1):
            try:
                self.mark_bbox_used(layer, self.prim_bound_box)
            except ValueError:
                pass

        self.sch_params = dict(
            lib_name=lib_name,
            cell_name=cell_name,
        )

    def _register_pin(self, lay: str, term_name: str, box: BBox, show_pins: bool) -> None:
        # TODO: find way to add WireArray if possible
        self.add_pin_primitive(term_name, lay, box, show=show_pins)


class IPMarginTemplate(TemplateBase):
    """A wrapper template the packages a TemplateBase into an IP block.

    This class adds the necessary margins so a TemplateBase can be packaged into an IP
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

        self._core: Optional[TemplateBase] = None

    @property
    def core(self) -> TemplateBase:
        return self._core

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        return self._core.get_schematic_class_inst()

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
        )

    def draw_layout(self) -> None:
        cls_name: str = self.params['cls_name']
        params: Param = self.params['params']

        gen_cls = cast(Type[TemplateBase], import_class(cls_name))
        master: TemplateBase = self.new_template(gen_cls, params=params)
        self._core = master
        top_layer = master.top_layer

        dx0 = master.get_margin(top_layer, Direction2D.WEST)
        dy0 = master.get_margin(top_layer, Direction2D.SOUTH)
        dx1 = master.get_margin(top_layer, Direction2D.EAST)
        dy1 = master.get_margin(top_layer, Direction2D.NORTH)
        inst = self.add_instance(master, inst_name='XINST', xform=Transform(dx0, dy0))

        inst_box = inst.bound_box
        self.set_size_from_bound_box(top_layer, BBox(0, 0, inst_box.xh + dx1, inst_box.yh + dy1))

        # re-export pins
        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name))

        # pass out schematic parameters
        self.sch_params = master.sch_params
