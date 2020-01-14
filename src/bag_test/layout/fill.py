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

"""This script demonstrates how to add substrate contact in transistor row."""

from typing import Any, Dict

from pybag.enum import RoundMode
from pybag.core import BBox

from bag.util.immutable import Param
from bag.layout.template import TemplateDB, TemplateBase


class FillEdgeTest(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='width.',
            h='height.',
            fill_layer='fill layer ID.',
        )

    def draw_layout(self):
        w: int = self.params['w']
        h: int = self.params['h']
        fill_layer: int = self.params['fill_layer']

        grid = self.grid
        tech_info = grid.tech_info
        fill_info = tech_info.get_max_space_fill_info(fill_layer)

        self.set_size_from_bound_box(fill_layer, BBox(0, 0, w, h), round_up=True)
        bbox = self.bound_box

        tdir = grid.get_direction(fill_layer)
        pdir = tdir.perpendicular()
        margin = fill_info.get_margin(pdir)
        margin_le = fill_info.get_margin(tdir)
        sp_le = fill_info.get_space(tdir)
        dim = bbox.get_dim(pdir)
        dim_le = bbox.get_dim(tdir)

        tidxl = grid.coord_to_track(fill_layer, margin, mode=RoundMode.LESS_EQ)
        tidxr = grid.coord_to_track(fill_layer, dim - margin, mode=RoundMode.GREATER_EQ)
        wlen = grid.get_min_cont_length(fill_layer, 1)

        # fill inner
        self.add_wires(fill_layer, tidxl + 1, margin_le, dim_le - margin_le, num=tidxr - tidxl - 1)

        lower = margin_le
        self.add_wires(fill_layer, tidxl, lower, lower + wlen)
        lower += wlen + sp_le
        self.add_wires(fill_layer, tidxl, lower, lower + wlen)
        lower += wlen + sp_le + 2
        self.add_wires(fill_layer, tidxl, lower, lower + wlen)
        lower += wlen + 2 * sp_le + wlen * 2
        self.add_wires(fill_layer, tidxl, lower, lower + wlen)

        self.do_max_space_fill(fill_layer, bbox)


class FillEdgeCenterTest(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            fill_layer='fill layer ID.',
        )

    def draw_layout(self):
        fill_layer: int = self.params['fill_layer']

        grid = self.grid
        tech_info = grid.tech_info
        fill_info = tech_info.get_max_space_fill_info(fill_layer)

        tdir = grid.get_direction(fill_layer)
        pdir = tdir.perpendicular()
        margin = fill_info.get_margin(pdir)
        margin_le = fill_info.get_margin(tdir)
        sp_le = fill_info.get_space(tdir)

        blk_arr = grid.get_block_size(fill_layer, half_blk_x=False, half_blk_y=False)
        dim_q = blk_arr[pdir.value]

        w = (int(round(margin * 1.5)) // dim_q) * dim_q
        wlen = grid.get_min_cont_length(fill_layer, 1)
        h = 2 * margin_le + 7 * wlen + 5 * sp_le

        self.set_size_from_bound_box(fill_layer, BBox(0, 0, w, h), round_up=True,
                                     half_blk_x=False, half_blk_y=False)
        bbox = self.bound_box

        dim = bbox.get_dim(pdir)

        tidx0 = grid.find_next_track(fill_layer, dim - margin, mode=RoundMode.LESS)
        tidx1 = grid.find_next_track(fill_layer, margin, mode=RoundMode.GREATER)
        tidx_l = tidx0 + 1
        tidx_r = tidx1 - 1

        lower = margin_le
        self.add_wires(fill_layer, tidx0, lower, lower + wlen)
        self.add_wires(fill_layer, tidx1, lower, lower + wlen)
        lower += wlen + sp_le
        self.add_wires(fill_layer, tidx1, lower, lower + wlen)
        lower += wlen + sp_le
        self.add_wires(fill_layer, tidx0, lower, lower + wlen)
        lower += 2 * wlen + 2 * sp_le
        self.add_wires(fill_layer, tidx_l, lower, lower + wlen)
        lower += 2 * wlen + sp_le
        self.add_wires(fill_layer, tidx_r, lower, lower + wlen)

        self.do_max_space_fill(fill_layer, bbox)


class FillEndTest(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='width.',
            h='height.',
            fill_layer='fill layer ID.',
        )

    def draw_layout(self):
        w: int = self.params['w']
        h: int = self.params['h']
        fill_layer: int = self.params['fill_layer']

        grid = self.grid
        tech_info = grid.tech_info
        fill_info = tech_info.get_max_space_fill_info(fill_layer)

        self.set_size_from_bound_box(fill_layer, BBox(0, 0, w, h), round_up=True)
        bbox = self.bound_box

        tdir = grid.get_direction(fill_layer)
        pdir = tdir.perpendicular()
        margin = fill_info.get_margin(pdir)
        margin_le = fill_info.get_margin(tdir)
        sp_le = fill_info.get_space(tdir)
        dim = bbox.get_dim(pdir)
        dim_le = bbox.get_dim(tdir)

        tidxl = grid.coord_to_track(fill_layer, margin, mode=RoundMode.LESS_EQ)
        tidxr = grid.coord_to_track(fill_layer, dim - margin, mode=RoundMode.GREATER_EQ)
        tidx_end = grid.coord_to_track(fill_layer, dim, mode=RoundMode.LESS)

        wlen = grid.get_min_cont_length(fill_layer, 1)

        # fill inner and transverse edges
        gap = (margin_le + wlen) // 2 + sp_le
        self.add_wires(fill_layer, tidxl, gap, dim_le - gap, num=tidxr - tidxl + 1)
        self.add_wires(fill_layer, 0, margin_le, dim_le - margin_le)
        self.add_wires(fill_layer, tidx_end, margin_le, dim_le - margin_le)

        self.do_max_space_fill(fill_layer, bbox)


class FillCenterTest(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='width.',
            h='height.',
            fill_layer='fill layer ID.',
        )

    def draw_layout(self):
        w: int = self.params['w']
        h: int = self.params['h']
        fill_layer: int = self.params['fill_layer']

        grid = self.grid
        tech_info = grid.tech_info
        fill_info = tech_info.get_max_space_fill_info(fill_layer)

        self.set_size_from_bound_box(fill_layer, BBox(0, 0, w, h), round_up=True)
        bbox = self.bound_box

        tdir = grid.get_direction(fill_layer)
        pdir = tdir.perpendicular()
        margin = fill_info.get_margin(pdir)
        margin_le = fill_info.get_margin(tdir)
        dim = bbox.get_dim(pdir)
        dim_le = bbox.get_dim(tdir)

        wlen = grid.get_min_cont_length(fill_layer, 1)

        # fill edges and ends
        tidxl = grid.coord_to_track(fill_layer, margin, mode=RoundMode.LESS_EQ, even=True)
        tidxr = grid.coord_to_track(fill_layer, dim - margin, mode=RoundMode.GREATER_EQ, even=True)

        tcoord_u = dim_le - margin_le
        num = tidxr - tidxl - 1
        self.add_wires(fill_layer, tidxl, margin_le, tcoord_u, num=2, pitch=tidxr - tidxl)
        self.add_wires(fill_layer, tidxl + 1, margin_le, margin_le + wlen, num=num, pitch=1)
        self.add_wires(fill_layer, tidxl + 1, tcoord_u - wlen, tcoord_u, num=num, pitch=1)

        self.do_max_space_fill(fill_layer, bbox)


class FillCenterTest2(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='width.',
            h='height.',
            fill_layer='fill layer ID.',
        )

    def draw_layout(self):
        w: int = self.params['w']
        h: int = self.params['h']
        fill_layer: int = self.params['fill_layer']

        grid = self.grid
        tech_info = grid.tech_info
        fill_info = tech_info.get_max_space_fill_info(fill_layer)

        self.set_size_from_bound_box(fill_layer, BBox(0, 0, w, h), round_up=True)
        bbox = self.bound_box

        tdir = grid.get_direction(fill_layer)
        pdir = tdir.perpendicular()
        margin = fill_info.get_margin(pdir)
        margin_le = fill_info.get_margin(tdir)
        sp_le = fill_info.get_space(tdir)
        dim = bbox.get_dim(pdir)
        dim_le = bbox.get_dim(tdir)

        tidxl = grid.coord_to_track(fill_layer, margin, mode=RoundMode.LESS_EQ)
        tidxr = grid.coord_to_track(fill_layer, dim - margin, mode=RoundMode.GREATER_EQ)
        tidx_end = grid.coord_to_track(fill_layer, dim, mode=RoundMode.LESS)

        wlen = grid.get_min_cont_length(fill_layer, 1)

        # fill inner and transverse edges
        gap = margin_le + wlen + sp_le
        with open('debug.txt', 'w') as f:
            print(margin_le, wlen, sp_le, dim_le, file=f)
        self.add_wires(fill_layer, tidxl, gap, dim_le - gap, num=tidxr - tidxl + 1)
        self.add_wires(fill_layer, 0, margin_le, dim_le - margin_le)
        self.add_wires(fill_layer, tidx_end, margin_le, dim_le - margin_le)

        self.do_max_space_fill(fill_layer, bbox)
