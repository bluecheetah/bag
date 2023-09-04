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

"""This module defines layout template classes.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Union, Dict, Any, List, TypeVar, Type, Optional, Tuple, Iterable, Mapping,
    Sequence, Set, cast
)
from bag.typing import PointType
from bag.util.search import BinaryIterator
from bag.env import create_routing_grid_from_file

import abc
from itertools import product
import os
from pathlib import Path

from pybag.enum import (
    PathStyle, BlockageType, BoundaryType, DesignOutput, Orient2D, SupplyWrapMode,
    Orientation, Direction, MinLenMode, RoundMode, PinMode, Direction2D, LogLevel
)
from pybag.core import (
    BBox, BBoxArray, PyLayCellView, Transform, PyLayInstRef, PyPath, PyBlockage, PyBoundary,
    PyRect, PyVia, PyPolygon, PyPolygon90, PyPolygon45, ViaParam, COORD_MIN, COORD_MAX,
    RTree, BBoxCollection, TrackColoring, make_tr_colors
)

from ..util.immutable import ImmutableSortedDict, Param
from ..util.cache import DesignMaster, MasterDB, format_cell_name
from ..util.interval import IntervalSet
from ..util.math import HalfInt
from ..design.module import Module

from .core import PyLayInstance
from .tech import TechInfo
from .routing.base import Port, TrackID, WireArray, TrackManager
from .routing.grid import RoutingGrid
from .data import MOMCapInfo, TemplateEdgeInfo

GeoType = Union[PyRect, PyPolygon90, PyPolygon45, PyPolygon]
TemplateType = TypeVar('TemplateType', bound='TemplateBase')
DiffWarrType = Tuple[Optional[WireArray], Optional[WireArray]]

if TYPE_CHECKING:
    from ..core import BagProject
    from ..typing import TrackType, SizeType


class TemplateDB(MasterDB):
    """A database of all templates.

    This class is a subclass of MasterDB that defines some extra properties/function
    aliases to make creating layouts easier.

    Parameters
    ----------
    routing_grid : RoutingGrid
        the default RoutingGrid object.
    lib_name : str
        the cadence library to put all generated templates in.
    log_file: str
        the log file path.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated layout name prefix.
    name_suffix : str
        generated layout name suffix.
    log_level : LogLevel
        the logging level.
    """

    def __init__(self, routing_grid: RoutingGrid, lib_name: str, log_file: str, prj: Optional[BagProject] = None,
                 name_prefix: str = '', name_suffix: str = '', log_level: LogLevel = LogLevel.DEBUG) -> None:
        MasterDB.__init__(self, lib_name, log_file, prj=prj, name_prefix=name_prefix, name_suffix=name_suffix,
                          log_level=log_level)

        self._grid = routing_grid
        self._tr_colors = make_tr_colors(self._grid.tech_info)

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: The global RoutingGrid instance."""
        return self._grid

    @property
    def tech_info(self) -> TechInfo:
        return self._grid.tech_info

    @property
    def tr_colors(self) -> TrackColoring:
        return self._tr_colors

    def new_template(self, temp_cls: Type[TemplateType], params: Optional[Mapping[str, Any]] = None,
                     **kwargs: Any) -> TemplateType:
        """Alias for new_master() for backwards compatibility.
        """
        return self.new_master(temp_cls, params=params, **kwargs)

    def instantiate_layout(self, template: TemplateBase, top_cell_name: str = '',
                           output: DesignOutput = DesignOutput.LAYOUT, **kwargs: Any) -> None:
        """Alias for instantiate_master(), with default output type of LAYOUT.
        """
        self.instantiate_master(output, template, top_cell_name, **kwargs)

    def batch_layout(self, info_list: Sequence[Tuple[TemplateBase, str]],
                     output: DesignOutput = DesignOutput.LAYOUT, **kwargs: Any) -> None:
        """Alias for batch_output(), with default output type of LAYOUT.
        """
        self.batch_output(output, info_list, **kwargs)


def get_cap_via_extensions(info: MOMCapInfo, grid: RoutingGrid, bot_layer: int,
                           top_layer: int) -> Dict[int, int]:
    via_ext_dict: Dict[int, int] = {lay: 0 for lay in range(bot_layer, top_layer + 1)}
    # get via extensions on each layer
    for lay0 in range(bot_layer, top_layer):
        lay1 = lay0 + 1

        # port-to-port via extension
        bot_tr_w = info.get_port_tr_w(lay0)
        top_tr_w = info.get_port_tr_w(lay1)
        ext_pp = grid.get_via_extensions(Direction.LOWER, lay0, bot_tr_w, top_tr_w)

        w0, sp0, _, _ = info.get_cap_specs(lay0)
        w1, sp1, _, _ = info.get_cap_specs(lay1)
        # cap-to-cap via extension
        ext_cc = grid.get_via_extensions_dim(Direction.LOWER, lay0, w0, w1)
        # cap-to-port via extension
        ext_cp = grid.get_via_extensions_dim_tr(Direction.LOWER, lay0, w0, top_tr_w)
        # port-to-cap via extension
        ext_pc = grid.get_via_extensions_dim_tr(Direction.UPPER, lay1, w1, bot_tr_w)

        via_ext_dict[lay0] = max(via_ext_dict[lay0], ext_pp[0], ext_cc[0], ext_cp[0], ext_pc[0])
        via_ext_dict[lay1] = max(via_ext_dict[lay1], ext_pp[1], ext_cc[1], ext_cp[1], ext_pc[1])

    return via_ext_dict


class TemplateBase(DesignMaster):
    """The base template class.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    params : Param
        the parameter values.
    log_file: str
        the log file path.
    log_level : LogLevel
        the logging level.
    **kwargs : Any
        dictionary of the following optional parameters:

        grid : RoutingGrid
            the routing grid to use for this template.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, log_file: str,
                 log_level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> None:

        # add hidden parameters
        DesignMaster.__init__(self, temp_db, params, log_file, log_level, **kwargs)

        # private attributes
        self._size: Optional[SizeType] = None
        self._ports: Dict[str, Port] = {}
        self._port_params: Dict[str, Dict[str, Any]] = {}
        self._array_box: Optional[BBox] = None
        self._fill_box: Optional[BBox] = None
        self._sch_params: Optional[Param] = None
        self._cell_boundary_added: bool = False
        self._instances: Dict[str, PyLayInstance] = {}
        self._use_color: bool = False
        self._blackbox_gds: List[Path] = []

        # public attributes
        self.prim_top_layer: Optional[int] = None
        self.prim_bound_box: Optional[BBox] = None

        # get private attributes from parameters
        tmp_grid: Union[RoutingGrid, str] = self.params['grid']
        if tmp_grid is None:
            self._grid: RoutingGrid = temp_db.grid
        else:
            if isinstance(tmp_grid, RoutingGrid):
                self._grid: RoutingGrid = tmp_grid
            else:
                self._grid: RoutingGrid = create_routing_grid_from_file(tmp_grid)

        tmp_colors: TrackColoring = self.params['tr_colors']
        if tmp_colors is None:
            self._tr_colors: TrackColoring = temp_db.tr_colors
        else:
            self._tr_colors: TrackColoring = tmp_colors

        self._show_pins: bool = self.params['show_pins']
        self._edge_info: Optional[TemplateEdgeInfo] = None

        # create Cython wrapper object
        self._layout: PyLayCellView = PyLayCellView(self._grid, self._tr_colors, self.cell_name)

    @property
    def blackbox_gds(self) -> List[Path]:
        return self._blackbox_gds

    @classmethod
    def get_hidden_params(cls) -> Dict[str, Any]:
        ans = DesignMaster.get_hidden_params()
        ans['grid'] = None
        ans['tr_colors'] = None
        ans['show_pins'] = True
        return ans

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return None

    @abc.abstractmethod
    def draw_layout(self) -> None:
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        pass

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        return self.get_schematic_class()

    def get_master_basename(self) -> str:
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self.get_layout_basename()

    def get_layout_basename(self) -> str:
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        return self.__class__.__name__

    def get_content(self, output_type: DesignOutput, rename_dict: Dict[str, str], name_prefix: str,
                    name_suffix: str, shell: bool, exact_cell_names: Set[str],
                    supply_wrap_mode: SupplyWrapMode) -> Tuple[str, Any]:
        if not self.finalized:
            raise ValueError('This template is not finalized yet')

        cell_name = format_cell_name(self.cell_name, rename_dict, name_prefix, name_suffix,
                                     exact_cell_names, supply_wrap_mode)
        return cell_name, self._layout

    def finalize(self) -> None:
        """Finalize this master instance.
        """
        # create layout
        self.draw_layout()

        # finalize this template
        grid = self.grid
        grid.tech_info.finalize_template(self)

        # construct port objects
        for net_name, port_params in self._port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            hide = port_params['hide']
            if port_params['show']:
                label = port_params['label']
                for lay, geo_list in pin_dict.items():
                    if isinstance(lay, int):
                        for warr in geo_list:
                            self._layout.add_pin_arr(net_name, label, warr.track_id,
                                                     warr.lower, warr.upper)
                    else:
                        for box in geo_list:
                            self._layout.add_pin(lay, net_name, label, box)
            self._ports[net_name] = Port(net_name, pin_dict, label, hide)

        # call super finalize routine
        DesignMaster.finalize(self)

    @property
    def show_pins(self) -> bool:
        """bool: True to show pins."""
        return self._show_pins

    @property
    def sch_params(self) -> Optional[Param]:
        """Optional[Dict[str, Any]]: The schematic parameters dictionary."""
        return self._sch_params

    @sch_params.setter
    def sch_params(self, new_params: Dict[str, Any]) -> None:
        self._sch_params = ImmutableSortedDict(new_params)

    @property
    def template_db(self) -> TemplateDB:
        """TemplateDB: The template database object"""
        # noinspection PyTypeChecker
        return self.master_db

    @property
    def is_empty(self) -> bool:
        """bool: True if this template is empty."""
        return self._layout.is_empty

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: The RoutingGrid object"""
        return self._grid

    @grid.setter
    def grid(self, new_grid: RoutingGrid) -> None:
        self._layout.set_grid(new_grid)
        self._grid = new_grid

    @property
    def tr_colors(self) -> TrackColoring:
        return self._tr_colors

    @property
    def array_box(self) -> Optional[BBox]:
        """Optional[BBox]: The array/abutment bounding box of this template."""
        return self._array_box

    @array_box.setter
    def array_box(self, new_array_box: BBox) -> None:
        if not self._finalized:
            self._array_box = new_array_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def fill_box(self) -> Optional[BBox]:
        """Optional[BBox]: The dummy fill bounding box of this template."""
        return self._fill_box

    @fill_box.setter
    def fill_box(self, new_box: BBox) -> None:
        if not self._finalized:
            self._fill_box = new_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def top_layer(self) -> int:
        """int: The top layer ID used in this template."""
        if self.size is None:
            if self.prim_top_layer is None:
                raise Exception('Both size and prim_top_layer are unset.')
            return self.prim_top_layer
        return self.size[0]

    @property
    def size(self) -> Optional[SizeType]:
        """Optional[SizeType]: The size of this template, in (layer, nx_blk, ny_blk) format."""
        return self._size

    @property
    def size_defined(self) -> bool:
        """bool: True if size or bounding box has been set."""
        return self.size is not None or self.prim_bound_box is not None

    @property
    def bound_box(self) -> Optional[BBox]:
        """Optional[BBox]: Returns the template BBox.  None if size not set yet."""
        mysize = self.size
        if mysize is None:
            if self.prim_bound_box is None:
                raise ValueError('Both size and prim_bound_box are unset.')
            return self.prim_bound_box

        wblk, hblk = self.grid.get_size_dimension(mysize)
        return BBox(0, 0, wblk, hblk)

    @size.setter
    def size(self, new_size: SizeType) -> None:
        if not self._finalized:
            self._size = new_size
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def layout_cellview(self) -> PyLayCellView:
        """PyLayCellView: The internal layout object."""
        return self._layout

    @property
    def edge_info(self) -> Optional[TemplateEdgeInfo]:
        return self._edge_info

    @property
    def use_color(self) -> bool:
        return self._use_color

    @edge_info.setter
    def edge_info(self, new_info: TemplateEdgeInfo) -> None:
        self._edge_info = new_info

    def get_margin(self, top_layer: int, edge_dir: Direction2D,
                   half_blk_x: bool = True, half_blk_y: bool = True) -> int:
        grid = self.grid
        tech_info = grid.tech_info

        edge_info = self.edge_info
        if edge_info is None:
            # TODO: implement this.  Need to recurse down instance hierarchy
            raise ValueError('Not implemented yet.  See developer.')

        my_edge = self.edge_info.get_edge_params(edge_dir)
        is_vertical = edge_dir.is_vertical
        margin = tech_info.get_margin(is_vertical, my_edge, None)

        blk_size = grid.get_block_size(top_layer, half_blk_x=half_blk_x, half_blk_y=half_blk_y)
        q = blk_size[is_vertical]
        return -(-margin // q) * q

    def get_rect_bbox(self, lay_purp: Tuple[str, str]) -> BBox:
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.

        Returns
        -------
        box : BBox
            the overall bounding box of the given layer.
        """
        return self._layout.get_rect_bbox(lay_purp[0], lay_purp[1])

    def new_template_with(self, **kwargs: Any) -> TemplateBase:
        """Create a new template with the given parameters.

        This method will update the parameter values with the given dictionary,
        then create a new template with those parameters and return it.

        Parameters
        ----------
        **kwargs : Any
            a dictionary of new parameter values.

        Returns
        -------
        new_temp : TemplateBase
            A new layout master object.
        """
        # get new parameter dictionary.
        new_params = self.params.copy(append=kwargs)
        return self.template_db.new_template(self.__class__, params=new_params)

    def set_size_from_bound_box(self, top_layer_id: int, bbox: BBox, *, round_up: bool = False,
                                half_blk_x: bool = True, half_blk_y: bool = True):
        """Compute the size from overall bounding box.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        bbox : BBox
            the overall bounding box
        round_up: bool
            True to round up bounding box if not quantized properly
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.
        """
        grid = self.grid

        if bbox.xl != 0 or bbox.yl != 0:
            raise ValueError('lower-left corner of overall bounding box must be (0, 0).')

        if grid.size_defined(top_layer_id):
            self.size = grid.get_size_tuple(top_layer_id, bbox.w, bbox.h, round_up=round_up,
                                            half_blk_x=half_blk_x, half_blk_y=half_blk_y)
        else:
            self.prim_top_layer = top_layer_id
            self.prim_bound_box = bbox

    def set_size_from_array_box(self, top_layer_id: int) -> None:
        """Automatically compute the size from array_box.

        Assumes the array box is exactly in the center of the template.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        """
        grid = self.grid

        array_box = self.array_box
        if array_box is None:
            raise ValueError("array_box is not set")

        dx = array_box.xl
        dy = array_box.yl
        if dx < 0 or dy < 0:
            raise ValueError('lower-left corner of array box must be in first quadrant.')

        # noinspection PyAttributeOutsideInit
        self.size = grid.get_size_tuple(top_layer_id, 2 * dx + self.array_box.w,
                                        2 * dy + self.array_box.h)

    def get_pin_name(self, name: str) -> str:
        """Get the actual name of the given pin from the renaming dictionary.

        Given a pin name, If this Template has a parameter called 'rename_dict',
        return the actual pin name from the renaming dictionary.

        Parameters
        ----------
        name : str
            the pin name.

        Returns
        -------
        actual_name : str
            the renamed pin name.
        """
        rename_dict = self.params.get('rename_dict', {})
        return rename_dict.get(name, name)

    def get_port(self, name: str = '') -> Port:
        """Returns the port object with the given name.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this template has only one port,
            then return it.

        Returns
        -------
        port : Port
            the port object.
        """
        if not name:
            if len(self._ports) != 1:
                raise ValueError('Template has %d ports != 1.' % len(self._ports))
            name = next(iter(self._ports))
        return self._ports[name]

    def has_port(self, port_name: str) -> bool:
        """Returns True if this template has the given port."""
        return port_name in self._ports

    def port_names_iter(self) -> Iterable[str]:
        """Iterates over port names in this template.

        Yields
        ------
        port_name : str
            name of a port in this template.
        """
        return self._ports.keys()

    def new_template(self, temp_cls: Type[TemplateType], *,
                     params: Optional[Mapping[str, Any]] = None,
                     show_pins: bool = False,
                     grid: Optional[RoutingGrid] = None) -> TemplateType:
        """Create a new template.

        Parameters
        ----------
        temp_cls : Type[TemplateType]
            the template class to instantiate.
        params : Optional[Mapping[str, Any]]
            the parameter dictionary.
        show_pins : bool
            True to pass show_pins in the generated template, if params does not already have
            show_pins.
        grid: Optional[RoutingGrid]
            routing grid for this cell.

        Returns
        -------
        template : TemplateType
            the new template instance.
        """
        if grid is None:
            grid = params.get('grid', self.grid)
        show_pins = params.get('show_pins', show_pins)
        if isinstance(params, ImmutableSortedDict):
            params = params.copy(append=dict(grid=grid, show_pins=show_pins))
        else:
            params['grid'] = grid
            params['show_pins'] = show_pins
        return self.template_db.new_template(params=params, temp_cls=temp_cls)

    def add_instance(self,
                     master: TemplateBase,
                     *,
                     inst_name: str = '',
                     xform: Optional[Transform] = None,
                     nx: int = 1,
                     ny: int = 1,
                     spx: int = 0,
                     spy: int = 0,
                     commit: bool = True,
                     ) -> PyLayInstance:
        """Adds a new (arrayed) instance to layout.

        Parameters
        ----------
        master : TemplateBase
            the master template object.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        xform : Optional[Transform]
            the transformation object.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        inst : PyLayInstance
            the added instance.
        """
        if xform is None:
            xform = Transform()

        ref = self._layout.add_instance(master.layout_cellview, inst_name, xform, nx, ny,
                                        spx, spy, False)
        ans = PyLayInstance(self, master, ref)
        if commit:
            ans.commit()

        self._instances[ans.name] = ans
        self._use_color = self._use_color or master.use_color
        return ans

    def add_instance_primitive(self,
                               lib_name: str,
                               cell_name: str,
                               *,
                               xform: Optional[Transform] = None,
                               view_name: str = 'layout',
                               inst_name: str = '',
                               nx: int = 1,
                               ny: int = 1,
                               spx: int = 0,
                               spy: int = 0,
                               params: Optional[Dict[str, Any]] = None,
                               commit: bool = True,
                               **kwargs: Any,
                               ) -> PyLayInstRef:
        """Adds a new (arrayed) primitive instance to layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        xform : Optional[Transform]
            the transformation object.
        view_name : str
            instance view name.  Defaults to 'layout'.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.  Used for adding pcell instance.
        commit : bool
            True to commit the object immediately.
        **kwargs : Any
            additional arguments.  Usually implementation specific.

        Returns
        -------
        ref : PyLayInstRef
            A reference to the primitive instance.
        """
        if not params:
            params = kwargs
        else:
            params.update(kwargs)
        if xform is None:
            xform = Transform()

        # TODO: support pcells
        if params:
            raise ValueError("layout pcells not supported yet; see developer")

        blackbox_gds: Path = Path(os.environ['BAG_TECH_CONFIG_DIR']) / 'blackbox_gds' / lib_name / f'{cell_name}.gds'
        if blackbox_gds not in self._blackbox_gds:
            self._blackbox_gds.append(blackbox_gds)
        return self._layout.add_prim_instance(lib_name, cell_name, view_name, inst_name, xform,
                                              nx, ny, spx, spy, commit)

    def is_horizontal(self, layer: str) -> bool:
        """Returns True if the given layer has no direction or is horizontal."""
        lay_id = self._grid.tech_info.get_layer_id(layer)
        return (lay_id is None) or self._grid.is_horizontal(lay_id)

    def add_rect(self, lay_purp: Tuple[str, str], bbox: BBox, commit: bool = True) -> PyRect:
        """Add a new rectangle.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        bbox : BBox
            the rectangle bounding box.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        rect : PyRect
            the added rectangle.
        """
        return self._layout.add_rect(lay_purp[0], lay_purp[1], bbox, commit)

    def add_rect_array(self, lay_purp: Tuple[str, str], bbox: BBox,
                       nx: int = 1, ny: int = 1, spx: int = 0, spy: int = 0) -> None:
        """Add a new rectangle array.
        """
        self._layout.add_rect_arr(lay_purp[0], lay_purp[1], bbox, nx, ny, spx, spy)

    def add_bbox_array(self, lay_purp: Tuple[str, str], barr: BBoxArray) -> None:
        """Add a new rectangle array.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        barr : BBoxArray
            the rectangle bounding box array.
        """
        self._layout.add_rect_arr(lay_purp[0], lay_purp[1], barr)

    def add_bbox_collection(self, lay_purp: Tuple[str, str], bcol: BBoxCollection) -> None:
        self._layout.add_rect_list(lay_purp[0], lay_purp[1], bcol)

    def has_res_metal(self) -> bool:
        return self._grid.tech_info.has_res_metal()

    def add_res_metal(self, layer_id: int, bbox: BBox) -> None:
        """Add a new metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.
        bbox : BBox
            the resistor bounding box.
        """
        if self.has_res_metal():
            for lay, purp in self._grid.tech_info.get_res_metal_layers(layer_id):
                self._layout.add_rect(lay, purp, bbox, True)
        else:
            raise ValueError('res_metal does not exist in the process.')

    def add_path(self, lay_purp: Tuple[str, str], width: int, points: List[PointType],
                 start_style: PathStyle, *, join_style: PathStyle = PathStyle.round,
                 stop_style: Optional[PathStyle] = None, commit: bool = True) -> PyPath:
        """Add a new path.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        width : int
            the path width.
        points : List[PointType]
            points defining this path.
        start_style : PathStyle
            the path beginning style.
        join_style : PathStyle
            path style for the joints.
        stop_style : Optional[PathStyle]
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        if stop_style is None:
            stop_style = start_style
        half_width = width // 2
        return self._layout.add_path(lay_purp[0], lay_purp[1], points, half_width, start_style,
                                     stop_style, join_style, commit)

    def add_path45_bus(self, lay_purp: Tuple[str, str], points: List[PointType], widths: List[int],
                       spaces: List[int], start_style: PathStyle, *,
                       join_style: PathStyle = PathStyle.round,
                       stop_style: Optional[PathStyle] = None, commit: bool = True) -> PyPath:
        """Add a path bus that only contains 45 degree turns.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        points : List[PointType]
            points defining this path.
        widths : List[int]
            width of each path in the bus.
        spaces : List[int]
            space between each path.
        start_style : PathStyle
            the path beginning style.
        join_style : PathStyle
            path style for the joints.
        stop_style : Optional[PathStyle]
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        if stop_style is None:
            stop_style = start_style
        return self._layout.add_path45_bus(lay_purp[0], lay_purp[1], points, widths, spaces,
                                           start_style, stop_style, join_style, commit)

    def add_polygon(self, lay_purp: Tuple[str, str], points: List[PointType],
                    commit: bool = True) -> PyPolygon:
        """Add a new polygon.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        points : List[PointType]
            vertices of the polygon.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        polygon : PyPolygon
            the added polygon object.
        """
        return self._layout.add_poly(lay_purp[0], lay_purp[1], points, commit)

    def add_blockage(self, layer: str, blk_type: BlockageType, points: List[PointType],
                     commit: bool = True) -> PyBlockage:
        """Add a new blockage object.

        Parameters
        ----------
        layer : str
            the layer name.
        blk_type : BlockageType
            the blockage type.
        points : List[PointType]
            vertices of the blockage object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        blockage : PyBlockage
            the added blockage object.
        """
        return self._layout.add_blockage(layer, blk_type, points, commit)

    def add_boundary(self, bnd_type: BoundaryType, points: List[PointType],
                     commit: bool = True) -> PyBoundary:
        """Add a new boundary.

        Parameters
        ----------
        bnd_type : str
            the boundary type.
        points : List[PointType]
            vertices of the boundary object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        boundary : PyBoundary
            the added boundary object.
        """
        return self._layout.add_boundary(bnd_type, points, commit)

    def add_cell_boundary(self, bbox: BBox) -> None:
        """Adds cell boundary in this template.

        By default, this method is called when finalizing a template (although the process
        implementation may override this behavior) to set the cell boundary, which is generally
        used for DRC or P&R purposes.

        This method can only be called once from the template.  All calls after the first one will
        be ignored.  Therefore, if you need to set the cell boundary to be something other than
        the template's bounding box, you can call this in the draw_layout() method.

        Parameters
        ----------
        bbox : BBox
            the cell boundary bounding box.
        """
        if not self._cell_boundary_added:
            self._cell_boundary_added = True
            self.grid.tech_info.add_cell_boundary(self, bbox)

    def disable_cell_boundary(self) -> None:
        """Disable cell boundary drawing in this template."""
        self._cell_boundary_added = True

    def reexport(self, port: Port, *,
                 net_name: str = '', label: str = '', show: Optional[bool] = None,
                 hide: Optional[bool] = None, connect: bool = False) -> None:
        """Re-export the given port object.

        Add all geometries in the given port as pins with optional new name
        and label.

        Parameters
        ----------
        port : Port
            the Port object to re-export.
        net_name : str
            the new net name.  If not given, use the port's current net name.
        label : str
            the label.  If not given, use net_name.
        show : Optional[bool]
            True to draw the pin in layout.  If None, use self.show_pins
        hide: Optional[bool]
            if given, it overrides the hide flag of the port, otherwise the default is used.
        connect : bool
            True to enable connection by name.
        """
        if show is None:
            show = self._show_pins

        net_name = net_name or port.net_name
        if not label:
            if net_name != port.net_name:
                if port.label.endswith(':'):
                    # inherit connect setting of the port
                    label = net_name + ':'
                else:
                    label = net_name
            else:
                label = port.label
        if connect and label[-1] != ':':
            label += ':'

        if hide is None:
            hide = port.hidden

        show = show and not hide
        if net_name not in self._port_params:
            self._port_params[net_name] = dict(label=label, pins={}, show=show, hide=hide)

        port_params = self._port_params[net_name]
        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')
        if port_params['hide'] != hide:
            raise ValueError('Conflicting hide port specification.')

        # export all port geometries
        port_pins = port_params['pins']
        for lay, geo_list in port.items():
            cur_geo_list = port_pins.get(lay, None)
            if cur_geo_list is None:
                port_pins[lay] = cur_geo_list = []
            cur_geo_list.extend(geo_list)

    def add_pin_primitive(self, net_name: str, layer: str, bbox: BBox, *,
                          label: str = '', show: Optional[bool] = None, hide: bool = False,
                          connect: bool = False):
        """Add a primitive pin to the layout.

        Parameters
        ----------
        net_name : str
            the net name associated with the pin.
        layer : str
            the pin layer name.
        bbox : BBox
            the pin bounding box.
        label : str
            the label of this pin.  If None or empty, defaults to be the net_name.
            this argument is used if you need the label to be different than net name
            for LVS purposes.  For example, unconnected pins usually need a colon after
            the name to indicate that LVS should short those pins together.
        show : Optional[bool]
            True to draw the pin in layout. If None, use self.show_pins
        hide : bool
            True to add a hidden pin.
        connect : bool
            True to enable connection by name.
        """
        if show is None:
            show = self._show_pins
        show = show and not hide

        label = label or net_name
        if connect and label[-1] != ':':
            label += ':'

        port_params = self._port_params.get(net_name, None)
        if port_params is None:
            self._port_params[net_name] = port_params = dict(label=label, pins={},
                                                             show=show, hide=hide)
        else:
            # check labels is consistent.
            if port_params['label'] != label:
                msg = 'Current port label = %s != specified label = %s'
                raise ValueError(msg % (port_params['label'], label))
            if port_params['show'] != show:
                raise ValueError('Conflicting show port specification.')
            if port_params['hide'] != hide:
                raise ValueError('Conflicting hide port specification.')

        port_pins = port_params['pins']

        if layer in port_pins:
            port_pins[layer].append(bbox)
        else:
            port_pins[layer] = [bbox]

    def add_label(self, label: str, lay_purp: Tuple[str, str], bbox: BBox) -> None:
        """Adds a label to the layout.

        This is mainly used to add voltage text labels.

        Parameters
        ----------
        label : str
            the label text.
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        bbox : BBox
            the label bounding box.
        """
        w = bbox.w
        text_h = bbox.h
        if text_h > w:
            orient = Orientation.R90
            text_h = w
        else:
            orient = Orientation.R0
        xform = Transform(bbox.xm, bbox.ym, orient)
        self._layout.add_label(lay_purp[0], lay_purp[1], xform, label, text_h)

    def add_pin(self, net_name: str, wire_arr_list: Union[WireArray, List[WireArray]],
                *, label: str = '', show: Optional[bool] = None,
                mode: PinMode = PinMode.ALL, hide: bool = False, connect: bool = False) -> None:
        """Add new pin to the layout.

        If one or more pins with the same net name already exists,
        they'll be grouped under the same port.

        Parameters
        ----------
        net_name : str
            the net name associated with the pin.
        wire_arr_list : Union[WireArray, List[WireArray]]
            WireArrays representing the pin geometry.
        label : str
            the label of this pin.  If None or empty, defaults to be the net_name.
            this argument is used if you need the label to be different than net name
            for LVS purposes.  For example, unconnected pins usually need a colon after
            the name to indicate that LVS should short those pins together.
        show : Optional[bool]
            if True, draw the pin in layout.  If None, use self.show_pins
        mode : PinMode
            location of the pin relative to the WireArray.
        hide : bool
            True if this is a hidden pin.
        connect : bool
            True to enable connection by name.
        """
        if show is None:
            show = self._show_pins
        show = show and not hide

        label = label or net_name
        if connect and label[-1] != ':':
            label += ':'

        port_params = self._port_params.get(net_name, None)
        if port_params is None:
            self._port_params[net_name] = port_params = dict(label=label, pins={},
                                                             show=show, hide=hide)
        else:
            # check labels is consistent.
            if port_params['label'] != label:
                msg = 'Current port label = %s != specified label = %s'
                raise ValueError(msg % (port_params['label'], label))
            if port_params['show'] != show:
                raise ValueError('Conflicting show port specification.')
            if port_params['hide'] != hide:
                raise ValueError('Conflicting hide port specification.')

        grid = self._grid
        for warr in WireArray.wire_grp_iter(wire_arr_list):
            # add pin array to port_pins
            tid = warr.track_id.copy_with(grid)
            layer_id = tid.layer_id
            if mode is not PinMode.ALL:
                # create new pin WireArray that's snapped to the edge
                cur_w = grid.get_wire_total_width(layer_id, tid.width)
                wl = warr.lower
                wu = warr.upper
                pin_len = min(grid.get_next_length(layer_id, tid.width, cur_w, even=True),
                              wu - wl)
                if mode is PinMode.LOWER:
                    wu = wl + pin_len
                elif mode is PinMode.UPPER:
                    wl = wu - pin_len
                else:
                    wl = (wl + wu - pin_len) // 2
                    wu = wl + pin_len
                warr = WireArray(tid, wl, wu)

            port_pins = port_params['pins']
            if layer_id not in port_pins:
                port_pins[layer_id] = [warr]
            else:
                port_pins[layer_id].append(warr)

        self._use_color = True

    def add_via(self, bbox: BBox, bot_lay_purp: Tuple[str, str], top_lay_purp: Tuple[str, str],
                bot_dir: Orient2D, *, extend: bool = True, top_dir: Optional[Orient2D] = None,
                add_layers: bool = False, commit: bool = True) -> PyVia:
        """Adds an arrayed via object to the layout.

        Parameters
        ----------
        bbox : BBox
            the via bounding box, not including extensions.
        bot_lay_purp : Tuple[str. str]
            the bottom layer/purpose pair.
        top_lay_purp : Tuple[str, str]
            the top layer/purpose pair.
        bot_dir : Orient2D
            the bottom layer extension direction.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[Orient2D]
            top layer extension direction.  Defaults to be perpendicular to bottom layer direction.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.
        commit : bool
            True to commit via immediately.

        Returns
        -------
        via : PyVia
            the new via object.
        """
        tech_info = self._grid.tech_info
        via_info = tech_info.get_via_info(bbox, Direction.LOWER, bot_lay_purp[0],
                                          top_lay_purp[0],
                                          bot_dir, purpose=bot_lay_purp[1],
                                          adj_purpose=top_lay_purp[1],
                                          extend=extend, adj_ex_dir=top_dir)

        if via_info is None:
            raise ValueError('Cannot create via between layers {} and {} '
                             'with BBox: {}'.format(bot_lay_purp, top_lay_purp, bbox))

        table = via_info['params']
        via_id = table['id']
        xform = table['xform']
        via_param = table['via_param']

        return self._layout.add_via(xform, via_id, via_param, add_layers, commit)

    def add_via_arr(self, barr: BBoxArray, bot_lay_purp: Tuple[str, str],
                    top_lay_purp: Tuple[str, str], bot_dir: Orient2D, *, extend: bool = True,
                    top_dir: Optional[Orient2D] = None, add_layers: bool = False) -> Dict[str, Any]:
        """Adds an arrayed via object to the layout.

        Parameters
        ----------
        barr : BBoxArray
            the BBoxArray representing the via bounding boxes, not including extensions.
        bot_lay_purp : Tuple[str. str]
            the bottom layer/purpose pair.
        top_lay_purp : Tuple[str, str]
            the top layer/purpose pair.
        bot_dir : Orient2D
            the bottom layer extension direction.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[Orient2D]
            top layer extension direction.  Defaults to be perpendicular to bottom layer direction.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.

        Returns
        -------
        via_info : Dict[str, Any]
            the via information dictionary.
        """
        tech_info = self._grid.tech_info
        base_box = barr.base
        via_info = tech_info.get_via_info(base_box, Direction.LOWER, bot_lay_purp[0],
                                          top_lay_purp[0], bot_dir, purpose=bot_lay_purp[1],
                                          adj_purpose=top_lay_purp[1], extend=extend,
                                          adj_ex_dir=top_dir)

        if via_info is None:
            raise ValueError('Cannot create via between layers {} and {} '
                             'with BBox: {}'.format(bot_lay_purp, top_lay_purp, base_box))

        table = via_info['params']
        via_id = table['id']
        xform = table['xform']
        via_param = table['via_param']

        self._layout.add_via_arr(xform, via_id, via_param, add_layers, barr.nx, barr.ny,
                                 barr.spx, barr.spy)

        return via_info

    def add_via_primitive(self, via_type: str, xform: Transform, cut_width: int, cut_height: int,
                          *, num_rows: int = 1, num_cols: int = 1, sp_rows: int = 0,
                          sp_cols: int = 0, enc1: Tuple[int, int, int, int] = (0, 0, 0, 0),
                          enc2: Tuple[int, int, int, int] = (0, 0, 0, 0), nx: int = 1, ny: int = 1,
                          spx: int = 0, spy: int = 0, priority: int = 1) -> None:
        """Adds via(s) by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        xform: Transform
            the transformation object.
        cut_width : CoordType
            via cut width.  This is used to create rectangle via.
        cut_height : CoordType
            via cut height.  This is used to create rectangle via.
        num_rows : int
            number of via cut rows.
        num_cols : int
            number of via cut columns.
        sp_rows : CoordType
            spacing between via cut rows.
        sp_cols : CoordType
            spacing between via cut columns.
        enc1 : Optional[List[CoordType]]
            a list of left, right, top, and bottom enclosure values on bottom layer.
            Defaults to all 0.
        enc2 : Optional[List[CoordType]]
            a list of left, right, top, and bottom enclosure values on top layer.
            Defaults to all 0.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : int
            column pitch.
        spy : int
            row pitch.
        priority : int
            via priority, defaults to 1
        """
        l1, r1, t1, b1 = enc1
        l2, r2, t2, b2 = enc2
        param = ViaParam(num_cols, num_rows, cut_width, cut_height, sp_cols, sp_rows,
                         l1, r1, t1, b1, l2, r2, t2, b2, priority)
        self._layout.add_via_arr(xform, via_type, param, True, nx, ny, spx, spy)

    def add_via_on_grid(self, tid1: TrackID, tid2: TrackID, *, extend: bool = True
                        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Add a via on the routing grid.

        Parameters
        ----------
        tid1 : TrackID
            the first TrackID
        tid2 : TrackID
            the second TrackID
        extend : bool
            True to extend outside the via bounding box.
        """
        return self._layout.add_via_on_intersections(tid1, tid2, COORD_MIN, COORD_MAX,
                                                     COORD_MIN, COORD_MAX, extend, False)

    def extend_wires(self, warr_list: Union[WireArray, List[Optional[WireArray]]], *,
                     lower: Optional[int] = None, upper: Optional[int] = None,
                     min_len_mode: Optional[int] = None) -> List[Optional[WireArray]]:
        """Extend the given wires to the given coordinates.

        Parameters
        ----------
        warr_list : Union[WireArray, List[Optional[WireArray]]]
            the wires to extend.
        lower : Optional[int]
            the wire lower coordinate.
        upper : Optional[int]
            the wire upper coordinate.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.

        Returns
        -------
        warr_list : List[Optional[WireArray]]
            list of added wire arrays.
            If any elements in warr_list were None, they will be None in the return.
        """
        grid = self._grid

        new_warr_list = []
        for warr in WireArray.wire_grp_iter(warr_list):
            if warr is None:
                new_warr_list.append(None)
            else:
                tid = warr.track_id.copy_with(grid)
                wlower = warr.lower
                wupper = warr.upper
                if lower is None:
                    cur_lower = wlower
                else:
                    cur_lower = min(lower, wlower)
                if upper is None:
                    cur_upper = wupper
                else:
                    cur_upper = max(upper, wupper)
                if min_len_mode is not None:
                    # extend track to meet minimum length
                    # make sure minimum length is even so that middle coordinate exists
                    tr_len = cur_upper - cur_lower
                    next_len = grid.get_next_length(tid.layer_id, tid.width, tr_len, even=True)
                    if next_len > tr_len:
                        ext = next_len - tr_len
                        if min_len_mode < 0:
                            cur_lower -= ext
                        elif min_len_mode > 0:
                            cur_upper += ext
                        else:
                            cur_lower -= ext // 2
                            cur_upper = cur_lower + next_len

                new_warr = WireArray(tid, cur_lower, cur_upper)
                self._layout.add_warr(tid, cur_lower, cur_upper)
                new_warr_list.append(new_warr)

        self._use_color = True
        return new_warr_list

    def add_wires(self, layer_id: int, track_idx: TrackType, lower: int, upper: int, *,
                  width: int = 1, num: int = 1, pitch: TrackType = 1) -> WireArray:
        """Add the given wire(s) to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        lower : CoordType
            the wire lower coordinate.
        upper : CoordType
            the wire upper coordinate.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : TrackType
            the wire pitch.

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch, grid=self._grid)
        warr = WireArray(tid, lower, upper)
        self._layout.add_warr(tid, lower, upper)
        self._use_color = True
        return warr

    def add_matched_wire(self, warr: WireArray, coord: int, layer_id: int) -> WireArray:
        """Adds a wire (without any via), matched to a provided wire array.
        The mirroring takes place with respect to a coordinate and the track direction on the
        layer of the coordinate

        Parameters
        ----------
        warr : WireArray
            the original wire array for which a matched wire should be drawn
        coord : int
            the coordinate which is used for mirroring
        layer_id : int
            the layer_id of the coordinate. this argument is used to figure out the axis around
            which things should be mirrored

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        grid = self._grid
        tid = warr.track_id
        wire_layer = warr.layer_id
        wire_dir = grid.get_direction(wire_layer)
        ref_dir = grid.get_direction(layer_id)

        if wire_dir is not ref_dir:
            # if wire and reference have different orientation
            new_lower = 2 * coord - warr.upper
            new_upper = 2 * coord - warr.lower
            self.add_wires(wire_layer, tid.base_index, new_lower, new_upper)
            return WireArray(tid, new_lower, new_upper)

        coord_tidx = grid.coord_to_track(layer_id, coord)
        new_tidx = 2 * coord_tidx - tid.base_index
        new_tid = TrackID(wire_layer, new_tidx, width=tid.width, num=tid.num,
                          pitch=tid.pitch, grid=grid)
        self.add_wires(wire_layer, new_tidx, warr.lower, warr.upper)
        return WireArray(new_tid, warr.lower, warr.upper)

    def connect_to_tracks_with_dummy_wires(self,
                                           wire_arr_list: Union[WireArray, List[WireArray]],
                                           track_id: TrackID,
                                           ref_coord: int,
                                           ref_layer_id: int,
                                           *,
                                           wire_lower: Optional[int] = None,
                                           wire_upper: Optional[int] = None,
                                           track_lower: Optional[int] = None,
                                           track_upper: Optional[int] = None,
                                           min_len_mode: MinLenMode = None,
                                           ret_wire_list: Optional[List[WireArray]] = None,
                                           debug: bool = False) -> Optional[WireArray]:
        """Implements connect_to_tracks but with matched wires drawn simultaneously
         Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        ref_coord: int
            the coordinate which is used for mirroring
        ref_layer_id: int
            the layer_id of the coordinate. this argument is used to figure out the axis around
            which things should be mirrored
        wire_lower : Optional[CoordType]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[CoordType]
            if given, extend wire(s) to this upper coordinate.
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        ret_wire_list : Optional[List[WireArray]]
            If not none, extended wires that are created will be appended to this list.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Optional[WireArray]
            WireArray representing the tracks created.
        """
        if ret_wire_list is None:
            ret_wire_list = []
        track_warr = self.connect_to_tracks(wire_arr_list,
                                            track_id,
                                            wire_lower=wire_lower,
                                            wire_upper=wire_upper,
                                            track_lower=track_lower,
                                            track_upper=track_upper,
                                            min_len_mode=min_len_mode,
                                            ret_wire_list=ret_wire_list,
                                            debug=debug)
        self.add_matched_wire(track_warr, ref_coord, ref_layer_id)
        for warr in ret_wire_list:
            self.add_matched_wire(warr, ref_coord, ref_layer_id)

        return track_warr

    def connect_wire_to_coord(self, wire: WireArray, layer_id: int, coord: int,
                              min_len_mode: MinLenMode = MinLenMode.NONE,
                              round_mode: RoundMode = RoundMode.NONE) -> WireArray:
        """ Connects a given wire to a wire on the next/previous layer aligned with a given
        coordinate.

        Parameters
        ----------
        wire : WireArray
            wire object to be connected.
        layer_id: int
            the wire layer ID.
        coord : CoordType
            the coordinate to be used for alignment.
        min_len_mode : MinLenMode
            the minimum length extension mode used in connect_to_tracks.
        round_mode: RoundMode
            the rounding mode used in coord_to_track conversion.

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        if layer_id not in [wire.layer_id + 1, wire.layer_id - 1]:
            raise ValueError(f'cannot connect wire of layer {wire.layer_id} to layer {layer_id}')
        tidx = self.grid.coord_to_track(layer_id, coord, round_mode)
        tid = TrackID(layer_id, tidx, width=wire.track_id.width)
        warr = self.connect_to_tracks(wire, tid, min_len_mode=min_len_mode)
        return warr

    def add_res_metal_warr(self, layer_id: int, track_idx: TrackType, lower: int, upper: int,
                           **kwargs: Any) -> Param:
        """Add metal resistor as WireArray to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        lower : CoordType
            the wire lower coordinate.
        upper : CoordType
            the wire upper coordinate.
        **kwargs :
            optional arguments to add_wires()

        Returns
        -------
        sch_params : Dict[str, Any]
            the metal resistor schematic parameters dictionary.
        """
        warr = self.add_wires(layer_id, track_idx, lower, upper, **kwargs)

        wdir = self.grid.get_direction(layer_id)
        npar = 0
        w = 0
        for _, _, box in warr.wire_iter(self._tr_colors):
            self.add_res_metal(layer_id, box)
            npar += 1
            if w == 0:
                w = box.get_dim(wdir.perpendicular().value)

        ans = dict(
            w=w,
            l=upper - lower,
            layer=layer_id,
            npar=npar,
        )
        return ImmutableSortedDict(ans)

    def add_mom_cap(self, cap_box: BBox, bot_layer: int, num_layer: int, *,
                    port_widths: Optional[Mapping[int, int]] = None,
                    port_plow: Optional[Mapping[int, bool]] = None,
                    array: bool = False,
                    cap_wires_list: Optional[List[Tuple[Tuple[str, str], Tuple[str, str],
                                                        BBoxArray, BBoxArray]]] = None,
                    cap_type: str = 'standard'
                    ) -> Dict[int, Tuple[List[WireArray], List[WireArray]]]:
        """Draw mom cap in the defined bounding box."""

        empty_dict = {}
        if num_layer <= 1:
            raise ValueError('Must have at least 2 layers for MOM cap.')
        if port_widths is None:
            port_widths = empty_dict
        if port_plow is None:
            port_plow = empty_dict

        grid = self.grid
        tech_info = grid.tech_info

        top_layer = bot_layer + num_layer - 1
        cap_info = MOMCapInfo(tech_info.tech_params['mom_cap'][cap_type], port_widths, port_plow)
        via_ext_dict = get_cap_via_extensions(cap_info, grid, bot_layer, top_layer)

        # find port locations and cap boundaries.
        port_tracks: Dict[int, Tuple[List[int], List[int]]] = {}
        cap_bounds: Dict[int, Tuple[int, int]] = {}
        cap_exts: Dict[int, Tuple[int, int]] = {}
        for cur_layer in range(bot_layer, top_layer + 1):
            cap_w, cap_sp, cap_margin, num_ports = cap_info.get_cap_specs(cur_layer)
            port_tr_w = cap_info.get_port_tr_w(cur_layer)
            port_tr_sep = grid.get_sep_tracks(cur_layer, port_tr_w, port_tr_w)

            dir_idx = grid.get_direction(cur_layer).value
            coord0, coord1 = cap_box.get_interval(1 - dir_idx)
            # get max via extension on adjacent layers
            adj_via_ext = 0
            if cur_layer != bot_layer:
                adj_via_ext = via_ext_dict[cur_layer - 1]
            if cur_layer != top_layer:
                adj_via_ext = max(adj_via_ext, via_ext_dict[cur_layer + 1])
            # find track indices
            if array:
                tidx0 = grid.coord_to_track(cur_layer, coord0)
                tidx1 = grid.coord_to_track(cur_layer, coord1)
            else:
                tidx0 = grid.find_next_track(cur_layer, coord0 + adj_via_ext, tr_width=port_tr_w,
                                             mode=RoundMode.GREATER_EQ)
                tidx1 = grid.find_next_track(cur_layer, coord1 - adj_via_ext, tr_width=port_tr_w,
                                             mode=RoundMode.LESS_EQ)

            if tidx0 + 2 * num_ports * port_tr_sep >= tidx1:
                raise ValueError('Cannot draw MOM cap; '
                                 f'not enough space between ports on layer {cur_layer}.')

            # compute space from MOM cap wires to port wires
            cap_margin = max(cap_margin, grid.get_min_space(cur_layer, port_tr_w))
            lower_tracks = [tidx0 + idx * port_tr_sep for idx in range(num_ports)]
            upper_tracks = [tidx1 - idx * port_tr_sep for idx in range(num_ports - 1, -1, -1)]

            tr_ll = grid.get_wire_bounds(cur_layer, lower_tracks[0], width=port_tr_w)[0]
            tr_lu = grid.get_wire_bounds(cur_layer, lower_tracks[num_ports - 1], width=port_tr_w)[1]
            tr_ul = grid.get_wire_bounds(cur_layer, upper_tracks[0], width=port_tr_w)[0]
            tr_uu = grid.get_wire_bounds(cur_layer, upper_tracks[num_ports - 1], width=port_tr_w)[1]
            port_tracks[cur_layer] = (lower_tracks, upper_tracks)
            cap_bounds[cur_layer] = (tr_lu + cap_margin, tr_ul - cap_margin)
            cap_exts[cur_layer] = (tr_ll, tr_uu)

        port_dict: Dict[int, Tuple[List[WireArray], List[WireArray]]] = {}
        cap_wire_dict: Dict[int, Tuple[Tuple[str, str], Tuple[str, str], BBoxArray, BBoxArray]] = {}
        # draw ports/wires
        for cur_layer in range(bot_layer, top_layer + 1):
            port_plow = cap_info.get_port_plow(cur_layer)
            port_tr_w = cap_info.get_port_tr_w(cur_layer)
            cap_w, cap_sp, cap_margin, num_ports = cap_info.get_cap_specs(cur_layer)

            # find port/cap wires lower/upper coordinates
            lower = COORD_MAX
            upper = COORD_MIN
            if cur_layer != top_layer:
                lower, upper = cap_exts[cur_layer + 1]
            if cur_layer != bot_layer:
                tmpl, tmpu = cap_exts[cur_layer - 1]
                lower = min(lower, tmpl)
                upper = max(upper, tmpu)

            via_ext = via_ext_dict[cur_layer]
            lower -= via_ext
            upper += via_ext

            # draw ports
            lower_tracks, upper_tracks = port_tracks[cur_layer]
            lower_warrs = [self.add_wires(cur_layer, tr_idx, lower, upper, width=port_tr_w)
                           for tr_idx in lower_tracks]
            upper_warrs = [self.add_wires(cur_layer, tr_idx, lower, upper, width=port_tr_w)
                           for tr_idx in upper_tracks]

            # assign port wires to positive/negative terminals
            num_ports = len(lower_warrs)
            if port_plow:
                if num_ports == 1:
                    plist = lower_warrs
                    nlist = upper_warrs
                else:
                    plist = [lower_warrs[0], upper_warrs[0]]
                    nlist = [lower_warrs[1], upper_warrs[1]]
            else:
                if num_ports == 1:
                    plist = upper_warrs
                    nlist = lower_warrs
                else:
                    plist = [lower_warrs[1], upper_warrs[1]]
                    nlist = [lower_warrs[0], upper_warrs[0]]

            # save ports
            port_dict[cur_layer] = plist, nlist

            # compute cap wires BBoxArray
            cap_bndl, cap_bndh = cap_bounds[cur_layer]
            cap_tot_space = cap_bndh - cap_bndl
            cap_pitch = cap_w + cap_sp
            num_cap_wires = cap_tot_space // cap_pitch
            cap_bndl += (cap_tot_space - (num_cap_wires * cap_pitch - cap_sp)) // 2

            cur_dir = grid.get_direction(cur_layer)
            cap_box0 = BBox(cur_dir, lower, upper, cap_bndl, cap_bndl + cap_w)
            lay_purp_list = tech_info.get_lay_purp_list(cur_layer)
            num_lay_purp = len(lay_purp_list)
            assert num_lay_purp <= 2, 'This method now only works for 1 or 2 colors.'
            num0 = (num_cap_wires + 1) // 2
            num1 = num_cap_wires - num0
            barr_pitch = cap_pitch * 2
            cap_box1 = cap_box0.get_move_by_orient(cur_dir, dt=0, dp=cap_pitch)
            barr0 = BBoxArray(cap_box0, cur_dir, np=num0, spp=barr_pitch)
            barr1 = BBoxArray(cap_box1, cur_dir, np=num1, spp=barr_pitch)
            if port_plow:
                capp_barr = barr1
                capn_barr = barr0
                capp_lp = lay_purp_list[-1]
                capn_lp = lay_purp_list[0]
            else:
                capp_barr = barr0
                capn_barr = barr1
                capp_lp = lay_purp_list[0]
                capn_lp = lay_purp_list[-1]

            # draw cap wires
            self.add_bbox_array(capp_lp, capp_barr)
            self.add_bbox_array(capn_lp, capn_barr)
            # save caps
            cap_barr_tuple = (capp_lp, capn_lp, capp_barr, capn_barr)
            cap_wire_dict[cur_layer] = cap_barr_tuple
            if cap_wires_list is not None:
                cap_wires_list.append(cap_barr_tuple)

            # connect port/cap wires to bottom port/cap
            if cur_layer != bot_layer:
                # connect ports to layer below
                bplist, bnlist = port_dict[cur_layer - 1]
                bcapp_lp, bcapn_lp, bcapp, bcapn = cap_wire_dict[cur_layer - 1]
                self._add_mom_cap_connect_ports(bplist, plist)
                self._add_mom_cap_connect_ports(bnlist, nlist)
                self._add_mom_cap_connect_cap_to_port(Direction.UPPER, capp_lp, capp_barr, bplist)
                self._add_mom_cap_connect_cap_to_port(Direction.UPPER, capn_lp, capn_barr, bnlist)
                self._add_mom_cap_connect_cap_to_port(Direction.LOWER, bcapp_lp, bcapp, plist)
                self._add_mom_cap_connect_cap_to_port(Direction.LOWER, bcapn_lp, bcapn, nlist)

        return port_dict

    def _add_mom_cap_connect_cap_to_port(self, cap_dir: Direction, cap_lp: Tuple[str, str],
                                         barr: BBoxArray, ports: List[WireArray]) -> None:
        num_ports = len(ports)
        if num_ports == 1:
            self.connect_bbox_to_tracks(cap_dir, cap_lp, barr, ports[0].track_id)
        else:
            port_dir = self.grid.get_direction(ports[0].layer_id)
            for idx, warr in enumerate(ports):
                new_barr = barr.get_sub_array(port_dir, num_ports, idx)
                self.connect_bbox_to_tracks(cap_dir, cap_lp, new_barr, warr.track_id)

    def _add_mom_cap_connect_ports(self, bot_ports: List[WireArray], top_ports: List[WireArray]
                                   ) -> None:
        for bot_warr, top_warr in product(bot_ports, top_ports):
            self.add_via_on_grid(bot_warr.track_id, top_warr.track_id, extend=True)

    def reserve_tracks(self, layer_id: int, track_idx: TrackType, *,
                       width: int = 1, num: int = 1, pitch: int = 0) -> None:
        """Reserve the given routing tracks so that power fill will not fill these tracks.

        Note: the size of this template should be set before calling this method.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : TrackType
            the wire pitch.
        """
        # TODO: fix this method
        raise ValueError('Not implemented yet.')

    def get_available_tracks(self, layer_id: int, tid_lo: TrackType, tid_hi: TrackType,
                             lower: int, upper: int, width: int = 1, sep: HalfInt = HalfInt(1),
                             include_last: bool = False, sep_margin: Optional[HalfInt] = None,
                             uniform_grid: bool = False) -> List[HalfInt]:
        """Returns a list of available tracks between the given bounds.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tid_lo : TrackType
            the lower track index, inclusive.
        tid_hi : TrackType
            the upper track index, exclusive by default.
        lower : int
            the lower wire coordinate.
        upper: int
            the upper wire coordinate.
        width : int
            the track width.
        sep : HalfInt
            the track separation
        include_last : bool
            True to make "upper" inclusive.
        sep_margin : Optional[HalfInt]
            the margin between available tracks and surrounding wires, in number of tracks.
        uniform_grid : bool
            True to get available tracks on a uniform grid; False to get densely packed available tracks

        Returns
        -------
        tidx_list : List[HalfInt]
            list of available tracks.
        """
        grid = self.grid

        orient = grid.get_direction(layer_id)
        tr_info = grid.get_track_info(layer_id)
        if sep_margin is None:
            sep_margin = grid.get_sep_tracks(layer_id, width, 1, same_color=False)
        bl, bu = grid.get_wire_bounds_htr(layer_id, 0, width)
        tr_w2 = (bu - bl) // 2
        margin = tr_info.pitch * sep_margin - (tr_info.width // 2) - tr_w2

        sp_list = [0, 0]
        sp_list[orient.value ^ 1] = margin
        spx, spy = sp_list

        htr0 = HalfInt.convert(tid_lo).dbl_value
        htr1 = HalfInt.convert(tid_hi).dbl_value
        if include_last:
            htr1 += 1
        htr_sep = HalfInt.convert(sep).dbl_value
        ans = []
        cur_htr = htr0
        while cur_htr < htr1:
            mid = grid.htr_to_coord(layer_id, cur_htr)
            box = BBox(orient, lower, upper, mid - tr_w2, mid + tr_w2)
            if not self._layout.get_intersect(layer_id, box, spx, spy, False):
                ans.append(HalfInt(cur_htr))
                cur_htr += htr_sep
            else:
                cur_htr += htr_sep if uniform_grid else 1

        return ans

    def connect_wires(self, wire_arr_list: Union[WireArray, List[WireArray]], *,
                      lower: Optional[int] = None,
                      upper: Optional[int] = None,
                      debug: bool = False,
                      ) -> List[WireArray]:
        """Connect all given WireArrays together.

        all WireArrays must be on the same layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArr, List[WireArr]]
            WireArrays to connect together.
        lower : Optional[CoordType]
            if given, extend connection wires to this lower coordinate.
        upper : Optional[CoordType]
            if given, extend connection wires to this upper coordinate.
        debug : bool
            True to print debug messages.

        Returns
        -------
        conn_list : List[WireArray]
            list of connection wires created.
        """
        grid = self._grid

        if lower is None:
            lower = COORD_MAX
        if upper is None:
            upper = COORD_MIN

        # record all wire ranges
        layer_id = None
        intv_set = IntervalSet()
        for wire_arr in WireArray.wire_grp_iter(wire_arr_list):
            # NOTE: no need to copy with new grid, this TrackID is not used to create WireArrays
            tid = wire_arr.track_id
            lay_id = tid.layer_id
            tr_w = tid.width
            if layer_id is None:
                layer_id = lay_id
            elif lay_id != layer_id:
                raise ValueError('WireArray layer ID != {}'.format(layer_id))

            cur_range = wire_arr.lower, wire_arr.upper
            for tidx in tid:
                intv = grid.get_wire_bounds(lay_id, tidx, width=tr_w)
                intv_rang_item = intv_set.get_first_overlap_item(intv)
                if intv_rang_item is None:
                    range_set = IntervalSet()
                    range_set.add(cur_range)
                    intv_set.add(intv, val=(range_set, tidx, tr_w))
                elif intv_rang_item[0] == intv:
                    tmp_rang_set: IntervalSet = intv_rang_item[1][0]
                    tmp_rang_set.add(cur_range, merge=True, abut=True)
                else:
                    raise ValueError(f'wire on lay={lay_id}, track={tidx} overlap existing wires. '
                                     f'wire interval={intv}, overlapped wire '
                                     f'interval={intv_rang_item[0]}')

        # draw wires, group into arrays
        new_warr_list = []
        base_start = None  # type: Optional[int]
        base_end = None  # type: Optional[int]
        base_tidx = None  # type: Optional[HalfInt]
        base_width = None  # type: Optional[int]
        count = 0
        pitch = 0
        last_tidx = 0
        for set_item in intv_set.items():
            intv = set_item[0]
            range_set: IntervalSet = set_item[1][0]
            cur_tidx: HalfInt = set_item[1][1]
            cur_tr_w: int = set_item[1][2]
            cur_start = min(lower, range_set.start)
            cur_end = max(upper, range_set.stop)

            if debug:
                print('wires intv: %s, range: (%d, %d)' % (intv, cur_start, cur_end))
            if count == 0:
                base_tidx = cur_tidx
                base_start = cur_start
                base_end = cur_end
                base_width = cur_tr_w
                count = 1
                pitch = 0
            else:
                assert base_tidx is not None, "count == 0 should have set base_intv"
                assert base_width is not None, "count == 0 should have set base_width"
                assert base_start is not None, "count == 0 should have set base_start"
                assert base_end is not None, "count == 0 should have set base_end"
                if cur_start == base_start and cur_end == base_end and base_width == cur_tr_w:
                    # length and width matches
                    cur_pitch = cur_tidx - last_tidx
                    if count == 1:
                        # second wire, set half pitch
                        pitch = cur_pitch
                        count += 1
                    elif pitch == cur_pitch:
                        # pitch matches
                        count += 1
                    else:
                        # pitch does not match, add current wires and start anew
                        track_id = TrackID(layer_id, base_tidx, width=base_width,
                                           num=count, pitch=pitch, grid=grid)
                        warr = WireArray(track_id, base_start, base_end)
                        new_warr_list.append(warr)
                        self._layout.add_warr(track_id, base_start, base_end)
                        base_tidx = cur_tidx
                        count = 1
                        pitch = 0
                else:
                    # length/width does not match, add cumulated wires and start anew
                    track_id = TrackID(layer_id, base_tidx, width=base_width,
                                       num=count, pitch=pitch, grid=grid)
                    warr = WireArray(track_id, base_start, base_end)
                    new_warr_list.append(warr)
                    self._layout.add_warr(track_id, base_start, base_end)
                    base_start = cur_start
                    base_end = cur_end
                    base_tidx = cur_tidx
                    base_width = cur_tr_w
                    count = 1
                    pitch = 0

            # update last lower coordinate
            last_tidx = cur_tidx

        if base_tidx is None:
            # no wires given at all
            return []

        assert base_tidx is not None, "count == 0 should have set base_intv"
        assert base_start is not None, "count == 0 should have set base_start"
        assert base_end is not None, "count == 0 should have set base_end"

        # add last wires
        track_id = TrackID(layer_id, base_tidx, base_width, num=count, pitch=pitch, grid=grid)
        warr = WireArray(track_id, base_start, base_end)
        self._layout.add_warr(track_id, base_start, base_end)
        new_warr_list.append(warr)
        self._use_color = True
        return new_warr_list

    def connect_bbox_to_tracks(self, layer_dir: Direction, lay_purp: Tuple[str, str],
                               box_arr: Union[BBox, BBoxArray], track_id: TrackID, *,
                               track_lower: Optional[int] = None,
                               track_upper: Optional[int] = None,
                               min_len_mode: MinLenMode = MinLenMode.NONE,
                               wire_lower: Optional[int] = None,
                               wire_upper: Optional[int] = None,
                               ret_bnds: Optional[List[int]] = None) -> WireArray:
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        layer_dir : Direction
            the primitive wire layer direction relative to the given tracks.  LOWER if
            the wires are below tracks, UPPER if the wires are above tracks.
        lay_purp : Tuple[str, str]
            the primitive wire layer/purpose name.
        box_arr : Union[BBox, BBoxArray]
            bounding box of the wire(s) to connect to tracks.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            The minimum length extension mode.
        wire_lower : Optional[int]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[int]
            if given, extend wire(s) to this upper coordinate.
        ret_bnds : Optional[List[int]]
            if given, return the bounds on the bounding box layer.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.
        """
        if isinstance(box_arr, BBox):
            box_arr = BBoxArray(box_arr)

        track_id = track_id.copy_with(self._grid)
        bnds = self._layout.connect_barr_to_tracks(layer_dir, lay_purp[0], lay_purp[1], box_arr,
                                                   track_id, track_lower, track_upper, min_len_mode,
                                                   wire_lower, wire_upper)
        tr_idx = 1 - layer_dir.value
        if ret_bnds is not None:
            ret_bnds[0] = bnds[layer_dir.value][0]
            ret_bnds[1] = bnds[layer_dir.value][1]

        self._use_color = True
        return WireArray(track_id, bnds[tr_idx][0], bnds[tr_idx][1])

    def connect_bbox_to_track_wires(self, layer_dir: Direction, lay_purp: Tuple[str, str],
                                    box_arr: Union[BBox, BBoxArray],
                                    track_wires: Union[WireArray, List[WireArray]], *,
                                    min_len_mode: MinLenMode = MinLenMode.NONE,
                                    ret_bnds: Optional[List[int]] = None
                                    ) -> Union[Optional[WireArray], List[Optional[WireArray]]]:
        ans = []
        bnds = [COORD_MAX, COORD_MIN]
        for warr in WireArray.wire_grp_iter(track_wires):
            cur_bnds = [0, 0]
            tr = self.connect_bbox_to_tracks(layer_dir, lay_purp, box_arr,
                                             warr.track_id, track_lower=warr.lower,
                                             track_upper=warr.upper, min_len_mode=min_len_mode,
                                             ret_bnds=cur_bnds)
            ans.append(tr)
            bnds[0] = min(bnds[0], cur_bnds[0])
            bnds[1] = max(bnds[1], cur_bnds[1])

        if ret_bnds is not None:
            ret_bnds[0] = bnds[0]
            ret_bnds[1] = bnds[1]

        if isinstance(track_wires, WireArray):
            return ans[0]
        return ans

    def connect_bbox_to_differential_tracks(self, p_lay_dir: Direction, n_lay_dir: Direction,
                                            p_lay_purp: Tuple[str, str],
                                            n_lay_purp: Tuple[str, str],
                                            pbox: Union[BBox, BBoxArray],
                                            nbox: Union[BBox, BBoxArray], tr_layer_id: int,
                                            ptr_idx: TrackType, ntr_idx: TrackType, *,
                                            width: int = 1, track_lower: Optional[int] = None,
                                            track_upper: Optional[int] = None,
                                            min_len_mode: MinLenMode = MinLenMode.NONE
                                            ) -> DiffWarrType:
        """Connect the given differential primitive wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        p_lay_dir : Direction
            positive signal layer direction.
        n_lay_dir : Direction
            negative signal layer direction.
        p_lay_purp : Tuple[str, str]
            positive signal layer/purpose pair.
        n_lay_purp : Tuple[str, str]
            negative signal layer/purpose pair.
        pbox : Union[BBox, BBoxArray]
            positive signal wires to connect.
        nbox : Union[BBox, BBoxArray]
            negative signal wires to connect.
        tr_layer_id : int
            track layer ID.
        ptr_idx : TrackType
            positive track index.
        ntr_idx : TrackType
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_bbox_to_matching_tracks([p_lay_dir, n_lay_dir],
                                                          [p_lay_purp, n_lay_purp], [pbox, nbox],
                                                          tr_layer_id, [ptr_idx, ntr_idx],
                                                          width=width, track_lower=track_lower,
                                                          track_upper=track_upper,
                                                          min_len_mode=min_len_mode)
        return track_list[0], track_list[1]

    def fix_track_min_length(self, tr_layer_id: int, width: int, track_lower: int, track_upper: int,
                             min_len_mode: MinLenMode) -> Tuple[int, int]:
        even = min_len_mode is MinLenMode.MIDDLE
        tr_len = self.grid.get_next_length(tr_layer_id, width, track_upper - track_lower, even=even)
        if min_len_mode is MinLenMode.LOWER:
            track_lower = track_upper - tr_len
        elif min_len_mode is MinLenMode.UPPER:
            track_upper = track_lower + tr_len
        elif min_len_mode is MinLenMode.MIDDLE:
            track_lower = (track_upper + track_lower - tr_len) // 2
            track_upper = track_lower + tr_len

        return track_lower, track_upper

    def connect_bbox_to_matching_tracks(self, lay_dir_list: List[Direction],
                                        lay_purp_list: List[Tuple[str, str]],
                                        box_arr_list: List[Union[BBox, BBoxArray]],
                                        tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                        width: int = 1, track_lower: Optional[int] = None,
                                        track_upper: Optional[int] = None,
                                        min_len_mode: MinLenMode = MinLenMode.NONE,
                                        ) -> List[Optional[WireArray]]:
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        lay_dir_list : List[Direction]
            the primitive wire layer direction list.
        lay_purp_list : List[Tuple[str, str]]
            the primitive wire layer/purpose list.
        box_arr_list : List[Union[BBox, BBoxArray]]
            bounding box of the wire(s) to connect to tracks.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        Returns
        -------
        wire_arr : List[Optional[WireArray]]
            WireArrays representing the tracks created.
        """
        grid = self._grid
        tr_dir = grid.get_direction(tr_layer_id)
        w_dir = tr_dir.perpendicular()

        num = len(lay_dir_list)
        if len(lay_purp_list) != num or len(box_arr_list) != num or len(tr_idx_list) != num:
            raise ValueError('Connection list parameters have mismatch length.')
        if num == 0:
            raise ValueError('Connection lists are empty.')

        wl = None
        wu = None
        for lay_dir, (lay, purp), box_arr, tr_idx in zip(lay_dir_list, lay_purp_list,
                                                         box_arr_list, tr_idx_list):
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)

            tid = TrackID(tr_layer_id, tr_idx, width=width, grid=self._grid)
            bnds = self._layout.connect_barr_to_tracks(lay_dir, lay, purp, box_arr, tid,
                                                       track_lower, track_upper, MinLenMode.NONE,
                                                       wl, wu)
            w_idx = lay_dir.value
            tr_idx = 1 - w_idx
            wl = bnds[w_idx][0]
            wu = bnds[w_idx][1]
            track_lower = bnds[tr_idx][0]
            track_upper = bnds[tr_idx][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for (lay, purp), box_arr, tr_idx in zip(lay_purp_list, box_arr_list, tr_idx_list):
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)
            else:
                box_arr = BBoxArray(box_arr.base, tr_dir, nt=box_arr.get_num(tr_dir),
                                    spt=box_arr.get_sp(tr_dir))

            box_arr.set_interval(w_dir, wl, wu)
            self._layout.add_rect_arr(lay, purp, box_arr)

            cur_tid = TrackID(tr_layer_id, tr_idx, width=width, grid=grid)
            warr = WireArray(cur_tid, track_lower, track_upper)
            self._layout.add_warr(cur_tid, track_lower, track_upper)
            ans.append(warr)

        self._use_color = True
        return ans

    def connect_to_tracks(self, wire_arr_list: Union[WireArray, List[WireArray]],
                          track_id: TrackID, *, wire_lower: Optional[int] = None,
                          wire_upper: Optional[int] = None, track_lower: Optional[int] = None,
                          track_upper: Optional[int] = None, min_len_mode: MinLenMode = None,
                          ret_wire_list: Optional[List[WireArray]] = None,
                          debug: bool = False) -> Optional[WireArray]:
        """Connect all given WireArrays to the given track(s).

        All given wires should be on adjacent layers of the track.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        wire_lower : Optional[CoordType]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[CoordType]
            if given, extend wire(s) to this upper coordinate.
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        ret_wire_list : Optional[List[WireArray]]
            If not none, extended wires that are created will be appended to this list.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Optional[WireArray]
            WireArray representing the tracks created.
        """
        if track_lower is None:
            track_lower = COORD_MAX
        if track_upper is None:
            track_upper = COORD_MIN

        # find min/max track Y coordinates
        track_id = track_id.copy_with(self._grid)
        tr_layer_id = track_id.layer_id
        tr_w = track_id.width

        # get top wire and bottom wire list
        warr_list_list = [[], []]
        for wire_arr in WireArray.wire_grp_iter(wire_arr_list):
            cur_layer_id = wire_arr.layer_id
            if cur_layer_id == tr_layer_id + 1:
                warr_list_list[1].append(wire_arr)
            elif cur_layer_id == tr_layer_id - 1:
                warr_list_list[0].append(wire_arr)
            else:
                raise ValueError(
                    'WireArray layer %d cannot connect to layer %d' % (cur_layer_id, tr_layer_id))

        if not warr_list_list[0] and not warr_list_list[1]:
            # no wires at all
            return None

        # connect wires together
        tmp = self._connect_to_tracks_helper(warr_list_list[0], track_id, wire_lower, wire_upper,
                                             track_lower, track_upper, ret_wire_list, 0, debug)
        track_lower, track_upper = tmp
        tmp = self._connect_to_tracks_helper(warr_list_list[1], track_id, wire_lower, wire_upper,
                                             track_lower, track_upper, ret_wire_list, 1, debug)
        track_lower, track_upper = tmp

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, tr_w, track_lower,
                                                             track_upper, min_len_mode)
        result = WireArray(track_id, track_lower, track_upper)
        self._layout.add_warr(track_id, track_lower, track_upper)
        self._use_color = True
        return result

    def _connect_to_tracks_helper(self, warr_list: List[WireArray], track_id: TrackID,
                                  wire_lower: Optional[int], wire_upper: Optional[int],
                                  track_lower: int, track_upper: int,
                                  ret_wire_list: Optional[List[WireArray]], idx: int,
                                  debug: bool) -> Tuple[Optional[int], Optional[int]]:
        # precondition: track_id has correct routing grid, but not WireArrays in warr_list
        for warr in self.connect_wires(warr_list, lower=wire_lower, upper=wire_upper,
                                       debug=debug):
            bnds = self._layout.connect_warr_to_tracks(warr.track_id, track_id,
                                                       warr.lower, warr.upper)
            if ret_wire_list is not None:
                new_tid = warr.track_id.copy_with(self._grid)
                ret_wire_list.append(WireArray(new_tid, bnds[idx][0], bnds[idx][1]))
            track_lower = min(track_lower, bnds[1 - idx][0])
            track_upper = max(track_upper, bnds[1 - idx][1])

        return track_lower, track_upper

    def connect_to_track_wires(self, wire_arr_list: Union[WireArray, List[WireArray]],
                               track_wires: Union[WireArray, List[WireArray]], *,
                               min_len_mode: Optional[MinLenMode] = None,
                               ret_wire_list: Optional[List[WireArray]] = None,
                               debug: bool = False) -> Union[Optional[WireArray],
                                                             List[Optional[WireArray]]]:
        """Connect all given WireArrays to the given WireArrays on adjacent layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_wires : Union[WireArray, List[WireArray]]
            list of tracks as WireArrays.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        ret_wire_list : Optional[List[WireArray]]
            If not none, extended wires that are created will be appended to this list.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Union[Optional[WireArray], List[Optional[WireArray]]]
            WireArrays representing the tracks created.  None if nothing to do.
        """
        ans = []  # type: List[Optional[WireArray]]
        for warr in WireArray.wire_grp_iter(track_wires):
            tr = self.connect_to_tracks(wire_arr_list, warr.track_id, track_lower=warr.lower,
                                        track_upper=warr.upper, min_len_mode=min_len_mode, ret_wire_list=ret_wire_list,
                                        debug=debug)
            ans.append(tr)

        if isinstance(track_wires, WireArray):
            return ans[0]
        return ans

    def connect_differential_tracks(self, pwarr_list: Union[WireArray, List[WireArray]],
                                    nwarr_list: Union[WireArray, List[WireArray]],
                                    tr_layer_id: int, ptr_idx: TrackType, ntr_idx: TrackType, *,
                                    width: int = 1, track_lower: Optional[int] = None,
                                    track_upper: Optional[int] = None
                                    ) -> Tuple[Optional[WireArray], Optional[WireArray]]:
        """Connect the given differential wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        pwarr_list : Union[WireArray, List[WireArray]]
            positive signal wires to connect.
        nwarr_list : Union[WireArray, List[WireArray]]
            negative signal wires to connect.
        tr_layer_id : int
            track layer ID.
        ptr_idx : TrackType
            positive track index.
        ntr_idx : TrackType
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_matching_tracks([pwarr_list, nwarr_list], tr_layer_id,
                                                  [ptr_idx, ntr_idx], width=width,
                                                  track_lower=track_lower, track_upper=track_upper)
        return track_list[0], track_list[1]

    def connect_differential_wires(self, pin_warrs: Union[WireArray, List[WireArray]],
                                   nin_warrs: Union[WireArray, List[WireArray]],
                                   pout_warr: WireArray, nout_warr: WireArray, *,
                                   track_lower: Optional[int] = None,
                                   track_upper: Optional[int] = None
                                   ) -> Tuple[Optional[WireArray], Optional[WireArray]]:
        """Connect the given differential wires to two WireArrays symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        pin_warrs : Union[WireArray, List[WireArray]]
            positive signal wires to connect.
        nin_warrs : Union[WireArray, List[WireArray]]
            negative signal wires to connect.
        pout_warr : WireArray
            positive track wires.
        nout_warr : WireArray
            negative track wires.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        p_tid = pout_warr.track_id
        lay_id = p_tid.layer_id
        pidx = p_tid.base_index
        nidx = nout_warr.track_id.base_index
        width = p_tid.width

        if track_lower is None:
            tr_lower = pout_warr.lower
        else:
            tr_lower = min(track_lower, pout_warr.lower)
        if track_upper is None:
            tr_upper = pout_warr.upper
        else:
            tr_upper = max(track_upper, pout_warr.upper)

        return self.connect_differential_tracks(pin_warrs, nin_warrs, lay_id, pidx, nidx,
                                                width=width, track_lower=tr_lower,
                                                track_upper=tr_upper)

    def connect_matching_tracks(self, warr_list_list: List[Union[WireArray, List[WireArray]]],
                                tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                width: int = 1,
                                track_lower: Optional[int] = None,
                                track_upper: Optional[int] = None,
                                min_len_mode: MinLenMode = MinLenMode.NONE
                                ) -> List[Optional[WireArray]]:
        """Connect wires to tracks with optimal matching.

        This method connects the wires to tracks in a way that minimizes the parasitic mismatches.

        Parameters
        ----------
        warr_list_list : List[Union[WireArray, List[WireArray]]]
            list of signal wires to connect.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.

        Returns
        -------
        track_list : List[WireArray]
            list of created tracks.
        """
        # simple error checking
        num_tracks = len(tr_idx_list)  # type: int
        if num_tracks != len(warr_list_list):
            raise ValueError('Connection list parameters have mismatch length.')
        if num_tracks == 0:
            raise ValueError('Connection lists are empty.')

        if track_lower is None:
            track_lower = COORD_MAX
        if track_upper is None:
            track_upper = COORD_MIN

        wbounds = [[COORD_MAX, COORD_MIN], [COORD_MAX, COORD_MIN]]
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            tid = TrackID(tr_layer_id, tr_idx, width=width, grid=self._grid)
            for warr in WireArray.wire_grp_iter(warr_list):
                cur_lay_id = warr.layer_id
                if cur_lay_id == tr_layer_id + 1:
                    wb_idx = 1
                elif cur_lay_id == tr_layer_id - 1:
                    wb_idx = 0
                else:
                    raise ValueError(
                        'WireArray layer {} cannot connect to layer {}'.format(cur_lay_id,
                                                                               tr_layer_id))

                bnds = self._layout.connect_warr_to_tracks(warr.track_id, tid,
                                                           warr.lower, warr.upper)
                wbounds[wb_idx][0] = min(wbounds[wb_idx][0], bnds[wb_idx][0])
                wbounds[wb_idx][1] = max(wbounds[wb_idx][1], bnds[wb_idx][1])
                track_lower = min(track_lower, bnds[1 - wb_idx][0])
                track_upper = max(track_upper, bnds[1 - wb_idx][1])

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            for warr in WireArray.wire_grp_iter(warr_list):
                wb_idx = (warr.layer_id - tr_layer_id + 1) // 2
                self._layout.add_warr(warr.track_id, wbounds[wb_idx][0], wbounds[wb_idx][1])

            cur_tid = TrackID(tr_layer_id, tr_idx, width=width, grid=self._grid)
            warr = WireArray(cur_tid, track_lower, track_upper)
            self._layout.add_warr(cur_tid, track_lower, track_upper)
            ans.append(warr)

        self._use_color = True
        return ans

    def draw_vias_on_intersections(self, bot_warr_list: Union[WireArray, List[WireArray]],
                                   top_warr_list: Union[WireArray, List[WireArray]]) -> None:
        """Draw vias on all intersections of the two given wire groups.

        Parameters
        ----------
        bot_warr_list : Union[WireArray, List[WireArray]]
            the bottom wires.
        top_warr_list : Union[WireArray, List[WireArray]]
            the top wires.
        """
        for bwarr in WireArray.wire_grp_iter(bot_warr_list):
            for twarr in WireArray.wire_grp_iter(top_warr_list):
                self._layout.add_via_on_intersections(bwarr.track_id, twarr.track_id,
                                                      bwarr.lower, bwarr.upper,
                                                      twarr.lower, twarr.upper, True, True)

    def mark_bbox_used(self, layer_id: int, bbox: BBox) -> None:
        """Marks the given bounding-box region as used in this Template."""
        # TODO: Fix this
        raise ValueError('Not implemented yet')

    def do_max_space_fill(self, layer_id: int, bound_box: Optional[BBox] = None,
                          fill_boundary: bool = True) -> None:
        """Draw density fill on the given layer."""
        if bound_box is None:
            bound_box = self.bound_box

        fill_info = self.grid.tech_info.get_max_space_fill_info(layer_id)
        self._layout.do_max_space_fill(layer_id, bound_box, fill_boundary, fill_info.info)
        self._use_color = True

    def do_device_fill(self, fill_cls: Type[TemplateBase], **kwargs: Any) -> None:
        """Fill empty region with device fills."""
        bbox = self.bound_box
        if bbox is None:
            raise ValueError('bound_box attribute is not set.')

        lookup = RTree()
        ed = ImmutableSortedDict()
        lookup.insert(None, bbox)

        # subtract instance bounding boxes
        for inst in self._instances.values():
            if inst.committed:
                inst_box = inst.bound_box
                inst_edges = inst.master.edge_info
                if inst_edges is None:
                    # TODO: implement this.  Need to recurse down instance hierarchy
                    raise ValueError('Not implemented, see developer.')
                # save items in list, because we'll remove them from the index
                item_list = list(lookup.intersect_iter(inst_box))
                for box, item_id in item_list:
                    if box.get_intersect(inst_box).is_physical():
                        box_edges = cast(Optional[TemplateEdgeInfo], lookup.pop(item_id))
                        _update_device_fill_area(lookup, ed, inst_box, inst_edges, box, box_edges)

        # draw fill
        cnt = 0
        for box, obj_id in lookup:
            kwargs['width'] = box.w
            kwargs['height'] = box.h
            kwargs['edges'] = lookup[obj_id]
            master = self.new_template(fill_cls, params=kwargs)
            self.add_instance(master, inst_name=f'XFILL{cnt}', xform=Transform(box.xl, box.yl))
            cnt += 1

    def do_power_fill(self, layer_id: int, tr_manager: TrackManager,
                      vdd_warrs: Optional[Union[WireArray, List[WireArray]]] = None,
                      vss_warrs: Optional[Union[WireArray, List[WireArray]]] = None, bound_box: Optional[BBox] = None,
                      x_margin: int = 0, y_margin: int = 0, sup_type: str = 'both', flip: bool = False,
                      uniform_grid: bool = False) -> Tuple[List[WireArray], List[WireArray]]:
        """Draw power fill on the given layer. Wrapper around do_multi_power_fill method
        that only returns the VDD and VSS wires.

        Parameters
        ----------
        layer_id : int
            the layer ID on which to draw power fill.
        tr_manager : TrackManager
            the TrackManager object.
        sup_list : List[Union[WireArray, List[WireArray]]]
            a list of supply wires to draw power fill for.
        bound_box : Optional[BBox]
            bound box over which to draw the power fill
        x_margin : int
            keepout margin on the x-axis. Fill is centered within margin.
        y_margin : int
            keepout margin on the y-axis. Fill is centered within margin.
        uniform_grid : bool
            draw power fill on a common grid instead of dense packing.
        flip : bool
            true to reverse order of power fill. Default (False) is {VDD, VSS}.

        Returns
        -------
        Tuple[List[WireArray], List[WireArray]]
            Tuple of VDD and VSS wires. If only one supply was specified, the other will be an empty list.
        """
        # Value checks
        if sup_type.lower() not in ['vdd', 'vss', 'both']:
            raise ValueError('sup_type has to be "VDD" or "VSS" or "both"(default)')
        if not vdd_warrs and not vss_warrs:
            raise ValueError('At least one of vdd_warrs or vss_warrs must be given.')

        # Build supply lists based on specficiation
        if sup_type.lower() == 'both' and vdd_warrs and vss_warrs:
            top_lists = [vdd_warrs, vss_warrs]
        elif sup_type.lower() == 'vss' and vss_warrs:
            top_lists = [vss_warrs]
        elif sup_type.lower() == 'vdd' and vdd_warrs:
            top_lists = [vdd_warrs]
        else:
            raise RuntimeError('Provided supply type and supply wires do not match.')

        # Run the actual power fill using the multi_power_fill function
        ret_warrs = self.do_multi_power_fill(layer_id, tr_manager, top_lists, bound_box,
                                             x_margin, y_margin, flip, uniform_grid)

        # Reorganize return values
        if sup_type.lower() == 'both':
            top_vdd, top_vss = (ret_warrs[0], ret_warrs[1]) if not flip else (ret_warrs[1], ret_warrs[0])
        elif sup_type.lower() == 'vss':
            top_vdd = []
            top_vss = ret_warrs[0]
        elif sup_type.lower() == 'vdd':
            top_vss = []
            top_vdd = ret_warrs[0]

        return top_vdd, top_vss

    def do_multi_power_fill(self, layer_id: int, tr_manager: TrackManager, sup_list: List[Union[WireArray, List[WireArray]]],
                            bound_box: Optional[BBox] = None, x_margin: int = 0, y_margin: int = 0, flip: bool = False,
                            uniform_grid: bool = False) -> List[List[WireArray]]:
        """Draw power fill on the given layer. Accepts as many different supply nets as provided.

        Parameters
        ----------
        layer_id : int
            the layer ID on which to draw power fill.
        tr_manager : TrackManager
            the TrackManager object.
        sup_list : List[Union[WireArray, List[WireArray]]]
            a list of supply wires to draw power fill for.
        bound_box : Optional[BBox]
            bound box over which to draw the power fill
        x_margin : int
            keepout margin on the x-axis. Fill is centered within margin.
        y_margin : int
            keepout margin on the y-axis. Fill is centered within margin.
        uniform_grid : bool
            draw power fill on a common grid instead of dense packing.
        flip : bool
            true to reverse order of power fill. Default is False.

        Returns
        -------
        List[List[WireArray]]
            List of the wire arrays for each supply in sup_list, given in the
            same order as sup_list.
        """
        if bound_box is None:
            if self.bound_box is None:
                raise ValueError("bound_box is not set")
            bound_box = self.bound_box
        bound_box = bound_box.expand(dx=-x_margin, dy=-y_margin)
        is_horizontal = (self.grid.get_direction(layer_id) == 0)
        num_sups = len(sup_list)
        if is_horizontal:
            cl, cu = bound_box.yl, bound_box.yh
            lower, upper = bound_box.xl, bound_box.xh
        else:
            cl, cu = bound_box.xl, bound_box.xh
            lower, upper = bound_box.yl, bound_box.yh
        fill_width = tr_manager.get_width(layer_id, 'sup')
        fill_space = tr_manager.get_sep(layer_id, ('sup', 'sup'))
        sep_margin = tr_manager.get_sep(layer_id, ('sup', ''))
        tr_bot = self.grid.coord_to_track(layer_id, cl, mode=RoundMode.GREATER_EQ)
        tr_top = self.grid.coord_to_track(layer_id, cu, mode=RoundMode.LESS_EQ)
        trs = self.get_available_tracks(layer_id, tid_lo=tr_bot, tid_hi=tr_top, lower=lower, upper=upper,
                                        width=fill_width, sep=fill_space, sep_margin=sep_margin,
                                        uniform_grid=uniform_grid)
        all_warrs = [[] for _ in range(num_sups)]
        htr_sep = HalfInt.convert(fill_space).dbl_value
        if len(trs) < num_sups:
            raise ValueError('Not enough available tracks to fill for all provided supplies')
        for ncur, tr_idx in enumerate(trs):
            warr = self.add_wires(layer_id, tr_idx, lower, upper, width=fill_width)
            _ncur = HalfInt.convert(tr_idx).dbl_value // htr_sep if uniform_grid else ncur
            if not flip:
                all_warrs[_ncur % num_sups].append(warr)
            else:
                all_warrs[(num_sups - 1) - (_ncur % num_sups)].append(warr)
        for top_warr, bot_warr in zip(sup_list, all_warrs):
            self.draw_vias_on_intersections(top_warr, bot_warr)
        return all_warrs

    def get_lef_options(self, options: Dict[str, Any], config: Mapping[str, Any]) -> None:
        """Populate the LEF options dictionary.

        Parameters
        ----------
        options : Mapping[str, Any]
            the result LEF options dictionary.
        config : Mapping[str, Any]
            the LEF configuration dictionary.
        """
        if not self.finalized:
            raise ValueError('This method only works on finalized master.')

        detail_layers_inc = config.get('detail_layers', [])

        top_layer = self.top_layer
        tech_info = self.grid.tech_info
        cover_layers = set(range(tech_info.bot_layer, top_layer + 1))
        detail_layers = set()
        for lay in detail_layers_inc:
            detail_layers.add(lay)
            cover_layers. discard(lay)

        options['detailed_layers'] = [lay for lay_id in sorted(detail_layers)
                                      for lay, _ in tech_info.get_lay_purp_list(lay_id)]
        options['cover_layers'] = [lay for lay_id in sorted(cover_layers)
                                   for lay, _ in tech_info.get_lay_purp_list(lay_id)]
        options['cell_type'] = config.get('cell_type', 'block')

    def find_track_width(self, layer_id: int, width: int) -> int:
        """Find the track width corresponding to the physical width

        Parameters
        ----------
        layer_id: int
            The metal layer ID
        width: int
            Physical width of the wire, in resolution units

        Returns
        -------
        tr_width: int
        """
        bin_iter = BinaryIterator(low=1)
        while bin_iter.has_next():
            cur = bin_iter.get_next()
            cur_width = self.grid.get_wire_total_width(layer_id, cur)
            if cur_width == width:
                return cur
            if cur_width < width:
                bin_iter.up()
            else:  # cur_width > width
                bin_iter.down()
        raise ValueError(f'Cannot find track width for width={width} on layer={layer_id}.')

    def connect_via_stack(self, tr_manager: TrackManager, warr: WireArray, top_layer: int, w_type: str = 'sig',
                          alignment_p: int = 0, alignment_o: int = 0,
                          mlm_dict: Optional[Mapping[int, MinLenMode]] = None,
                          ret_warr_dict: Optional[Mapping[int, WireArray]] = None,
                          coord_list_p_override: Optional[Sequence[int]] = None,
                          coord_list_o_override: Optional[Sequence[int]] = None, alternate_o: bool = False
                          ) -> WireArray:
        """Helper method to draw via stack and connections upto top layer, assuming connections can be on grid.
        Should work regardless of direction of top layer and bot layer.

        This method supports equally spaced WireArrays only. Needs modification for non uniformly spaced WireArrays.

        Parameters
        ----------
        tr_manager: TrackManager
            the track manager for this layout generator
        warr: WireArray
            The bot_layer wire array that has to via up
        top_layer: int
            The top_layer upto which stacked via has to go
        w_type: str
            The wire type, for querying widths from track manager
        alignment_p: int
            alignment for wire arrays which are parallel to bot_layer warr
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        alignment_o: int
            alignment for wire arrays which are orthogonal to bot_layer warr
        mlm_dict: Optional[Mapping[int, MinLenMode]]
            Dictionary of MinLenMode for every metal layer. Uses MinLenMode.MIDDLE by default
        ret_warr_dict: Optional[Mapping[int, WireArray]]
            If provided, this dictionary will contain all the WireArrays created during via stacking
        coord_list_p_override: Optional[Sequence[int]]
            List of co-ordinates for WireArrays parallel to bot_layer wire, assumed to be equally spaced
        coord_list_o_override: Optional[Sequence[int]]
            List of co-ordinates for WireArrays orthogonal to bot_layer wire, assumed to be equally spaced
        alternate_o: bool
            If coord_o_list is computed (i.e. coord_o_list_override is not used) then every other track is skipped
            for via spacing. Using alternate_o, we can choose which set of tracks is used and which is skipped.
            This is useful to avoid line end spacing issues when two adjacent wires require via stacks.

        Returns
        -------
        top_warr: WireArray
            The top_layer warr after via stacking all the way up
        """
        if ret_warr_dict is None:
            ret_warr_dict = {}
        if mlm_dict is None:
            mlm_dict = {}
        bot_layer = warr.layer_id
        # assert bot_layer + 2 <= top_layer, f'top_layer={top_layer} must be at least 2 higher than ' \
        #                                    f'bot_layer={bot_layer}  to have via stack'

        # Make sure widths are enough so that we can via up to top_layer
        top_layer_w = tr_manager.get_width(top_layer, w_type)
        w_dict = {top_layer: top_layer_w}
        for _layer in range(top_layer - 1, bot_layer - 1, -1):
            top_layer_w = w_dict[_layer] = self.grid.get_min_track_width(_layer, top_ntr=top_layer_w)
        assert warr.track_id.width >= w_dict[bot_layer], f'It is not possible to via up from given WireArray to the ' \
                                                         f'top_layer={top_layer} with width={top_layer_w} specified ' \
                                                         f'by the TrackManager.'

        # Find number of tracks to be used for layers with direction orthogonal to bot_layer, based on topmost
        # layer in that direction, called as top_layer_o ("_o" means orthogonal to bot_layer)
        bot_dir = self.grid.get_direction(bot_layer)
        top_dir = self.grid.get_direction(top_layer)
        top_layer_o = top_layer - 1 if bot_dir == top_dir else top_layer

        if coord_list_o_override is None:
            tidx_l = self.grid.coord_to_track(top_layer_o, warr.lower, RoundMode.GREATER_EQ)
            tidx_r = self.grid.coord_to_track(top_layer_o, warr.upper, RoundMode.LESS_EQ)
            # Divide by 2 for via separation
            num_wires_o = tr_manager.get_num_wires_between(top_layer_o, w_type, tidx_l, w_type, tidx_r, w_type) + 2
            num_wires_o = max(-(- num_wires_o // 2), 1)
            if num_wires_o == 1:
                tidx_list_o = [self.grid.coord_to_track(top_layer_o, warr.middle, RoundMode.NEAREST)]
            else:
                tidx_list_o = tr_manager.spread_wires(top_layer_o, [w_type] * (2 * num_wires_o - 1), tidx_l, tidx_r,
                                                      (w_type, w_type), alignment=alignment_o)
                tidx_list_o = tidx_list_o[1::2] if alternate_o else tidx_list_o[0::2]
                num_wires_o = len(tidx_list_o)
            # need to compute coord_list for conversion to tidx in layers which are same direction as top_layer_o
            coord_list_o = [self.grid.track_to_coord(top_layer_o, tidx) for tidx in tidx_list_o]
        else:
            num_wires_o = len(coord_list_o_override)
            coord_list_o = list(coord_list_o_override)
            coord_list_o.sort()

        if coord_list_p_override is None:
            # Find coord_list_p for co-ordinates of tracks of layers that are parallel to bot_layer
            coord_list_p = [self.grid.track_to_coord(bot_layer, tidx) for tidx in warr.track_id]
        else:
            coord_list_p = list(coord_list_p_override)
        coord_list_p.sort()

        for _layer in range(bot_layer + 1, top_layer + 1):
            _dir = self.grid.get_direction(_layer)
            _w = tr_manager.get_width(_layer, w_type)
            assert _w >= w_dict[_layer], f'It is not possible to via up because of width={_w} of layer={_layer} ' \
                                         f'specified by TrackManager.'
            _mlm = mlm_dict.get(_layer, MinLenMode.MIDDLE)
            if _dir == bot_dir:
                if coord_list_p_override is None and len(coord_list_p) > 1:
                    tidx_l = self.grid.coord_to_track(_layer, coord_list_p[0], RoundMode.NEAREST)
                    tidx_r = self.grid.coord_to_track(_layer, coord_list_p[-1], RoundMode.NEAREST)
                    # Divide by 2 for via separation
                    num_wires_p = tr_manager.get_num_wires_between(_layer, w_type, tidx_l, w_type, tidx_r, w_type) + 2
                    num_wires_p = max(-(- num_wires_p // 2), 1)
                    num_avail = len(coord_list_p)
                    if num_wires_p < num_avail:
                        # adjust num_wires_p to ensure that vias are formed in a stack
                        num_wires_p = min(num_wires_p, (num_avail + 1) // 2)
                        if num_avail % 2 == 0:
                            if alignment_p > 0:
                                tidx_l = self.grid.coord_to_track(_layer, coord_list_p[1], RoundMode.NEAREST)
                            else:
                                tidx_r = self.grid.coord_to_track(_layer, coord_list_p[-2], RoundMode.NEAREST)
                    else:  # num_wires_p >= num_avail:
                        # use entire coord_list_p
                        num_wires_p = num_avail
                    _tidx_list = tr_manager.spread_wires(_layer, [w_type] * num_wires_p, tidx_l, tidx_r,
                                                         (w_type, w_type),
                                                         alignment=0)
                    _num = num_wires_p
                else:
                    _num = len(coord_list_p)
                    _tidx_list = [self.grid.coord_to_track(_layer, coord, RoundMode.NEAREST) for coord in coord_list_p]
            else:
                _tidx_list = [self.grid.coord_to_track(_layer, coord, RoundMode.NEAREST) for coord in coord_list_o]
                _num = num_wires_o
            sep = _tidx_list[1] - _tidx_list[0] if _num > 1 else 0
            # TODO: support non-uniformly spaced list of WireArrays
            warr = self.connect_to_tracks(warr, TrackID(_layer, _tidx_list[0], _w, num=_num, pitch=sep),
                                          min_len_mode=_mlm)
            ret_warr_dict[_layer] = warr

        return warr

    @property
    def has_guard_ring(self) -> bool:
        return self._grid.tech_info.has_guard_ring


def _update_device_fill_area(lookup: RTree, ed: Param, inst_box: BBox, inst_edges: TemplateEdgeInfo,
                             sp_box: BBox, sp_edges: Optional[TemplateEdgeInfo]) -> None:
    # find instance edge with no constraints
    cut_edge_dir: Optional[Direction2D] = None
    cut_edge_dir_backup: Optional[Direction2D] = None
    two_backup = False
    # start at 1 so we prefer cutting horizontally
    for edir in (Direction2D.SOUTH, Direction2D.EAST, Direction2D.NORTH, Direction2D.WEST):
        if not inst_edges.get_edge_params(edir):
            if inst_edges.get_edge_params(edir.flip()):
                two_backup = cut_edge_dir_backup is not None
                if not two_backup:
                    cut_edge_dir_backup = edir
            else:
                cut_edge_dir = edir
                break

    bxl = sp_box.xl
    byl = sp_box.yl
    bxh = sp_box.xh
    byh = sp_box.yh
    ixl = inst_box.xl
    iyl = inst_box.yl
    ixh = inst_box.xh
    iyh = inst_box.yh
    if sp_edges is None:
        bel = beb = ber = bet = ed
    else:
        bel, beb, ber, bet = sp_edges.to_tuple()
    iel, ieb, ier, iet = inst_edges.to_tuple()
    sq_list = [(BBox(bxl, byl, ixl, iyl), (bel, beb, ed, ed)),
               (BBox(ixl, byl, ixh, iyl), (ed, beb, ed, iet)),
               (BBox(ixh, byl, bxh, iyl), (ed, beb, ber, ed)),
               (BBox(ixh, iyl, bxh, iyh), (ier, ed, ber, ed)),
               (BBox(ixh, iyh, bxh, byh), (ed, ed, ber, bet)),
               (BBox(ixl, iyh, ixh, byh), (ed, ieb, ed, bet)),
               (BBox(bxl, iyh, ixl, byh), (bel, ed, ed, bet)),
               (BBox(bxl, iyl, ixl, iyh), (bel, ed, iel, ed)),
               ]
    if cut_edge_dir is not None:
        # found opposite edges with no constraints, we're done
        if cut_edge_dir.is_vertical:
            # cut horizontally
            tile_list = [sq_list[3], sq_list[7], _fill_merge(sq_list, 0, True),
                         _fill_merge(sq_list, 4, True)]
        else:
            # cut vertically
            tile_list = [sq_list[1], sq_list[5], _fill_merge(sq_list, 2, True),
                         _fill_merge(sq_list, 6, True)]
    elif cut_edge_dir_backup is not None:
        if two_backup:
            # two adjacent cut idx.  Cut horizontally
            istart = 2 * cut_edge_dir_backup.value + 3
            istop = istart + 3
            if istop > 8:
                tile_list = sq_list[istart:]
                tile_list.extend(sq_list[:istop - 8])
            else:
                tile_list = sq_list[istart:istop]

            istart = istop % 8
            tile_list.append(_fill_merge(sq_list, istart, True))
            tile_list.append(_fill_merge(sq_list, (istart + 3) % 8, False))
        else:
            istart = 2 * cut_edge_dir_backup.value + 1
            istop = istart + 5
            if istop > 8:
                tile_list = sq_list[istart:]
                tile_list.extend(sq_list[:istop - 8])
            else:
                tile_list = sq_list[istart:istop]

            istart = istop % 8
            tile_list.append(_fill_merge(sq_list, istart, True))
    else:
        tile_list = sq_list

    for box, edges in tile_list:
        if box.is_physical():
            lookup.insert(edges, box)


def _fill_merge(sq_list: List[Tuple[BBox, Tuple[Param, Param, Param, Param]]],
                istart: int, merge_two: bool) -> Tuple[BBox, Tuple[Param, Param, Param, Param]]:
    box = sq_list[istart][0]
    edges = list(sq_list[istart][1])
    istop = istart + 3 if merge_two else istart + 2
    for idx in range(istart + 1, istop):
        cur_box, cur_edges = sq_list[idx % 8]
        if not box.is_physical():
            box = cur_box
            edges = list(cur_edges)
        elif cur_box.is_physical():
            if cur_box.xl < box.xl:
                edges[0] = cur_edges[0]
            if cur_box.yl < box.yl:
                edges[1] = cur_edges[1]
            if cur_box.xh > box.xh:
                edges[2] = cur_edges[2]
            if cur_box.yh > box.yh:
                edges[3] = cur_edges[3]
            box.merge(cur_box)
    return box, (edges[0], edges[1], edges[2], edges[3])
