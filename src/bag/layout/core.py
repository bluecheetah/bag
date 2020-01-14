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

"""This module defines some core layout classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, TypeVar, Union, Iterable, List, Callable

from pybag.core import PyLayInstRef, BBox, Transform, BBoxArray

from .routing.base import Port, WireArray

if TYPE_CHECKING:
    from .template import TemplateBase

T = TypeVar('T')


class PyLayInstance:
    def __init__(self, parent: TemplateBase, master: TemplateBase, ref: PyLayInstRef) -> None:
        self._parent = parent
        self._master = master
        self._ref = ref

    @property
    def name(self) -> str:
        return self._ref.inst_name

    @property
    def committed(self) -> bool:
        return self._ref.committed

    @property
    def nx(self) -> int:
        """int: Number of columns."""
        return self._ref.nx

    @property
    def ny(self) -> int:
        """int: Number of rows."""
        return self._ref.ny

    @property
    def spx(self) -> int:
        """int: The column pitch."""
        return self._ref.spx

    @property
    def spy(self) -> int:
        """int: The row pitch."""
        return self._ref.spy

    @property
    def master(self) -> TemplateBase:
        """TemplateBase: The master of this instance."""
        return self._master

    @property
    def transformation(self) -> Transform:
        """Transform: The instance transformation object."""
        return self._ref.xform

    @property
    def bound_box(self) -> BBox:
        """BBox: Returns the overall bounding box of this instance."""
        barr = BBoxArray(self._master.bound_box, nx=self.nx, ny=self.ny, spx=self.spx, spy=self.spy)
        return barr.transform(self.transformation).bound_box

    @property
    def array_box(self) -> BBox:
        """Returns the array box of this instance."""
        master_box: BBox = getattr(self._master, 'array_box', None)
        if master_box is None:
            raise ValueError('Master template array box is not defined.')

        barr = BBoxArray(master_box, nx=self.nx, ny=self.ny, spx=self.spx, spy=self.spy)
        return barr.transform(self.transformation).bound_box

    @property
    def fill_box(self) -> BBox:
        """Returns the fill box of this instance."""
        master_box: BBox = getattr(self._master, 'fill_box', None)
        if master_box is None:
            raise ValueError('Master template fill box is not defined.')

        barr = BBoxArray(master_box, nx=self.nx, ny=self.ny, spx=self.spx, spy=self.spy)
        return barr.transform(self.transformation).bound_box

    @nx.setter
    def nx(self, val: int) -> None:
        self._ref.nx = val

    @ny.setter
    def ny(self, val: int) -> None:
        self._ref.ny = val

    @spx.setter
    def spx(self, val: int) -> None:
        self._ref.spx = val

    @spy.setter
    def spy(self, val: int) -> None:
        self._ref.spy = val

    def get_item_location(self, row: int = 0, col: int = 0) -> Tuple[int, int]:
        """Returns the location of the given item in the array.

        Parameters
        ----------
        row : int
            the item row index.  0 is the bottom-most row.
        col : int
            the item column index.  0 is the left-most column.

        Returns
        -------
        xo : int
            the item X coordinate.
        yo : int
            the item Y coordinate.
        """
        if row < 0 or row >= self.ny or col < 0 or col >= self.nx:
            raise ValueError('Invalid row/col index: row=%d, col=%d' % (row, col))

        return col * self.spx, row * self.spy

    def get_bound_box_of(self, row: int = 0, col: int = 0) -> BBox:
        """Returns the bounding box of an instance in this mosaic.

        Parameters
        ----------
        row : int
            the item row index.  0 is the bottom-most row.
        col : int
            the item column index.  0 is the left-most column.

        Returns
        -------
        bbox : BBox
            the bounding box.
        """
        dx, dy = self.get_item_location(row=row, col=col)
        box = self._master.bound_box.get_transform(self.transformation)
        return box.move_by(dx, dy)

    def move_by(self, dx: int = 0, dy: int = 0) -> None:
        """Moves this instance by the given amount.

        Parameters
        ----------
        dx : int
            the X shift.
        dy : int
            the Y shift.
        """
        self._ref.move_by(dx, dy)

    def transform(self, xform: Transform) -> None:
        """Transform the location of this instance.

        Parameters
        ----------
        xform : Transform
            the transformation to apply to this instance.
        """
        self._ref.transform(xform)

    def new_master_with(self, **kwargs: Any) -> None:
        """Change the master template of this instance.

        This method will get the old master template layout parameters, update
        the parameter values with the given dictionary, then create a new master
        template with those parameters and associate it with this instance.

        Parameters
        ----------
        **kwargs : Any
            a dictionary of new parameter values.
        """
        self._master = self._master.new_template_with(**kwargs)
        self._ref.set_master(self._master.layout_cellview)

    def transform_master_object(self, obj: T, row: int = 0, col: int = 0) -> T:
        """Transforms the given object in instance master w.r.t. this instance's Transform object.

        Parameters
        ----------
        obj : T
            the object to transform.  Must have get_transform() method defined.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.

        Returns
        -------
        ans : T
            the transformed object.
        """
        dx, dy = self.get_item_location(row=row, col=col)
        xform = self.transformation.get_move_by(dx, dy)
        if isinstance(obj, Transform):
            return obj.get_transform_by(xform)
        return obj.get_transform(xform)

    def get_port(self, name: str = '', row: int = 0, col: int = 0) -> Port:
        """Returns the port object of the given instance in the array.

        Parameters
        ----------
        name : str
            the port terminal name.  If empty, check if this
            instance has only one port, then return it.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.

        Returns
        -------
        port : Port
            the port object.
        """
        return self.transform_master_object(self._master.get_port(name), row, col)

    def get_primitive_port(self, name: str, check_fun: Callable[[int], bool],
                           row: int = 0, col: int = 0) -> Port:
        """Returns the port object of the given instance in the array.

        Parameters
        ----------
        name : str
            the port terminal name.  If empty, check if this
            instance has only one port, then return it.
        check_fun : Callable[[int], bool]
            a function that returns true if a layer is considered "private".
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.

        Returns
        -------
        port : Port
            the port object.
        """
        port = self._master.get_port(name).to_primitive(self._master.tr_colors, check_fun)
        return self.transform_master_object(port, row, col)

    def get_pin(self, name: str = '', row: int = 0, col: int = 0, layer: int = Port.default_layer
                ) -> Union[WireArray, BBox]:
        """Returns the first pin with the given name.

        This is an efficient method if you know this instance has exactly one pin.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Returns
        -------
        pin : Union[WireArray, BBox]
            the first pin associated with the port of given name.
        """
        return self.get_port(name, row, col).get_pins(layer)[0]

    def port_pins_iter(self, name: str = '', layer: Union[int, str] = Port.default_layer
                       ) -> Iterable[Union[WireArray, BBox]]:
        """Iterate through all pins of all ports with the given name in this instance array.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        layer : Union[int, str]
            the pin layer ID or primitive pin layer name.  If negative, check to see if the
            given port has only one layer.  If so then use that layer.

        Yields
        ------
        pin : Union[WireArray, BBox]
            the pin as WireArray or bounding box.
        """
        for col in range(self.nx):
            for row in range(self.ny):
                try:
                    port = self.get_port(name, row, col)
                except KeyError:
                    return
                for obj in port.get_pins(layer):
                    yield obj

    def get_all_port_pins(self, name: str = '', layer: Union[int, str] = Port.default_layer
                          ) -> Union[List[WireArray], List[BBox]]:
        """Returns a list of all pins of all ports with the given name in this instance array.

        This method gathers ports from all instances in this array with the given name,
        then find all pins of those ports on the given layer, then return as list of WireArrays.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        layer : Union[int, str]
            the pin layer ID or primitive pin layer name.  If negative, check to see if the
            given port has only one layer.  If so then use that layer.

        Returns
        -------
        pin_list : Union[List[WireArray], List[BBox]]
            the list of pins as WireArrays or bounding boxes.
        """
        return list(self.port_pins_iter(name=name, layer=layer))

    def port_names_iter(self) -> Iterable[str]:
        """Iterates over port names in this instance.

        Yields
        ------
        port_name : str
            name of a port in this instance.
        """
        return self._master.port_names_iter()

    def has_port(self, port_name: str) -> bool:
        """Returns True if this instance has the given port."""
        return self._master.has_port(port_name)

    def commit(self) -> None:
        parent_grid = self._parent.grid
        old_master = self._master
        if parent_grid.tech_info.use_track_coloring and old_master.use_color:
            # update track parity
            tr_colors = parent_grid.get_track_coloring_at(self._parent.tr_colors,
                                                          self.transformation, old_master.grid,
                                                          old_master.top_layer)
            if tr_colors != old_master.tr_colors:
                self.new_master_with(tr_colors=tr_colors)

        self._parent.add_child_key(self._master.key)
        self._ref.commit()
