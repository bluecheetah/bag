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

"""This module provides basic routing classes.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Tuple, Union, Iterable, Iterator, Dict, List, Sequence, Any, Optional, Mapping,
    cast, Callable
)

from pybag.enum import RoundMode
from pybag.core import BBox, Transform, PyTrackID, TrackColoring, get_wire_iterator

from ...typing import TrackType
from ...util.math import HalfInt
from ...util.immutable import ImmutableSortedDict, combine_hash
from ...util.search import BinaryIterator

if TYPE_CHECKING:
    from .grid import RoutingGrid

WDictType = Mapping[str, Mapping[int, int]]
SpDictType = Mapping[Tuple[str, str], Mapping[int, TrackType]]


class TrackID(PyTrackID):
    """A class that represents locations of track(s) on the routing grid.

    Parameters
    ----------
    layer_id : int
        the layer ID.
    track_idx : TrackType
        the smallest middle track index in the array.  Multiples of 0.5
    width : int
        width of one track in number of tracks.
    num : int
        number of tracks in this array.
    pitch : TrackType
        pitch between adjacent tracks, in number of track pitches.
    grid: Optional[RoutingGrid]
        the routing grid associated with this TrackID object.
    """

    def __init__(self, layer_id: int, track_idx: TrackType, width: int = 1, num: int = 1,
                 pitch: TrackType = 0, grid: Optional[RoutingGrid] = None) -> None:
        if num < 1:
            raise ValueError('TrackID must have 1 or more tracks.')

        PyTrackID.__init__(self, layer_id, int(round(2 * track_idx)), width, num,
                           int(round(2 * pitch)))
        self._grid = grid

    def __iter__(self) -> Iterator[HalfInt]:
        """Iterate over all middle track indices in this TrackID."""
        return (HalfInt(self.base_htr + idx * self.htr_pitch) for idx in range(self.num))

    @property
    def base_index(self) -> HalfInt:
        """HalfInt: the base index."""
        return HalfInt(self.base_htr)

    @property
    def pitch(self) -> HalfInt:
        """HalfInt: the track pitch."""
        return HalfInt(self.htr_pitch)

    @property
    def grid(self) -> Optional[RoutingGrid]:
        """Optional[RoutingGrid]: the routing grid of this TrackID object."""
        return self._grid

    def __getitem__(self, idx: Union[int, slice]) -> TrackID:
        num = self.num
        pitch = self.pitch
        if isinstance(idx, int):
            if idx < 0:
                idx += num
            if idx < 0 or idx >= num:
                raise ValueError(f'Invalid index {idx} with {num} wires.')
            return TrackID(self.layer_id, self.base_index + idx * pitch, width=self.width,
                           grid=self._grid)
        else:
            start = idx.start
            stop = idx.stop
            step = idx.step
            if step is None:
                step = 1
            elif not isinstance(step, int):
                raise ValueError(f'TrackID slicing step {step} has to be integer')

            if start is None:
                start = 0
            elif start < 0:
                start += num
            if start < 0 or start >= num:
                raise ValueError(f'Invalid start index {start} with {num} wires.')

            if stop is None:
                stop = num
            elif stop < 0:
                stop += num
            if stop <= 0 or stop > num:
                raise ValueError(f'Invalid stop index {stop} with {num} wires.')

            if stop <= start:
                raise ValueError('slice got empty TrackID.')

            q, r = divmod(stop - start, step)
            return TrackID(self.layer_id, self.base_index + start * pitch, width=self.width,
                           num=q + (r != 0), pitch=step * pitch, grid=self._grid)

    def transform(self, xform: Transform) -> TrackID:
        """Transform this TrackID."""
        if self._grid is None:
            raise ValueError('Cannot transform TrackID without RoutingGrid.')

        lay_id = self.layer_id
        self.base_htr = self._grid.transform_htr(lay_id, self.base_htr, xform)
        axis_scale = xform.axis_scale[1 - self._grid.get_direction(lay_id).value]
        self.htr_pitch = self.htr_pitch * axis_scale
        return self

    def get_transform(self, xform: Transform) -> TrackID:
        """returns a transformed TrackID."""
        return TrackID(self.layer_id, self.base_index, width=self.width,
                       num=self.num, pitch=self.pitch, grid=self._grid).transform(xform)

    def copy_with(self, grid: RoutingGrid) -> TrackID:
        return TrackID(self.layer_id, self.base_index, width=self.width,
                       num=self.num, pitch=self.pitch, grid=grid)


class WireArray:
    """An array of wires on the routing grid.

    Parameters
    ----------
    track_id : TrackID
        TrackArray representing the track locations of this wire array.
    lower : int
        the lower coordinate along the track direction.
    upper : int
        the upper coordinate along the track direction.
    """

    def __init__(self, track_id: TrackID, lower: int, upper: int) -> None:
        self._tid = track_id
        self._lower = lower
        self._upper = upper

    @property
    def track_id(self) -> TrackID:
        """TrackID: The TrackID of this WireArray."""
        return self._tid

    @property
    def layer_id(self) -> int:
        return self._tid.layer_id

    @property
    def lower(self) -> int:
        return self._lower

    @property
    def upper(self) -> int:
        return self._upper

    @property
    def middle(self) -> int:
        return (self._lower + self._upper) // 2

    @property
    def bound_box(self) -> BBox:
        """BBox: the bounding box of this WireArray."""
        tid = self._tid
        layer_id = tid.layer_id
        grid = tid.grid
        if grid is None:
            raise ValueError('Cannot computing WireArray bounding box without RoutingGrid.')

        lower, upper = grid.get_wire_bounds_htr(layer_id, tid.base_htr, tid.width)

        delta = (tid.num - 1) * int(tid.pitch * grid.get_track_pitch(layer_id))
        if delta >= 0:
            upper += delta
        else:
            lower += delta

        return BBox(grid.get_direction(layer_id), self._lower, self._upper, lower, upper)

    @classmethod
    def list_to_warr(cls, warr_list: Sequence[WireArray]) -> WireArray:
        """Convert a list of WireArrays to a single WireArray.

        this method assumes all WireArrays have the same layer, width, and lower/upper coordinates.
        Overlapping WireArrays will be compacted.
        """
        if len(warr_list) == 1:
            return warr_list[0]

        tid0 = warr_list[0]._tid
        layer = tid0.layer_id
        width = tid0.width
        lower = warr_list[0].lower
        upper = warr_list[0].upper
        tid_list = sorted(set((idx for warr in warr_list for idx in warr.track_id)))
        base_idx = tid_list[0]
        if len(tid_list) < 2:
            return WireArray(TrackID(layer, base_idx, width=width, grid=tid0.grid), lower, upper)
        diff = tid_list[1] - tid_list[0]
        for idx in range(1, len(tid_list) - 1):
            if tid_list[idx + 1] - tid_list[idx] != diff:
                raise ValueError('pitch mismatch.')

        return WireArray(TrackID(layer, base_idx, width=width, num=len(tid_list), pitch=diff,
                                 grid=tid0.grid), lower, upper)

    @classmethod
    def single_warr_iter(cls, warr: Union[WireArray, Sequence[WireArray]]) -> Iterable[WireArray]:
        """Iterate through single wires in the given WireArray or WireArray list."""
        if isinstance(warr, WireArray):
            yield from warr.warr_iter()
        else:
            for w in warr:
                yield from w.warr_iter()

    @classmethod
    def wire_grp_iter(cls, warr: Union[WireArray, Sequence[WireArray]]) -> Iterable[WireArray]:
        """Iterate through WireArrays in the given WireArray or WireArray list."""
        if isinstance(warr, WireArray):
            yield warr
        else:
            yield from warr

    def __getitem__(self, idx: int) -> WireArray:
        return WireArray(self._tid[idx], self._lower, self._upper)

    def __repr__(self) -> str:
        return f'WireArray({self._tid}, {self._lower}, {self._upper})'

    def to_warr_list(self) -> List[WireArray]:
        """Convert this WireArray into a list of single wires."""
        return list(self.warr_iter())

    def warr_iter(self) -> Iterable[WireArray]:
        """Iterates through single wires in this WireArray."""
        tid = self._tid
        layer = tid.layer_id
        width = tid.width
        lower = self.lower
        upper = self.upper
        for tr in tid:
            yield WireArray(TrackID(layer, tr, width=width, grid=tid.grid), lower, upper)

    def wire_iter(self, tr_colors: TrackColoring) -> Iterable[Tuple[str, str, BBox]]:
        return get_wire_iterator(self._tid.grid, tr_colors, self._tid, self._lower, self._upper)

    def transform(self, xform: Transform) -> WireArray:
        """Transform this WireArray.

        Parameters
        ----------
        xform : Transform
            the transformation object.

        Returns
        -------
        warr : WireArray
            a reference to this object.
        """
        # noinspection PyAttributeOutsideInit
        self._tid = self._tid.get_transform(xform)
        layer_id = self._tid.layer_id
        dir_idx = self._tid.grid.get_direction(layer_id).value
        scale = xform.axis_scale[dir_idx]
        delta = xform.location[dir_idx]
        if scale < 0:
            tmp = -self._upper + delta
            self._upper = -self._lower + delta
            self._lower = tmp
        else:
            self._lower += delta
            self._upper += delta

        return self

    def get_transform(self, xform: Transform) -> WireArray:
        """Return a new transformed WireArray.

        Parameters
        ----------
        xform : Transform
            the transformation object.

        Returns
        -------
        warr : WireArray
            the new WireArray object.
        """
        return WireArray(self.track_id, self.lower, self.upper).transform(xform)


class Port:
    """A layout port.

    a port is a group of pins that represent the same net.
    The pins can be on different layers.

    Parameters
    ----------
    term_name : str
        the terminal name of the port.
    pin_dict : Dict[Union[int, str], Union[List[WireArray], List[BBox]]]
        a dictionary from layer ID to pin geometries on that layer.
    label : str
        the label of this port.
    """

    default_layer = -1000

    def __init__(self, term_name: str,
                 pin_dict: Dict[Union[int, str], Union[List[WireArray], List[BBox]]],
                 label: str, hidden: bool) -> None:
        self._term_name = term_name
        self._pin_dict = pin_dict
        self._label = label
        self._hidden = hidden

    def get_single_layer(self) -> Union[int, str]:
        """Returns the layer of this port if it only has a single layer."""
        if len(self._pin_dict) > 1:
            raise ValueError('This port has more than one layer.')
        return next(iter(self._pin_dict))

    def _get_layer(self, layer: Union[int, str]) -> Union[int, str]:
        """Get the layer ID or name."""
        if isinstance(layer, str):
            return self.get_single_layer() if not layer else layer
        else:
            return self.get_single_layer() if layer == Port.default_layer else layer

    @property
    def net_name(self) -> str:
        """str: The net name of this port."""
        return self._term_name

    @property
    def label(self) -> str:
        """str: The label of this port."""
        return self._label

    @property
    def hidden(self) -> bool:
        """bool: True if this is a hidden port."""
        return self._hidden

    def items(self) -> Iterable[Union[int, str], Union[List[WireArray], List[BBox]]]:
        return self._pin_dict.items()

    def get_pins(self, layer: Union[int, str] = -1000) -> Union[List[WireArray], List[BBox]]:
        """Returns the pin geometries on the given layer.

        Parameters
        ----------
        layer : Union[int, str]
            the layer ID.  If equal to Port.default_layer, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        track_bus_list : Union[List[WireArray], List[BBox]]
            pins on the given layer representing as WireArrays.
        """
        layer = self._get_layer(layer)
        return self._pin_dict.get(layer, [])

    def get_bounding_box(self, layer: Union[int, str] = -1000) -> BBox:
        """Calculate the overall bounding box of this port on the given layer.

        Parameters
        ----------
        layer : Union[int, str]
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        bbox : BBox
            the bounding box.
        """
        layer = self._get_layer(layer)
        box = BBox.get_invalid_bbox()
        for geo in self._pin_dict[layer]:
            if isinstance(geo, BBox):
                box.merge(geo)
            else:
                box.merge(geo.bound_box)
        return box

    def get_transform(self, xform: Transform) -> Port:
        """Return a new transformed Port.

        Parameters
        ----------
        xform : Transform
            the transform object.
        """
        new_pin_dict = {}
        for lay, geo_list in self._pin_dict.items():
            if isinstance(lay, str):
                new_geo_list = [cast(BBox, geo).get_transform(xform) for geo in geo_list]
            else:
                new_geo_list = [geo.get_transform(xform) for geo in geo_list]
            new_pin_dict[lay] = new_geo_list

        return Port(self._term_name, new_pin_dict, self._label, self._hidden)

    def to_primitive(self, tr_colors: TrackColoring, check_fun: Callable[[int], bool]) -> Port:
        new_pin_dict = {}
        for lay, geo_list in self._pin_dict.items():
            if isinstance(lay, int) and check_fun(lay):
                for geo in geo_list:
                    for blay, _, bbox in geo.wire_iter(tr_colors):
                        box_list: List[BBox] = new_pin_dict.get(blay, None)
                        if box_list is None:
                            new_pin_dict[blay] = box_list = []
                        box_list.append(bbox)
            else:
                new_pin_dict[lay] = geo_list

        return Port(self._term_name, new_pin_dict, self._label, self._hidden)


class TrackManager:
    """A class that makes it easy to compute track locations.

    This class provides many helper methods for computing track locations and spacing when
    each track could have variable width.  All methods in this class accepts a "track_type",
    which is either a string in the track dictionary or an integer representing the track
    width.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    tr_widths : WDictType
        dictionary from wire types to its width on each layer.
    tr_spaces : SpDictType
        dictionary from wire types to its spaces on each layer.
    **kwargs : Any
        additional options.
    """

    def __init__(self, grid: RoutingGrid, tr_widths: WDictType, tr_spaces: SpDictType,
                 **kwargs: Any) -> None:
        half_space = kwargs.get('half_space', True)

        self._grid = grid
        self._tr_widths = ImmutableSortedDict(tr_widths)
        self._tr_spaces = ImmutableSortedDict(tr_spaces)
        self._half_space = half_space

        # compute hash
        seed = hash(self._grid)
        seed = combine_hash(seed, hash(self._tr_widths))
        seed = combine_hash(seed, hash(self._tr_spaces))
        seed = combine_hash(seed, hash(self._half_space))
        self._hash = seed

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TrackManager):
            return (self._grid == other._grid and self._tr_widths == other._tr_widths and
                    self._tr_spaces == other._tr_spaces and self._half_space == other._half_space)
        else:
            return False

    @classmethod
    def _get_space_from_tuple(cls, layer_id: int, ntup: Tuple[str, str],
                              sp_dict: Optional[SpDictType]) -> Optional[TrackType]:
        if sp_dict is not None:
            test = sp_dict.get(ntup, None)
            if test is not None:
                return test.get(layer_id, None)
            ntup = (ntup[1], ntup[0])
            test = sp_dict.get(ntup, None)
            if test is not None:
                return test.get(layer_id, None)
        return None

    @property
    def grid(self) -> RoutingGrid:
        return self._grid

    @property
    def half_space(self) -> bool:
        return self._half_space

    @property
    def tr_widths(self) -> ImmutableSortedDict[str, ImmutableSortedDict[int, int]]:
        return self._tr_widths

    @property
    def tr_spaces(self) -> ImmutableSortedDict[Tuple[str, str],
                                               ImmutableSortedDict[int, TrackType]]:
        return self._tr_spaces

    def get_width(self, layer_id: int, track_type: Union[str, int]) -> int:
        """Returns the track width.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        track_type : Union[str, int]
            the track type.
        """
        if isinstance(track_type, int):
            return track_type
        if track_type not in self._tr_widths:
            return 1
        return self._tr_widths[track_type].get(layer_id, 1)

    def get_sep(self, layer_id: int, type_tuple: Tuple[Union[str, int], Union[str, int]],
                **kwargs: Any) -> HalfInt:
        """Returns the track separation.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        type_tuple : Tuple[Union[str, int], Union[str, int]],
            Tuple of the two types of wire.  If a type is an integer instead of a string,
            we use that as the track width.
        **kwargs : Any
            optional parameters.

        Returns
        -------
        tr_sp : TrackType
            the track spacing
        """
        same_color = kwargs.get('same_color', False)
        half_space = kwargs.get('half_space', self._half_space)
        sp_override = kwargs.get('sp_override', None)
        sp_dict = self._tr_spaces if sp_override is None else sp_override

        if isinstance(type_tuple[0], int):
            w1 = type_tuple[0]
            if isinstance(type_tuple[1], int):
                # user specify track width for both wires
                w2 = type_tuple[1]
                extra_sep = 0
            else:
                w2 = self.get_width(layer_id, type_tuple[1])
                extra_sep = self._get_space_from_tuple(layer_id, (type_tuple[1], ''), sp_dict)
                if extra_sep is None:
                    extra_sep = 0
        else:
            w1 = self.get_width(layer_id, type_tuple[0])
            if isinstance(type_tuple[1], int):
                w2 = type_tuple[1]
                extra_sep = self._get_space_from_tuple(layer_id, (type_tuple[0], ''), sp_dict)
                if extra_sep is None:
                    extra_sep = 0
            else:
                w2 = self.get_width(layer_id, type_tuple[1])
                extra_sep = self._get_space_from_tuple(layer_id, type_tuple, sp_dict)
                if extra_sep is None:
                    # check single spacing
                    extra_sep1 = self._get_space_from_tuple(layer_id, (type_tuple[0], ''), sp_dict)
                    if extra_sep1 is None:
                        extra_sep1 = 0
                    extra_sep2 = self._get_space_from_tuple(layer_id, (type_tuple[1], ''), sp_dict)
                    if extra_sep2 is None:
                        extra_sep2 = 0
                    extra_sep = max(extra_sep1, extra_sep2)

        ans = self._grid.get_sep_tracks(layer_id, w1, w2, same_color=same_color) + extra_sep
        return ans.up_even(not half_space)

    def get_next_track(self, layer_id: int, cur_idx: TrackType, cur_type: Union[str, int],
                       next_type: Union[str, int], up: Union[bool, int] = True, **kwargs: Any
                       ) -> HalfInt:
        """Compute the track location of a wire next to a given one.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        cur_idx : TrackType
            the current wire track index.
        cur_type : Union[str, int]
            the current wire type.
        next_type : Union[str, int]
            the next wire type.
        up : Union[bool, int]
            True to return the next track index that is larger than cur_idx. Can also be integer
            to count number of tracks.
        **kwargs : Any
            optional parameters.

        Returns
        -------
        next_int : HalfInt
            the next track index.
        """
        sep = self.get_sep(layer_id, (cur_type, next_type), **kwargs)
        sep1 = self.get_sep(layer_id, (next_type, next_type), **kwargs)
        cur_idx = HalfInt.convert(cur_idx)

        if isinstance(up, bool):
            up: int = 2 * int(up) - 1

        delta = sep + (abs(up) - 1) * sep1
        sign = up > 0
        return cur_idx + (2 * sign - 1) * delta

    def get_num_wires_between(self, layer_id: int, bot_wire: str, bot_idx: HalfInt,
                              top_wire: str, top_idx: HalfInt, fill_wire: str) -> int:
        idx0 = self.get_next_track(layer_id, bot_idx, bot_wire, fill_wire, up=True)
        idx1 = self.get_next_track(layer_id, top_idx, top_wire, fill_wire, up=False)
        if idx1 < idx0:
            return 0

        sep = self.get_sep(layer_id, (fill_wire, fill_wire))
        return ((idx1.dbl_value - idx0.dbl_value) // sep.dbl_value) + 1

    def place_wires(self, layer_id: int, type_list: Sequence[Union[str, int]],
                    align_track: Optional[HalfInt] = None, align_idx: int = 0,
                    center_coord: Optional[int] = None, **kwargs: Any
                    ) -> Tuple[HalfInt, List[HalfInt]]:
        """Place the given wires next to each other.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        align_track : Optional[HalfInt]
            If not None, will make sure the the track at location align_idx has this value.
        align_idx : Optional[int]
            the align wire index.
        center_coord : Optional[int]
            If not None, will try to center the wires around this coordinate.
            align_track takes precedence over center_coord.
        **kwargs : Any
            optional parameters for get_num_space_tracks() method of RoutingGrid.

        Returns
        -------
        num_tracks : HalfInt
            number of tracks used.
        locations : List[HalfInt]
            the center track index of each wire.
        """
        if not type_list:
            return HalfInt(0), []

        grid = self.grid

        w0 = self.get_width(layer_id, type_list[0])
        mid_idx = grid.find_next_track(layer_id, 0, tr_width=w0, half_track=True,
                                       mode=RoundMode.GREATER_EQ)

        ans = [mid_idx]
        num_wires = len(type_list)
        idx_half = num_wires // 2
        for idx in range(1, num_wires):
            cur_idx = self.get_next_track(layer_id, ans[-1], type_list[idx - 1],
                                          type_list[idx], up=True, **kwargs)
            ans.append(cur_idx)

        if align_track is not None:
            delta = align_track - ans[align_idx]
            for idx in range(num_wires):
                ans[idx] += delta
        elif center_coord is not None:
            if num_wires & 1:
                mid_coord = grid.track_to_coord(layer_id, ans[idx_half])
            else:
                coord1 = grid.track_to_coord(layer_id, ans[idx_half - 1])
                coord2 = grid.track_to_coord(layer_id, ans[idx_half])
                mid_coord = (coord1 + coord2) // 2

            coord_delta = center_coord - mid_coord
            delta = grid.coord_to_track(layer_id, coord_delta, mode=RoundMode.NEAREST)
            delta -= grid.coord_to_track(layer_id, 0)
            for idx in range(num_wires):
                ans[idx] += delta

        w1 = self.get_width(layer_id, type_list[-1])
        upper = grid.get_wire_bounds(layer_id, ans[-1], width=w1)[1]
        top_idx = grid.coord_to_track(layer_id, upper, mode=RoundMode.GREATER_EQ)
        lower = grid.get_wire_bounds(layer_id, ans[0],
                                     width=self.get_width(layer_id, type_list[0]))[0]
        bot_idx = grid.coord_to_track(layer_id, lower, mode=RoundMode.LESS_EQ)
        ntr = top_idx - bot_idx

        return ntr, ans

    @classmethod
    def _get_align_delta(cls, tot_ntr: TrackType, num_used: TrackType, alignment: int) -> HalfInt:
        if alignment == -1 or num_used == tot_ntr:
            # we already aligned to left
            return HalfInt(0)
        elif alignment == 0:
            # center tracks
            return HalfInt.convert(tot_ntr - num_used).div2()
        elif alignment == 1:
            # align to right
            return HalfInt.convert(tot_ntr - num_used)
        else:
            raise ValueError('Unknown alignment code: %d' % alignment)

    def align_wires(self, layer_id: int, type_list: Sequence[Union[str, int]], tot_ntr: TrackType,
                    alignment: int = 0, start_idx: TrackType = 0, **kwargs: Any) -> List[HalfInt]:
        """Place the given wires in the given space with the specified alignment.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        tot_ntr : TrackType
            total available space in number of tracks.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : TrackType
            the starting track index.
        **kwargs : Any
            optional parameters for place_wires().

        Returns
        -------
        locations : List[HalfInt]
            the center track index of each wire.
        """
        num_used, idx_list = self.place_wires(layer_id, type_list, start_idx=start_idx, **kwargs)
        if num_used > tot_ntr:
            raise ValueError('Given tracks occupy more space than given.')

        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]

    def get_next_track_obj(self,
                           warr_tid_obj: Union[TrackID, WireArray],
                           cur_type: Union[str, int],
                           next_type: Union[str, int],
                           count_rel_tracks: int = 1,
                           **kwargs) -> TrackID:
        """Computes next TrackID relative the WireArray or TrackID object, given wire types

        Parameters
        ----------
        warr_tid_obj: Union[TrackID, WireArray]
            the wire array or track id object used as the reference
        cur_type: Union[str, int]
            the wire type of current reference warr/tid
        next_type: Union[str, int]
            the wire type of the returned tid
        count_rel_tracks: int
            the number of spacings to skip
            +1 means the immediate next track id
            -1 means immediate previous track id,
            +2 means the one after the next track id, etc.
            if |count_rel_tracks| > 1, the skipped distance is
            space(cur_type, next_type) + (|count_rel_tracks| - 1) * space(next_type, next_type)

        Returns
        -------
        track_id : TrackID
            the TrackID object of the next track id
        """

        layer_id = warr_tid_obj.layer_id
        if isinstance(warr_tid_obj, TrackID):
            cur_idx = warr_tid_obj.base_index
        else:
            cur_idx = warr_tid_obj.track_id.base_index

        sep0 = self.get_sep(layer_id, (cur_type, next_type), **kwargs)
        sep1 = self.get_sep(layer_id, (next_type, next_type), **kwargs)
        cur_idx = HalfInt.convert(cur_idx)

        sign = count_rel_tracks > 0
        delta = sep0 + (abs(count_rel_tracks) - 1) * sep1
        next_tidx = cur_idx + (2 * sign - 1) * delta

        return TrackID(layer_id, next_tidx, width=self.get_width(layer_id, next_type))

    def get_shield_tracks(self, layer_id: int, tidx_lo: HalfInt, tidx_hi: HalfInt,
                          wtype_lo: Union[str, int], wtype_hi: Union[str, int]) -> List[TrackID]:
        """Fill the given space with shielding tracks

        Try to fill with the widest metal allowed in the PDK
        Respect DRC spacing rules relative to lower and higher wires
        Currently this method just returns a bunch of width 1 wires.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        tidx_lo : HalfInt
            lower bound track index
        tidx_hi : HalfInt
            upper bound track index
        wtype_lo: Union[str, int]
            type of lower bound wire
        wtype_hi: Union[str, int]
            type of upper bound wire

        Returns
        -------
        idx_list : List[TrackID]
            list of TrackIDs
        """
        tr_width = 1

        sh_tr_lower = self.get_next_track(layer_id, tidx_lo, wtype_lo, tr_width, up=True)
        sh_tr_upper = self.get_next_track(layer_id, tidx_hi, wtype_hi, tr_width, up=False)
        num_tracks = (sh_tr_upper - sh_tr_lower + 1) // 1
        tr_locs = [sh_tr_lower + i for i in range(num_tracks)]
        return [TrackID(layer_id, tr_idx, width=tr_width) for tr_idx in tr_locs]

    def spread_wires(self, layer_id: int, type_list: Sequence[Union[str, int]],
                     lower: HalfInt, upper: HalfInt, sp_type: Tuple[str, str],
                     alignment: int = 0, max_iter: int = 1000) -> List[HalfInt]:
        """Spread out the given wires in the given space.

        This method tries to spread out wires by increasing the space around the given
        wire/combination of wires.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        lower : HalfInt
            the lower bound track index, inclusive.
        upper : HalfInt
            the upper bound track index, inclusive.
        sp_type : Tuple[str, str]
            The space to increase.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        max_iter : int
            maximum number of iterations.

        Returns
        -------
        locations : List[HalfInt]
            the center track index of each wire.
        """
        test_dict = self._tr_spaces.get(sp_type, None)
        if test_dict is not None:
            cur_sp = test_dict.get(layer_id, 0)
        else:
            cur_sp = 0
        cur_sp = HalfInt.convert(cur_sp)

        sp_override = self._tr_spaces.to_dict()
        sp_override[sp_type] = cur_dict = {layer_id: cur_sp}

        if alignment < 0:
            align_track = lower
            align_idx = 0
            center_coord = None
        elif alignment > 0:
            align_track = upper
            align_idx = len(type_list) - 1
            center_coord = None
        else:
            grid = self.grid
            align_track = None
            align_idx = 0
            c0 = grid.track_to_coord(layer_id, lower)
            c1 = grid.track_to_coord(layer_id, upper)
            center_coord = (c0 + c1) // 2

        bin_iter = BinaryIterator(cur_sp.dbl_value, None)
        for cnt in range(max_iter):
            if not bin_iter.has_next():
                break
            new_sp_dbl = bin_iter.get_next()
            cur_dict[layer_id] = HalfInt(new_sp_dbl)
            result = self.place_wires(layer_id, type_list, align_track=align_track,
                                      align_idx=align_idx, center_coord=center_coord,
                                      sp_override=sp_override)[1]
            if result[0] < lower or result[-1] > upper:
                bin_iter.down()
            else:
                bin_iter.save_info(result)
                bin_iter.up()

        if bin_iter.get_last_save_info() is None:
            raise ValueError(f'Unable to place specified wires in range [{lower}, {upper}].')

        return bin_iter.get_last_save_info()
