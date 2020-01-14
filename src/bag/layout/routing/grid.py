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

"""This module defines the RoutingGrid class.
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Dict, Any, Union

from warnings import warn
from dataclasses import dataclass

from pybag.core import PyRoutingGrid, Transform, coord_to_custom_htr
from pybag.enum import Orient2D, Direction, RoundMode

from bag.util.search import BinaryIterator
from bag.math import lcm
from bag.layout.tech import TechInfo

from ...util.math import HalfInt
from ...typing import TrackType

SizeType = Tuple[int, HalfInt, HalfInt]
FillConfigType = Dict[int, Tuple[int, int, int, int]]
OptHalfIntType = Optional[HalfInt]


@dataclass(eq=True, frozen=True)
class TrackSpec:
    layer: int
    direction: Orient2D
    width: int
    space: int
    offset: int


class RoutingGrid(PyRoutingGrid):
    """A class that represents the routing grid.

    This class provides various methods to convert between Cartesian coordinates and
    routing tracks.  This class assumes the lower-left coordinate is (0, 0)

    the track numbers are at half-track pitch.  That is, even track numbers corresponds
    to physical tracks, and odd track numbers corresponds to middle between two tracks.
    This convention is chosen so it is easy to locate a via for 2-track wide wires, for
    example.

    Assumptions:

    1. the pitch of all layers evenly divides the largest pitch.

    Parameters
    ----------
    tech_info : TechInfo
        the TechInfo instance used to create metals and vias.
    config_fname : str
        the routing grid configuration file.
    copy : Optional[PyRoutingGrid]
        copy create a new routing grid that's the same as the given copy
    """

    def __init__(self, tech_info: TechInfo, config_fname: str,
                 copy: Optional[PyRoutingGrid] = None) -> None:
        if copy is None:
            PyRoutingGrid.__init__(self, tech_info, config_fname)
        else:
            PyRoutingGrid.__init__(self, copy)
        self._tech_info = tech_info

    @classmethod
    def get_middle_track(cls, tr1: TrackType, tr2: TrackType, round_up: bool = False) -> HalfInt:
        """Get the track between the two given tracks."""
        tmp = HalfInt.convert(tr1)
        return (tmp + tr2).div2(round_up=round_up)

    @property
    def tech_info(self) -> TechInfo:
        """TechInfo: The TechInfo technology object."""
        return self._tech_info

    def is_horizontal(self, layer_id: int) -> bool:
        """Returns true if the given layer is horizontal."""
        return self.get_direction(layer_id) is Orient2D.x

    def get_num_tracks(self, size: SizeType, layer_id: int) -> HalfInt:
        """Returns the number of tracks on the given layer for a block with the given size.

        Parameters
        ----------
        size : SizeType
            the block size tuple.
        layer_id : int
            the layer ID.

        Returns
        -------
        num_tracks : HalfInt
            number of tracks on that given layer.
        """
        blk_dim = self.get_size_dimension(size)[self.get_direction(layer_id).value]
        tr_half_pitch = self.get_track_pitch(layer_id) // 2
        return HalfInt(blk_dim // tr_half_pitch)

    def dim_to_num_tracks(self, layer_id: int, dim: int, round_mode: RoundMode = RoundMode.NONE
                          ) -> HalfInt:
        """Returns how many track pitches are in the given dimension."""
        tr_pitch2 = self.get_track_pitch(layer_id) // 2
        q, r = divmod(dim, tr_pitch2)
        if round_mode is RoundMode.NONE:
            if r != 0:
                raise ValueError(f'Dimension {dim} is not divisible by half-pitch {tr_pitch2}')
        elif round_mode is RoundMode.LESS:
            q -= (r == 0)
        elif round_mode is RoundMode.GREATER_EQ:
            q += (r != 0)
        elif round_mode is RoundMode.GREATER:
            q += 1

        return HalfInt(q)

    def get_sep_tracks(self, layer: int, ntr1: int = 1, ntr2: int = 1,
                       same_color: bool = False, half_space: bool = True) -> HalfInt:
        """Returns the track separations needed between two adjacent wires.

        Parameters
        ----------
        layer : int
            wire layer ID.
        ntr1 : int
            width (in number of tracks) of the first wire.
        ntr2 : int
            width (in number of tracks) of the second wire.
        same_color : bool
            True to assume they  have the same color.
        half_space : bool
            True to allow half-track spacing.

        Returns
        -------
        sep_index : HalfInt
            minimum track index difference of the adjacent wires
        """
        htr = self.get_sep_htr(layer, ntr1, ntr2, same_color)
        return HalfInt(htr + (htr & (not half_space)))

    def get_line_end_sep_tracks(self, layer_dir: Direction, le_layer: int, le_ntr: int = 1,
                                adj_ntr: int = 1, half_space: bool = True) -> HalfInt:
        """Returns the track separations needed to satisfy via extension + line-end constraints.

        When you have two separate wires on the same track and need to connect them to adjacent
        layers, if the adjacent wires are too close, the via extensions could violate
        line-end spacing constraints.  This method computes the minimum track index difference
        those two wires must have to avoid this error.

        Parameters
        ----------
        layer_dir : Direction
            the direction of the specified layer.  LOWER if the layer is the
            bottom layer, UPPER if the layer is the top layer.
        le_layer : int
            line-end wire layer ID.
        le_ntr : int
            width (in number of tracks) of the line-end wire.
        adj_ntr : int
            width (in number of tracks) of the wire on the adjacent layer.
        half_space : bool
            True to allow half-track spacing.

        Returns
        -------
        sep_index : HalfInt
            minimum track index difference of the adjacent wires
        """
        htr = self.get_line_end_sep_htr(layer_dir.value, le_layer, le_ntr, adj_ntr)
        return HalfInt(htr + (htr & (not half_space)))

    def get_max_track_width(self, layer_id: int, num_tracks: int, tot_space: int,
                            half_end_space: bool = False) -> int:
        """Compute maximum track width and space that satisfies DRC rule.

        Given available number of tracks and numbers of tracks needed, returns
        the maximum possible track width.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        num_tracks : int
            number of tracks to draw.
        tot_space : int
            available number of tracks.
        half_end_space : bool
            True if end spaces can be half of minimum spacing.  This is true if you're
            these tracks will be repeated, or there are no adjacent tracks.

        Returns
        -------
        tr_w : int
            track width.
        """
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            tr_w = bin_iter.get_next()
            tr_sep = self.get_sep_tracks(layer_id, tr_w, tr_w)
            if half_end_space:
                used_tracks = tr_sep * num_tracks
            else:
                used_tracks = tr_sep * (num_tracks - 1) + 2 * self.get_sep_tracks(layer_id, tr_w, 1)
            if used_tracks > tot_space:
                bin_iter.down()
            else:
                bin_iter.save()
                bin_iter.up()

        opt_w = bin_iter.get_last_save()
        return opt_w

    @staticmethod
    def get_evenly_spaced_tracks(num_tracks: int, tot_space: int, track_width: int,
                                 half_end_space: bool = False) -> List[HalfInt]:
        """Evenly space given number of tracks in the available space.

        Currently this method may return half-integer tracks.

        Parameters
        ----------
        num_tracks : int
            number of tracks to draw.
        tot_space : int
            avilable number of tracks.
        track_width : int
            track width in number of tracks.
        half_end_space : bool
            True if end spaces can be half of minimum spacing.  This is true if you're
            these tracks will be repeated, or there are no adjacent tracks.

        Returns
        -------
        idx_list : List[HalfInt]
            list of track indices.  0 is the left-most track.
        """
        if half_end_space:
            tot_space_htr = 2 * tot_space
            scale = 2 * tot_space_htr
            offset = tot_space_htr + num_tracks
            den = 2 * num_tracks
        else:
            tot_space_htr = 2 * tot_space
            width_htr = 2 * track_width - 2
            # magic math.  You can work it out
            scale = 2 * (tot_space_htr + width_htr)
            offset = 2 * tot_space_htr - width_htr * (num_tracks - 1) + (num_tracks + 1)
            den = 2 * (num_tracks + 1)

        return [HalfInt((scale * idx + offset) // den - 1) for idx in range(num_tracks)]

    def get_fill_size(self, top_layer: int, fill_config: FillConfigType, *,
                      include_private: bool = False, half_blk_x: bool = True,
                      half_blk_y: bool = True) -> Tuple[int, int]:
        """Returns unit block size given the top routing layer and power fill configuration.

        Parameters
        ----------
        top_layer : int
            the top layer ID.
        fill_config : Dict[int, Tuple[int, int, int, int]]
            the fill configuration dictionary.
        include_private : bool
            True to include private layers in block size calculation.
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.

        Returns
        -------
        block_width : int
            the block width in resolution units.
        block_height : int
            the block height in resolution units.
        """
        blk_w, blk_h = self.get_block_size(top_layer, include_private=include_private,
                                           half_blk_x=half_blk_x, half_blk_y=half_blk_y)

        dim_list = [[blk_w], [blk_h]]
        for lay, (tr_w, tr_sp, _, _) in fill_config.items():
            if lay <= top_layer:
                cur_pitch = self.get_track_pitch(lay)
                cur_dim = (tr_w + tr_sp) * cur_pitch * 2
                dim_list[1 - self.get_direction(lay).value].append(cur_dim)

        blk_w = lcm(dim_list[0])
        blk_h = lcm(dim_list[1])
        return blk_w, blk_h

    def get_size_tuple(self, layer_id: int, width: int, height: int, *, round_up: bool = False,
                       half_blk_x: bool = False, half_blk_y: bool = False) -> SizeType:
        """Compute the size tuple corresponding to the given width and height from block pitch.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        width : int
            width of the block, in resolution units.
        height : int
            height of the block, in resolution units.
        round_up : bool
            True to round up instead of raising an error if the given width and height
            are not on pitch.
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.

        Returns
        -------
        size : SizeType
            the size tuple.  the first element is the top layer ID, second element is the width in
            number of vertical tracks, and third element is the height in number of
            horizontal tracks.
        """
        w_pitch, h_pitch = self.get_size_pitch(layer_id)

        wblk, hblk = self.get_block_size(layer_id, half_blk_x=half_blk_x, half_blk_y=half_blk_y)
        if width % wblk != 0:
            if round_up:
                width = -(-width // wblk) * wblk
            else:
                raise ValueError('width = %d not on block pitch (%d)' % (width, wblk))
        if height % hblk != 0:
            if round_up:
                height = -(-height // hblk) * hblk
            else:
                raise ValueError('height = %d not on block pitch (%d)' % (height, hblk))

        return layer_id, HalfInt(2 * width // w_pitch), HalfInt(2 * height // h_pitch)

    def get_size_dimension(self, size: SizeType) -> Tuple[int, int]:
        """Compute width and height from given size.

        Parameters
        ----------
        size : SizeType
            size of a block.

        Returns
        -------
        width : int
            the width in resolution units.
        height : int
            the height in resolution units.
        """
        w_pitch, h_pitch = self.get_size_pitch(size[0])
        return int(size[1] * w_pitch), int(size[2] * h_pitch)

    def convert_size(self, size: SizeType, new_top_layer: int) -> SizeType:
        """Convert the given size to a new top layer.

        Parameters
        ----------
        size : SizeType
            size of a block.
        new_top_layer : int
            the new top level layer ID.

        Returns
        -------
        new_size : SizeType
            the new size tuple.
        """
        wblk, hblk = self.get_size_dimension(size)
        return self.get_size_tuple(new_top_layer, wblk, hblk)

    def get_wire_bounds(self, layer_id: int, tr_idx: TrackType, width: int = 1) -> Tuple[int, int]:
        """Calculate the wire bounds coordinate.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        tr_idx : TrackType
            the center track index.
        width : int
            width of wire in number of tracks.

        Returns
        -------
        lower : int
            the lower bound coordinate perpendicular to wire direction.
        upper : int
            the upper bound coordinate perpendicular to wire direction.
        """
        return self.get_wire_bounds_htr(layer_id, int(round(2 * tr_idx)), width)

    def coord_to_track(self, layer_id: int, coord: int, mode: RoundMode = RoundMode.NONE,
                       even: bool = False) -> HalfInt:
        """Convert given coordinate to track number.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        mode : RoundMode
            the rounding mode.

            If mode == NEAREST, return the nearest track (default).

            If mode == LESS_EQ, return the nearest track with coordinate less
            than or equal to coord.

            If mode == LESS, return the nearest track with coordinate less
            than coord.

            If mode == GREATER, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with coordinate greater
            than coord.

            If mode == NONE, raise error if coordinate is not on track.

        even : bool
            True to round coordinate to integer tracks.

        Returns
        -------
        track : HalfInt
            the track number
        """
        return HalfInt(self.coord_to_htr(layer_id, coord, mode, even))

    def coord_to_fill_track(self, layer_id: int, coord: int, fill_config: Dict[int, Any],
                            mode: RoundMode = RoundMode.NEAREST) -> HalfInt:
        """Returns the fill track number closest to the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        fill_config : Dict[int, Any]
            the fill configuration dictionary.
        mode : RoundMode
            the rounding mode.

            If mode == NEAREST, return the nearest track (default).

            If mode == LESS_EQ, return the nearest track with coordinate less
            than or equal to coord.

            If mode == LESS, return the nearest track with coordinate less
            than coord.

            If mode == GREATER, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with coordinate greater
            than coord.

            If mode == NONE, raise error if coordinate is not on track.

        Returns
        -------
        track : HalfInt
            the track number
        """
        ntr_w, ntr_sp, _, _ = fill_config[layer_id]

        num_htr = round(2 * (ntr_w + ntr_sp))
        fill_pitch = num_htr * self.get_track_pitch(layer_id) // 2
        return HalfInt(coord_to_custom_htr(coord, fill_pitch, fill_pitch // 2, mode, False))

    def coord_to_nearest_track(self, layer_id: int, coord: int, *,
                               half_track: bool = True,
                               mode: Union[RoundMode, int] = RoundMode.NEAREST) -> HalfInt:
        """Returns the track number closest to the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        half_track : bool
            if True, allow half integer track numbers.
        mode : Union[RoundMode, int]
            the rounding mode.

            If mode == NEAREST, return the nearest track (default).

            If mode == LESS_EQ, return the nearest track with coordinate less
            than or equal to coord.

            If mode == LESS, return the nearest track with coordinate less
            than coord.

            If mode == GREATER, return the nearest track with coordinate greater
            than or equal to coord.

            If mode == GREATER_EQ, return the nearest track with coordinate greater
            than coord.

        Returns
        -------
        track : HalfInt
            the track number
        """
        warn('coord_to_nearest_track is deprecated, use coord_to_track with optional flags instead',
             DeprecationWarning)
        return HalfInt(self.coord_to_htr(layer_id, coord, mode, not half_track))

    def find_next_track(self, layer_id: int, coord: int, *, tr_width: int = 1,
                        half_track: bool = True,
                        mode: Union[RoundMode, int] = RoundMode.GREATER_EQ) -> HalfInt:
        """Find the track such that its edges are on the same side w.r.t. the given coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        coord : int
            the coordinate perpendicular to the track direction.
        tr_width : int
            the track width, in number of tracks.
        half_track : bool
            True to allow half integer track center numbers.
        mode : Union[RoundMode, int]
            the rounding mode.  NEAREST and NONE are not supported.

            If mode == LESS_EQ, return the track with both edges less
            than or equal to coord.

            If mode == LESS, return the nearest track with both edges less
            than coord.

            If mode == GREATER, return the nearest track with both edges greater
            than coord.

            If mode == GREATER_EQ, return the nearest track with both edges greater
            than or equal to coord.

        Returns
        -------
        tr_idx : HalfInt
            the center track index.
        """
        return HalfInt(self.find_next_htr(layer_id, coord, tr_width, mode, not half_track))

    def transform_track(self, layer_id: int, track_idx: TrackType, xform: Transform) -> HalfInt:
        """Transform the given track index.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        track_idx : TrackType
            the track index.
        xform : Transform
            the transformation object.

        Returns
        -------
        tidx : HalfInt
            the transformed track index.
        """
        return HalfInt(self.transform_htr(layer_id, int(round(2 * track_idx)), xform))

    def get_track_index_range(self, layer_id: int, lower: int, upper: int, *,
                              num_space: TrackType = 0, edge_margin: int = 0,
                              half_track: bool = True) -> Tuple[OptHalfIntType, OptHalfIntType]:
        """ Returns the first and last track index strictly in the given range.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        lower : int
            the lower coordinate.
        upper : int
            the upper coordinate.
        num_space : TrackType
            number of space tracks to the tracks right outside of the given range.
        edge_margin : int
            minimum space from outer tracks to given range.
        half_track : bool
            True to allow half-integer tracks.

        Returns
        -------
        start_track : OptHalfIntType
            the first track index.  None if no solution.
        end_track : OptHalfIntType
            the last track index.  None if no solution.
        """
        even = not half_track
        # get start track half index
        lower_bnd = self.find_next_track(layer_id, lower, mode=RoundMode.LESS_EQ)
        start_track = self.find_next_track(layer_id, lower + edge_margin, mode=RoundMode.GREATER_EQ)
        start_track = max(start_track, lower_bnd + num_space).up_even(even)

        # get end track half index
        upper_bnd = self.find_next_track(layer_id, upper, mode=RoundMode.GREATER_EQ)
        end_track = self.find_next_track(layer_id, upper - edge_margin, mode=RoundMode.LESS_EQ)
        end_track = min(end_track, upper_bnd - num_space).down_even(even)

        if end_track < start_track:
            # no solution
            return None, None
        return start_track, end_track

    def get_overlap_tracks(self, layer_id: int, lower: int, upper: int,
                           half_track: bool = True) -> Tuple[OptHalfIntType, OptHalfIntType]:
        """ Returns the first and last track index that overlaps with the given range.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        lower : int
            the lower coordinate.
        upper : int
            the upper coordinate.
        half_track : bool
            True to allow half-integer tracks.

        Returns
        -------
        start_track : OptHalfIntType
            the first track index.  None if no solution.
        end_track : OptHalfIntType
            the last track index.  None if no solution.
        """
        even = not half_track
        lower_tr = self.find_next_track(layer_id, lower, mode=RoundMode.LESS_EQ)
        lower_tr = lower_tr.up().up_even(even)
        upper_tr = self.find_next_track(layer_id, upper, mode=RoundMode.GREATER_EQ)
        upper_tr = upper_tr.down().down_even(even)

        if upper_tr < lower_tr:
            return None, None
        return lower_tr, upper_tr

    def track_to_coord(self, layer_id: int, track_idx: TrackType) -> int:
        """Convert given track number to coordinate.

        Parameters
        ----------
        layer_id : int
            the layer number.
        track_idx : TrackType
            the track number.

        Returns
        -------
        coord : int
            the coordinate perpendicular to track direction.
        """
        return self.htr_to_coord(layer_id, int(round(2 * track_idx)))

    def interval_to_track(self, layer_id: int, intv: Tuple[int, int]) -> Tuple[HalfInt, int]:
        """Convert given coordinates to track number and width.

        Parameters
        ----------
        layer_id : int
            the layer number.
        intv : Tuple[int, int]
            lower and upper coordinates perpendicular to the track direction.

        Returns
        -------
        track : HalfInt
            the track number
        width : int
            the track width, in number of tracks.
        """
        start, stop = intv
        htr = self.coord_to_htr(layer_id, (start + stop) // 2, RoundMode.NONE, False)
        width = stop - start

        # binary search to take width override into account
        bin_iter = BinaryIterator(1, None)
        while bin_iter.has_next():
            cur_ntr = bin_iter.get_next()
            wire_width = self.get_wire_total_width(layer_id, cur_ntr)
            if wire_width == width:
                return HalfInt(htr), cur_ntr
            elif wire_width > width:
                bin_iter.down()
            else:
                bin_iter.up()

        # never found solution; width is not quantized.
        raise ValueError('Interval {} on layer {} width not quantized'.format(intv, layer_id))

    def get_copy_with(self, top_ignore_lay: Optional[int] = None,
                      top_private_lay: Optional[int] = None,
                      tr_specs: Optional[List[TrackSpec]] = None
                      ) -> RoutingGrid:
        if top_ignore_lay is None:
            top_ignore_lay = self.top_ignore_layer
        if top_private_lay is None:
            top_private_lay = self.top_private_layer
        if tr_specs is None:
            tr_specs_cpp = []
        else:
            tr_specs_cpp = [(spec.layer, spec.direction.value, spec.width, spec.space, spec.offset)
                            for spec in tr_specs]

        new_grid = super(RoutingGrid, self).get_copy_with(top_ignore_lay, top_private_lay,
                                                          tr_specs_cpp)
        return RoutingGrid(self._tech_info, '', copy=new_grid)

