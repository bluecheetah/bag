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

"""This module defines classes that provides automatic fill utility on a grid.
"""

from typing import Optional, List, Tuple

import dataclasses

from bag.util.search import BinaryIterator, minimize_cost_golden


@dataclasses.dataclass(eq=True)
class FillInfo:
    tot_area: int
    sp_nominal: int
    sp_edge: int
    sp_mid: int
    blk0: int
    blk1: int
    blkm: int
    num_half: int
    num_diff_sp: int
    num_blk1_half: int
    inc_sp: bool
    fill_on_edge: bool
    cyclic: bool
    invert: bool

    @property
    def num_fill(self) -> int:
        if self.invert:
            return 2 * (self.num_half - self.fill_on_edge) + (self.blkm >= 0) + 1
        else:
            return 2 * self.num_half + (self.blkm >= 0)

    @property
    def sp_max(self) -> int:
        if self.invert:
            return max(self.blk0, self.blk1)
        else:
            return self.sp_nominal + (self.num_diff_sp > 0 and self.inc_sp)

    @property
    def blk_min(self) -> int:
        if self.invert:
            return self.sp_nominal - (self.num_diff_sp > 0 and not self.inc_sp)
        else:
            return min(self.blk0, self.blk1)

    @property
    def blk_max(self) -> int:
        if self.invert:
            return self.sp_nominal + (self.num_diff_sp > 0 and self.inc_sp)
        else:
            return max(self.blk0, self.blk1)

    def get_fill_area(self, scale: int, extend: int) -> int:
        k = self.num_blk1_half
        m = self.num_half
        ans = 2 * (k * self.blk1 + (m - k) * self.blk0) + (self.blkm >= 0) * self.blkm
        # subtract double counted edge block
        ans -= (self.cyclic and self.fill_on_edge) * self.blk1
        ans = self.invert * self.tot_area + (1 - 2 * self.invert) * ans
        return scale * ans + extend * self.num_fill

    def meet_area_specs(self, area_specs: List[Tuple[int, int, int]]) -> bool:
        for target, scale, extend in area_specs:
            if self.get_fill_area(scale, extend) < target:
                return False
        return True

    def get_area_fom(self, area_specs: List[Tuple[int, int, int]]) -> int:
        fom = 0
        for target, scale, extend in area_specs:
            cur_area = self.get_fill_area(scale, extend)
            fom += min(0, cur_area - target)
        return fom


def fill_symmetric_max_density(area: int, n_min: int, n_max: int, sp_min: int,
                               area_specs: List[Tuple[int, int, int]],
                               sp_max: Optional[int] = None, fill_on_edge: bool = True,
                               cyclic: bool = False) -> List[Tuple[int, int]]:
    """Fill the given 1-D area with density constraints, using largest blocks possible.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. all fill blocks have lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    area_specs : List[Tuple[int, int, int]]
        list of area specifications, in (target, scale, extension) format.
    sp_max : Optional[int]
        if given, make sure space between blocks does not exceed this value.
        Must be greater than sp_min
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    fill_interval : List[Tuple[int, int]]
        a list of [start, stop) intervals that needs to be filled.
    """
    max_result = fill_symmetric_max_density_info(area, n_min, n_max, sp_min, area_specs,
                                                 sp_max=sp_max, fill_on_edge=fill_on_edge,
                                                 cyclic=cyclic)
    return fill_symmetric_interval(max_result)


def fill_symmetric_min_density(area: int, n_min: int, n_max: int, sp_min: int,
                               area_specs: List[Tuple[int, int, int]],
                               sp_max: Optional[int] = None, fill_on_edge: bool = True,
                               cyclic: bool = False) -> List[Tuple[int, int]]:
    info = fill_symmetric_min_density_info(area, n_min, n_max, sp_min, area_specs,
                                           sp_max=sp_max, fill_on_edge=fill_on_edge, cyclic=cyclic)
    return fill_symmetric_interval(info)


def fill_symmetric_min_density_info(area: int, n_min: int, n_max: int, sp_min: int,
                                    area_specs: List[Tuple[int, int, int]],
                                    sp_max: Optional[int] = None, fill_on_edge: bool = True,
                                    cyclic: bool = False) -> FillInfo:
    """Fill the given 1-D area to satisfy minimum density constraint

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. all fill blocks have lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    area_specs : List[Tuple[int, int, int]]
        list of area specifications, in (target, scale, extension) format.
    sp_max : Optional[int]
        if given, make sure space between blocks does not exceed this value.
        Must be greater than sp_min
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : FillInfo
        the fill information object.
    """
    # first, fill as much as possible using scale/extension of the first area spec.
    max_result = fill_symmetric_max_density_info(area, n_min, n_max, sp_min, area_specs,
                                                 sp_max=sp_max, fill_on_edge=fill_on_edge,
                                                 cyclic=cyclic)

    if not max_result.meet_area_specs(area_specs):
        # we cannot meet area spec; return max result
        return max_result

    # now, reduce fill by doing binary search on n_max
    nfill_opt = max_result.num_fill
    n_max_iter = BinaryIterator(n_min, n_max)
    while n_max_iter.has_next():
        n_max_cur = n_max_iter.get_next()
        try:
            info = fill_symmetric_max_num_info(area, nfill_opt, n_min, n_max_cur, sp_min,
                                               fill_on_edge=fill_on_edge, cyclic=cyclic)
            if info.meet_area_specs(area_specs) and (sp_max is None or info.sp_max <= sp_max):
                # both specs passed
                n_max_iter.save_info(info)
                n_max_iter.down()
            else:
                # reduce n_max too much
                n_max_iter.up()

        except ValueError:
            # get here if n_min == n_max and there's no solution.
            n_max_iter.up()

    last_save = n_max_iter.get_last_save_info()
    if last_save is None:
        # no solution, return max result
        return max_result
    else:
        max_result = last_save

    # see if we can further reduce fill by doing binary search on nfill_opt
    nfill_iter = BinaryIterator(1, nfill_opt)
    n_max = n_max_iter.get_last_save()
    while nfill_iter.has_next():
        nfill_cur = nfill_iter.get_next()
        try:
            info = fill_symmetric_max_num_info(area, nfill_cur, n_min, n_max, sp_min,
                                               fill_on_edge=fill_on_edge, cyclic=cyclic)
            if info.meet_area_specs(area_specs) and (sp_max is None or info.sp_max <= sp_max):
                # both specs passed
                nfill_iter.save_info(info)
                nfill_iter.down()
            else:
                # reduce nfill too much
                nfill_iter.up()

        except ValueError:
            nfill_iter.up()

    last_save = nfill_iter.get_last_save_info()
    if last_save is None:
        return max_result
    # return new minimum solution
    return last_save


def fill_symmetric_max_density_info(area: int, n_min: int, n_max: int, sp_min: int,
                                    area_specs: List[Tuple[int, int, int]],
                                    sp_max: Optional[int] = None, fill_on_edge: bool = True,
                                    cyclic: bool = False) -> FillInfo:
    """Fill the given 1-D area with density constraints, using largest blocks possible.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. all fill blocks have lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.
    5. we do the best to meet area specs by using the largest blocks possible.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    area_specs : List[Tuple[int, int, int]]
        list of area specifications, in (target, scale, extension) format.
    sp_max : Optional[int]
        if given, make sure space between blocks does not exceed this value.
        Must be greater than sp_min
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : FillInfo
        the fill information object.
    """

    # min area test
    nfill_min = 1
    try:
        try:
            fill_symmetric_max_num_info(area, nfill_min, n_min, n_max, sp_min,
                                        fill_on_edge=fill_on_edge, cyclic=cyclic)
        except (NoFillAbutEdgeError, NoFillChoiceError):
            # we need at least 2 fiils
            nfill_min = 2
            fill_symmetric_max_num_info(area, nfill_min, n_min, n_max, sp_min,
                                        fill_on_edge=fill_on_edge, cyclic=cyclic)
    except InsufficientAreaError:
        # cannot fill at all
        return _fill_symmetric_info(area, 0, area, inc_sp=False,
                                    fill_on_edge=fill_on_edge, cyclic=cyclic)

    if sp_max is not None:
        # find minimum nfill that meets sp_max spec
        if sp_max <= sp_min:
            raise ValueError(f'Cannot have sp_max = {sp_max} <= {sp_min} = sp_min')

        def sp_max_fun(nfill):
            try:
                info2 = fill_symmetric_max_num_info(area, nfill, n_min, n_max, sp_min,
                                                    fill_on_edge=fill_on_edge, cyclic=cyclic)
                return -info2.sp_max
            except ValueError:
                return -sp_max - 1

        min_result = minimize_cost_golden(sp_max_fun, -sp_max, offset=nfill_min, maxiter=None)
        if min_result.x is None:
            # try even steps
            min_result = minimize_cost_golden(sp_max_fun, -sp_max, offset=nfill_min,
                                              step=2, maxiter=None)
            nfill_min = min_result.x
            if nfill_min is None:
                raise MaxSpaceTooStrictError(f'No solution for sp_max = {sp_max}')
        else:
            nfill_min = min_result.x

    # fill area first monotonically increases with number of fill blocks, then monotonically
    # decreases (as we start adding more space than fill).  Therefore, a golden section search
    # can be done on the number of fill blocks to determine the optimum.
    worst_fom = -sum((spec[0] for spec in area_specs))

    def area_fun(nfill):
        try:
            info2 = fill_symmetric_max_num_info(area, nfill, n_min, n_max, sp_min,
                                                fill_on_edge=fill_on_edge, cyclic=cyclic)
            return info2.get_area_fom(area_specs)
        except ValueError:
            return worst_fom

    min_result = minimize_cost_golden(area_fun, area, offset=nfill_min, maxiter=None)
    nfill_opt = min_result.x
    if nfill_opt is None:
        nfill_opt = min_result.xmax
    info = fill_symmetric_max_num_info(area, nfill_opt, n_min, n_max, sp_min,
                                       fill_on_edge=fill_on_edge, cyclic=cyclic)
    return info


class MaxSpaceTooStrictError(ValueError):
    pass


class InsufficientAreaError(ValueError):
    pass


class FillTooSmallError(ValueError):
    pass


class NoFillAbutEdgeError(ValueError):
    pass


class NoFillChoiceError(ValueError):
    pass


class EmptyRegionError(ValueError):
    pass


def fill_symmetric_max_num_info(tot_area: int, nfill: int, n_min: int, n_max: int, sp_min: int,
                                fill_on_edge: bool = True, cyclic: bool = False) -> FillInfo:
    """Fill the given 1-D area as much as possible with given number of fill blocks.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. the area is filled as much as possible with exactly nfill blocks,
       with lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    tot_area : int
        total number of space we need to fill.
    nfill : int
        number of fill blocks to draw.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : FillInfo
        the fill information object.
    """
    # error checking
    if nfill < 0:
        raise ValueError(f'nfill = {nfill} < 0')
    if n_min > n_max:
        raise ValueError(f'n_min = {n_min} > {n_max} = n_max')
    if n_min <= 0:
        raise ValueError(f'n_min = {n_min} <= 0')

    if nfill == 0:
        # no fill at all
        return _fill_symmetric_info(tot_area, 0, tot_area, inc_sp=False,
                                    fill_on_edge=False, cyclic=False)

    # check no solution
    sp_delta = 0 if cyclic else (-1 if fill_on_edge else 1)
    nsp = nfill + sp_delta
    if n_min * nfill + nsp * sp_min > tot_area:
        raise InsufficientAreaError(f'Cannot draw {nfill} fill blocks with n_min = {n_min}')

    # first, try drawing nfill blocks without block length constraint.
    # may throw exception if no solution
    info = _fill_symmetric_info(tot_area, nfill, sp_min, inc_sp=True,
                                fill_on_edge=fill_on_edge, cyclic=cyclic)
    if info.blk_min < n_min:
        # could get here if cyclic = True, fill_on_edge = True, n_min is odd
        # in this case actually no solution
        raise FillTooSmallError(f'Cannot draw {nfill} fill blocks with n_min = {n_min}')
    if info.blk_max <= n_max:
        # we satisfy block length constraint, just return
        return info

    # we broke maximum block length constraint, so we flip
    # space and fill to have better control on fill length
    if nsp == 0 and n_max != tot_area and n_max - 1 != tot_area:
        # we get here only if nfill = 1 and fill_on_edge is True.
        # In this case there's no way to draw only one fill and abut both edges
        raise NoFillAbutEdgeError('Cannot draw only one fill abutting both edges.')
    info = _fill_symmetric_info(tot_area, nsp, n_max, inc_sp=False,
                                fill_on_edge=not fill_on_edge, cyclic=cyclic)
    if info.num_diff_sp > 0 and n_min == n_max:
        # no solution with same fill length, but we must have same fill length everywhere.
        raise NoFillChoiceError(f'Cannot draw {nfill} fill blocks with n_min = n_max = {n_min}')
    info.invert = True
    return info


def fill_symmetric_const_space(area: int, sp_max: int, n_min: int, n_max: int
                               ) -> List[Tuple[int, int]]:
    """Fill the given 1-D area given maximum space spec alone.

    The method draws the minimum number of fill blocks needed to satisfy maximum spacing spec.
    The given area is filled with the following properties:

    1. all spaces are as close to the given space as possible (differ by at most 1),
       without exceeding it.
    2. the filled area is as uniform as possible.
    3. the filled area is symmetric about the center.
    4. fill is drawn as much as possible given the above constraints.

    fill is drawn such that space blocks abuts both area boundaries.

    Parameters
    ----------
    area : int
        the 1-D area to fill.
    sp_max : int
        the maximum space.
    n_min : int
        minimum fill length.
    n_max : int
        maximum fill length

    Returns
    -------
    fill_intv : List[Tuple[int, int]]
        list of fill intervals.
    """
    if n_min > n_max:
        raise ValueError('min fill length = %d > %d = max fill length' % (n_min, n_max))

    # suppose we draw N fill blocks, then the filled area is A - (N + 1) * sp.
    # therefore, to maximize fill, with A and sp given, we need to minimize N.
    # since N = (A - sp) / (f + sp), where f is length of the fill, this tells
    # us we want to try filling with max block.
    # so we calculate the maximum number of fill blocks we'll use if we use
    # largest fill block.
    num_fill = -(-(area - sp_max) // (n_max + sp_max))
    if num_fill == 0:
        # we don't need fill; total area is less than sp_max.
        return []

    # at this point, using (num_fill - 1) max blocks is not enough, but num_fill
    # max blocks either fits perfectly or exceeds area.

    # calculate the fill block length if we use num_fill fill blocks, and sp_max
    # between blocks.
    blk_len = (area - (num_fill + 1) * sp_max) // num_fill
    if blk_len >= n_min:
        # we can draw fill using num_fill fill blocks.
        return fill_symmetric_helper(area, num_fill, sp_max, inc_sp=False,
                                     invert=False, fill_on_edge=False, cyclic=False)

    # trying to draw num_fill fill blocks with sp_max between them results in fill blocks
    # that are too small.  This means we need to reduce the space between fill blocks.
    sp_max, remainder = divmod(area - num_fill * n_min, num_fill + 1)
    # we can achieve the new sp_max using fill with length n_min or n_min + 1.
    if n_max > n_min or remainder == 0:
        # if everything divides evenly or we can use two different fill lengths,
        # then we're done.
        return fill_symmetric_helper(area, num_fill, sp_max, inc_sp=False,
                                     invert=False, fill_on_edge=False, cyclic=False)
    # If we're here, then we must use only one fill length
    # fill by inverting fill/space to try to get only one fill length
    sol, num_diff_sp = fill_symmetric_helper(area, num_fill + 1, n_max, inc_sp=False,
                                             invert=True, fill_on_edge=True, cyclic=False)
    if num_diff_sp == 0:
        # we manage to fill using only one fill length
        return sol

    # If we're here, that means num_fill + 1 is even.  So using num_fill + 2 will
    # guarantee solution.
    return fill_symmetric_helper(area, num_fill + 2, n_max, inc_sp=False,
                                 invert=True, fill_on_edge=True, cyclic=False)


def fill_symmetric_helper(tot_area: int, num_blk_tot: int, sp: int,
                          inc_sp: bool = True, invert: bool = False,
                          fill_on_edge: bool = True, cyclic: bool = False) -> List[Tuple[int, int]]:
    """Helper method for all fill symmetric methods.

    This method fills an area with given number of fill blocks such that the space between
    blocks is equal to the given space.  Other fill_symmetric methods basically transpose
    the constraints into this problem, with the proper options.

    The solution has the following properties:

    1. it is symmetric about the center.
    2. it is as uniform as possible.
    3. it uses at most 3 consecutive values of fill lengths.
    4. it uses at most 2 consecutive values of space lengths.  If inc_sp is True,
       we use sp and sp + 1.  If inc_sp is False, we use sp - 1 and sp.  In addition,
       at most two space blocks have length different than sp.

    Here are all the scenarios that affect the number of different fill/space lengths:

    1. All spaces will be equal to sp under the following condition:
       i. cyclic is False, and num_blk_tot is odd.
       ii. cyclic is True, fill_on_edge is True, and num_blk_tot is even.
       iii. cyclic is True, fill_on_edge is False, sp is even, and num_blk_tot is odd.

       In particular, this means if you must have the same space between fill blocks, you
       can change num_blk_tot by 1.
    2. The only case where at most 2 space blocks have length different than sp is
       when cyclic is True, fill_on_edge is False, sp is odd, and num_blk_tot is even.
    3. In all other cases, at most 1 space block have legnth different than sp.
    4, The only case where at most 3 fill lengths are used is when cyclic is True,
       fill_on_edge is True, and num_blk_tot is even,

    Parameters
    ----------
    tot_area : int
        the fill area length.
    num_blk_tot : int
        total number of fill blocks to use.
    sp : int
        space between blocks.  We will try our best to keep this spacing constant.
    inc_sp : bool
        If True, then we use sp + 1 if necessary.  Otherwise, we use sp - 1
        if necessary.
    invert : bool
        If True, we return space intervals instead of fill intervals.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    ans : List[(int, int)]
        list of fill or space intervals.
    """
    fill_info = _fill_symmetric_info(tot_area, num_blk_tot, sp, inc_sp=inc_sp,
                                     fill_on_edge=fill_on_edge, cyclic=cyclic)
    fill_info.invert = invert
    return fill_symmetric_interval(fill_info)


def _fill_symmetric_info(tot_area: int, num_blk_tot: int, sp: int, inc_sp: bool = True,
                         fill_on_edge: bool = True, cyclic: bool = False) -> FillInfo:
    """Calculate symmetric fill information.

    This method computes fill information without generating fill interval list.  This makes
    it fast to explore various fill settings.  See fill_symmetric_helper() to see a description
    of the fill algorithm.

    Parameters
    ----------
    tot_area : int
        the fill area length.
    num_blk_tot : int
        total number of fill blocks to use.
    sp : int
        space between blocks.  We will try our best to keep this spacing constant.
    inc_sp : bool
        If True, then we use sp + 1 if necessary.  Otherwise, we use sp - 1
        if necessary.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : FillInfo
        the fill information object.
    """
    # error checking
    if num_blk_tot < 0:
        raise ValueError(f'num_blk_tot = {num_blk_tot} < 0')

    adj_sp_sgn = 1 if inc_sp else -1
    if num_blk_tot == 0:
        # special case, no fill at all
        if sp == tot_area:
            return FillInfo(tot_area, tot_area, tot_area, tot_area, 0, 0, -1, 0, 0, 0,
                            inc_sp, False, cyclic, False)
        elif sp == tot_area - adj_sp_sgn:
            return FillInfo(tot_area, tot_area, tot_area, tot_area, 0, 0, -1, 0, 1, 0,
                            inc_sp, False, cyclic, False)
        else:
            raise EmptyRegionError(f'Cannot have empty region = {tot_area} with sp = {sp}')

    # determine the number of space blocks
    if cyclic:
        num_sp_tot = num_blk_tot
    else:
        if fill_on_edge:
            num_sp_tot = num_blk_tot - 1
        else:
            num_sp_tot = num_blk_tot + 1

    # compute total fill area
    fill_area = tot_area - num_sp_tot * sp

    # find minimum fill length
    blk_len, num_blk1 = divmod(fill_area, num_blk_tot)
    # find number of fill intervals
    if cyclic and fill_on_edge:
        # if cyclic and fill on edge, number of intervals = number of blocks + 1,
        # because the interval on the edge double counts.
        num_blk_interval = num_blk_tot + 1
    else:
        num_blk_interval = num_blk_tot

    # find space length on edge, if applicable
    num_diff_sp = 0
    sp_edge = sp
    if cyclic and not fill_on_edge and sp_edge % 2 == 1:
        # edge space must be even.  To fix, we convert space to fill
        num_diff_sp += 1
        sp_edge += adj_sp_sgn
        num_blk1 += -adj_sp_sgn
        fill_area += -adj_sp_sgn
        if num_blk1 == num_blk_tot:
            blk_len += 1
            num_blk1 = 0
        elif num_blk1 < 0:
            blk_len -= 1
            num_blk1 += num_blk_tot

    blk_m = sp_mid = -1
    # now we have num_blk_tot blocks with length blk0.  We have num_blk1 fill units
    # remaining that we need to distribute to the fill blocks
    if num_blk_interval % 2 == 0:
        # we have even number of fill intervals, so we have a space block in the middle
        sp_mid = sp
        # test condition for cyclic and fill_on_edge is different than other cases
        test_val = num_blk1 + blk_len if cyclic and fill_on_edge else num_blk1
        if test_val % 2 == 1:
            # we cannot distribute remaining fill units evenly, have to convert to space
            num_diff_sp += 1
            sp_mid += adj_sp_sgn
            num_blk1 += -adj_sp_sgn
            fill_area += -adj_sp_sgn
            if num_blk1 == num_blk_tot:
                blk_len += 1
                num_blk1 = 0
            elif num_blk1 < 0:
                blk_len -= 1
                num_blk1 += num_blk_tot
        if num_blk1 % 2 == 1:
            # the only way we get here is if cyclic and fill_on_edge is True.
            # in this case, we need to add one to fill unit to account
            # for edge fill double counting.
            num_blk1 += 1

        # get number of half fill intervals
        num_half = num_blk_interval // 2
    else:
        # we have odd number of fill intervals, so we have a fill block in the middle
        blk_m = blk_len
        if cyclic and fill_on_edge:
            # special handling for this case, because edge fill block must be even
            if blk_len % 2 == 0 and num_blk1 % 2 == 1:
                # assign one fill unit to middle block
                blk_m += 1
                num_blk1 -= 1
            elif blk_len % 2 == 1:
                # edge fill block is odd; we need odd number of fill units so we can
                # correct this.
                if num_blk1 % 2 == 0:
                    # we increment middle fill block to get odd number of fill units
                    blk_m += 1
                    num_blk1 -= 1
                    if num_blk1 < 0:
                        # we get here only if num_blk1 == 0.  This means middle blk
                        # borrow one unit from edge block.  So we set num_blk1 to
                        # num_blk_tot - 2 to make sure rest of the blocks are one
                        # larger than edge block.
                        blk_len -= 1
                        num_blk1 = num_blk_tot - 2
                    else:
                        # Add one to account for edge fill double counting.
                        num_blk1 += 1
                else:
                    # Add one to account for edge fill double counting.
                    num_blk1 += 1
        elif num_blk1 % 2 == 1:
            # assign one fill unit to middle block
            blk_m += 1
            num_blk1 -= 1

        num_half = (num_blk_interval - 1) // 2

    if blk_len <= 0:
        raise InsufficientAreaError('Insufficient area; cannot draw fill with length <= 0.')

    # now we need to distribute the fill units evenly.  We do so using cumulative modding
    num_large = num_blk1 // 2
    num_small = num_half - num_large
    if cyclic and fill_on_edge:
        # if cyclic and fill is on the edge, we need to make sure left-most block is even length
        if blk_len % 2 == 0:
            blk1, blk0 = blk_len, blk_len + 1
            num_blk1_half = num_small
        else:
            blk0, blk1 = blk_len, blk_len + 1
            num_blk1_half = num_large
    else:
        # make left-most fill interval be the most frequent fill length
        if num_large >= num_small:
            blk0, blk1 = blk_len, blk_len + 1
            num_blk1_half = num_large
        else:
            blk1, blk0 = blk_len, blk_len + 1
            num_blk1_half = num_small

    return FillInfo(tot_area, sp, sp_edge, sp_mid, blk0, blk1, blk_m, num_half, num_diff_sp,
                    num_blk1_half, inc_sp, fill_on_edge, cyclic, False)


def fill_symmetric_interval(info: FillInfo, d0: int = 0, d1: int = 0, scale: int = 1
                            ) -> List[Tuple[int, int]]:
    """Construct interval list from FillInfo object.

    Parameters
    ----------
    info : FillInfo
        the Fillinfo object.
    d0 : int
        offset of the starting coordinate.
    d1 : int
        offset of the stopping coordinate.
    scale : int
        the scale factor.
    """
    tot_area = info.tot_area
    sp_nominal = info.sp_nominal
    sp_edge = info.sp_edge
    sp_mid = info.sp_mid
    blk0 = info.blk0
    blk1 = info.blk1
    blkm = info.blkm
    num_half = info.num_half
    num_blk1_half = info.num_blk1_half
    fill_on_edge = info.fill_on_edge
    cyclic = info.cyclic
    invert = info.invert

    ans: List[Tuple[int, int]] = []
    if cyclic:
        if fill_on_edge:
            marker = -(blk1 // 2)
        else:
            marker = -(sp_edge // 2)
    else:
        marker = 0
    cur_sum = 0
    prev_sum = 1
    for fill_idx in range(num_half):
        # determine current fill length from cumulative modding result
        if cur_sum <= prev_sum:
            cur_len = blk1
        else:
            cur_len = blk0

        cur_sp = sp_edge if fill_idx == 0 else sp_nominal
        # record fill/space interval
        if invert:
            if fill_on_edge:
                ans.append((scale * (marker + cur_len) + d0,
                            scale * (marker + cur_sp + cur_len) + d1))
            else:
                ans.append((scale * marker + d0,
                            scale * (marker + cur_sp) + d1))
        else:
            if fill_on_edge:
                ans.append((scale * marker + d0,
                            scale * (marker + cur_len) + d1))
            else:
                ans.append((scale * (marker + cur_sp) + d0,
                            scale * (marker + cur_sp + cur_len) + d1))

        marker += cur_len + cur_sp
        prev_sum = cur_sum
        cur_sum = (cur_sum + num_blk1_half) % num_half

    # add middle fill or space
    if blkm >= 0:
        # fill in middle
        if invert:
            if not fill_on_edge:
                # we have one more space block before reaching middle block
                cur_sp = sp_edge if num_half == 0 else sp_nominal
                ans.append((scale * marker + d0, scale * (marker + cur_sp) + d1))
            half_len = len(ans)
        else:
            # we don't want to replicate middle fill, so get half length now
            half_len = len(ans)
            if fill_on_edge:
                ans.append((scale * marker + d0, scale * (marker + blkm) + d1))
            else:
                cur_sp = sp_edge if num_half == 0 else sp_nominal
                ans.append((scale * (marker + cur_sp) + d0, scale * (marker + cur_sp + blkm) + d1))
    else:
        # space in middle
        if invert:
            if fill_on_edge:
                # the last space we added is wrong, we need to remove
                del ans[-1]
                marker -= sp_nominal
            # we don't want to replicate middle space, so get half length now
            half_len = len(ans)
            ans.append((scale * marker + d0, scale * (marker + sp_mid) + d1))
        else:
            # don't need to do anything if we're recording blocks
            half_len = len(ans)

    # now add the second half of the list
    shift = scale * tot_area + d0 + d1
    for idx in range(half_len - 1, -1, -1):
        start, stop = ans[idx]
        ans.append((shift - stop, shift - start))

    return ans
