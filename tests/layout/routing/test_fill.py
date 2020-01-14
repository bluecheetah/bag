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

import pytest

from bag.layout.routing.fill import fill_symmetric_helper


def check_disjoint_union(outer_list, inner_list, start, stop):
    # test outer list has 1 more element than inner list
    assert len(outer_list) == len(inner_list) + 1

    sintv, eintv = outer_list[0], outer_list[-1]
    if inner_list:
        # test outer list covers more range than inner list
        assert sintv[0] <= inner_list[0][0] and eintv[1] >= inner_list[-1][1]
    # test outer list touches both boundaries
    assert sintv[0] == start and eintv[1] == stop

    # test intervals are disjoint and union is equal to given interval
    for idx in range(len(outer_list)):
        intv1 = outer_list[idx]
        # test interval is non-negative
        assert intv1[0] <= intv1[1]
        if idx < len(inner_list):
            intv2 = inner_list[idx]
            # test interval is non-negative
            assert intv2[0] <= intv2[1]
            # test interval abuts
            assert intv1[1] == intv2[0]
            assert intv2[1] == outer_list[idx + 1][0]


def check_symmetric(intv_list, start, stop):
    # test given interval list is symmetric
    flip_list = [(stop + start - b, stop + start - a) for a, b in reversed(intv_list)]
    for i1, i2 in zip(intv_list, flip_list):
        assert i1[0] == i2[0] and i1[1] == i2[1]


def check_props(fill_list, space_list, num_diff_sp1, num_diff_sp2, n, tot_intv, inc_sp, sp,
                eq_sp_parity, num_diff_sp_max, num_fill, fill_first, start, stop, n_flen_max,
                sp_edge_tweak=False):
    # check num_diff_sp is the same
    assert num_diff_sp1 == num_diff_sp2
    if n % 2 == eq_sp_parity and not sp_edge_tweak:
        # check all spaces are the same
        assert num_diff_sp1 == 0
    else:
        # check num_diff_sp is less than or equal to 1
        assert num_diff_sp1 <= num_diff_sp_max
    # test we get correct number of fill
    assert len(fill_list) == num_fill
    # test fill and space are disjoint and union is correct
    if fill_first:
        check_disjoint_union(fill_list, space_list, start, stop)
    else:
        check_disjoint_union(space_list, fill_list, start, stop)
    # check symmetry
    check_symmetric(fill_list, tot_intv[0], tot_intv[1])
    check_symmetric(space_list, tot_intv[0], tot_intv[1])
    # check fill has only two lengths, and they differ by 1
    len_list = sorted(set((b - a) for a, b in fill_list))
    assert len(len_list) <= n_flen_max
    assert (len_list[-1] - len_list[0]) <= n_flen_max - 1

    if space_list:
        # check space has only two lengths, and they differ by 1
        len_list = sorted(set((b - a) for a, b in space_list))
        assert len(len_list) <= (2 if num_diff_sp1 > 0 else 1)
        assert (len_list[-1] - len_list[0]) <= 1
        # check that space is the right values
        if len(len_list) == 1:
            # if only one space, check that it is sp + inc only if num_diff_sp > 0
            if num_diff_sp1 > 0:
                sp_correct = sp + 1 if inc_sp else sp - 1
            else:
                sp_correct = sp
            assert len_list[0] == sp_correct
        else:
            # check it has space sp and sp + inc_sp
            if inc_sp:
                assert len_list[0] == sp
            else:
                assert len_list[-1] == sp


@pytest.mark.parametrize('sp', [3, 4, 5])
@pytest.mark.parametrize('inc_sp', [True, False])
@pytest.mark.parametrize('offset', [0, 4, 7])
@pytest.mark.parametrize('foe', [True, False])
def test_fill_symmetric_non_cyclic(sp, inc_sp, offset, foe):
    # test fill symmetric for non-cyclic
    area_max = 50
    for area in range(sp + 1, area_max + 1):
        tot_intv = offset, offset + area
        for nfill in range(1, area - sp + 1):
            nsp = nfill - 1 if foe else nfill + 1
            # compute minimum possible footprint
            if nfill % 2 == 1 or inc_sp:
                # minimum possible footprint
                min_footprint = nfill * 1 + nsp * sp
            else:
                # if we have even fill and we can decrease space, then decrease middle space by 1
                min_footprint = nfill * 1 + nsp * sp - 1
            if min_footprint > area:
                # test exception when drawing with no solution
                # we have no solution when minimum possible footprint > area
                with pytest.raises(ValueError):
                    fill_symmetric_helper(area, nfill, sp, offset=offset, inc_sp=inc_sp,
                                          invert=False, fill_on_edge=foe, cyclic=False)
                with pytest.raises(ValueError):
                    fill_symmetric_helper(area, nfill, sp, offset=offset, inc_sp=inc_sp,
                                          invert=True, fill_on_edge=foe, cyclic=False)
            else:
                # get fill and space list
                fill_list, num_diff_sp1 = fill_symmetric_helper(area, nfill, sp, offset=offset,
                                                                inc_sp=inc_sp,
                                                                invert=False, fill_on_edge=foe,
                                                                cyclic=False)
                space_list, num_diff_sp2 = fill_symmetric_helper(area, nfill, sp, offset=offset,
                                                                 inc_sp=inc_sp,
                                                                 invert=True, fill_on_edge=foe,
                                                                 cyclic=False)

                check_props(fill_list, space_list, num_diff_sp1, num_diff_sp2, nfill, tot_intv,
                            inc_sp, sp,
                            1, 1, nfill, foe, tot_intv[0], tot_intv[1], 2)


@pytest.mark.parametrize('sp', [3, 4, 5])
@pytest.mark.parametrize('inc_sp', [True, False])
@pytest.mark.parametrize('offset', [0, 4, 7])
def test_fill_symmetric_cyclic_edge_fill(sp, inc_sp, offset):
    # test fill symmetric for cyclic, fill on edge
    area_max = 50
    for area in range(sp + 1, area_max + 1):
        tot_intv = offset, offset + area
        for nfill in range(1, area - sp + 1):
            nsp = nfill
            if nfill % 2 == 0 or inc_sp:
                # minimum possible footprint.  Edge fill block must be even (hence the + 1)
                min_footprint = nfill * 1 + 1 + nsp * sp
            else:
                # if we have odd fill and we can decrease space, then decrease middle space by 1
                min_footprint = nfill * 1 + 1 + nsp * sp - 1
            if min_footprint > area:
                # test exception when drawing with no solution
                # we have no solution when minimum possible footprint > area
                with pytest.raises(ValueError):
                    fill_symmetric_helper(area, nfill, sp, offset=offset, inc_sp=inc_sp,
                                          invert=False, fill_on_edge=True, cyclic=True)
                with pytest.raises(ValueError):
                    fill_symmetric_helper(area, nfill, sp, offset=offset, inc_sp=inc_sp,
                                          invert=True, fill_on_edge=True, cyclic=True)
            else:
                # get fill and space list
                fill_list, num_diff_sp1 = fill_symmetric_helper(area, nfill, sp, offset=offset,
                                                                inc_sp=inc_sp,
                                                                invert=False, fill_on_edge=True,
                                                                cyclic=True)
                space_list, num_diff_sp2 = fill_symmetric_helper(area, nfill, sp, offset=offset,
                                                                 inc_sp=inc_sp,
                                                                 invert=True, fill_on_edge=True,
                                                                 cyclic=True)
                # test boundary fills centers on edge
                sintv, eintv = fill_list[0], fill_list[-1]
                assert (sintv[1] + sintv[0]) % 2 == 0 and (eintv[1] + eintv[0]) % 2 == 0
                assert ((sintv[1] + sintv[0]) // 2 == tot_intv[0] and
                        (eintv[1] + eintv[0]) // 2 == tot_intv[1])
                # test other properties
                check_props(fill_list, space_list, num_diff_sp1, num_diff_sp2, nfill, tot_intv,
                            inc_sp, sp,
                            0, 1, nfill + 1, True, sintv[0], eintv[1], 3)


@pytest.mark.parametrize('sp', [3, 4, 5])
@pytest.mark.parametrize('inc_sp', [True, False])
@pytest.mark.parametrize('offset', [0, 4, 7])
def test_fill_symmetric_cyclic_edge_space(sp, inc_sp, offset):
    # test fill symmetric for cyclic, space on edge
    area_max = 50
    for area in range(sp + 1, area_max + 1):
        tot_intv = offset, offset + area
        for nfill in range(1, area - sp + 1):
            nsp = nfill
            adj_sp = 1 if inc_sp else -1
            sp_edge_tweak = sp % 2 == 1
            if sp_edge_tweak:
                # minimum possible footprint.  Edge space block must be even (hence the + adj_sp)
                min_footprint = nfill * 1 + nsp * sp + adj_sp
            else:
                min_footprint = nfill * 1 + nsp * sp
            if nfill % 2 == 0 and not inc_sp:
                # if we have middle space block, we can subtract one more from middle.
                min_footprint -= 1
            if min_footprint > area:
                # test exception when drawing with no solution
                # we have no solution when minimum possible footprint > area
                with pytest.raises(ValueError):
                    fill_symmetric_helper(area, nfill, sp, offset=offset, inc_sp=inc_sp,
                                          invert=False, fill_on_edge=False, cyclic=True)
                    print(area, nfill, sp, inc_sp)
                with pytest.raises(ValueError):
                    fill_symmetric_helper(area, nfill, sp, offset=offset, inc_sp=inc_sp,
                                          invert=True, fill_on_edge=False, cyclic=True)
            else:
                # get fill and space list
                fill_list, num_diff_sp1 = fill_symmetric_helper(area, nfill, sp, offset=offset,
                                                                inc_sp=inc_sp,
                                                                invert=False, fill_on_edge=False,
                                                                cyclic=True)
                space_list, num_diff_sp2 = fill_symmetric_helper(area, nfill, sp, offset=offset,
                                                                 inc_sp=inc_sp,
                                                                 invert=True, fill_on_edge=False,
                                                                 cyclic=True)

                # test boundary space centers on edge
                sintv, eintv = space_list[0], space_list[-1]
                assert (sintv[1] + sintv[0]) % 2 == 0 and (eintv[1] + eintv[0]) % 2 == 0
                assert ((sintv[1] + sintv[0]) // 2 == tot_intv[0] and
                        (eintv[1] + eintv[0]) // 2 == tot_intv[1])
                # test other properties
                check_props(fill_list, space_list, num_diff_sp1, num_diff_sp2, nfill, tot_intv,
                            inc_sp, sp,
                            1, 2, nfill, False, sintv[0], eintv[1], 2, sp_edge_tweak)
