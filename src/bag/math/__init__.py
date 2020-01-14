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

"""This package defines design template classes.
"""

from typing import Iterable, Tuple

import numpy as np
from . import interpolate

__all__ = ['lcm', 'gcd', 'interpolate', 'float_to_si_string', 'si_string_to_float',
           'get_si_prefix']


si_mag = [-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24]
si_pre = ['y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']


def get_si_prefix(num: float) -> Tuple[int, str]:
    num = float(num)
    if abs(num) < 1e-21:
        return 0, ''
    exp = np.log10(abs(num))

    pre_idx = len(si_mag) - 1
    for idx in range(len(si_mag)):
        if exp < si_mag[idx]:
            pre_idx = idx - 1
            break
    return si_mag[pre_idx], si_pre[pre_idx]


def float_to_si_string(num: float, precision: int = 6) -> str:
    """Converts the given floating point number to a string using SI prefix.

    Parameters
    ----------
    num : float
        the number to convert.
    precision : int
        number of significant digits, defaults to 6.

    Returns
    -------
    ans : str
        the string representation of the given number using SI suffix.
    """
    mag, pre = get_si_prefix(num)
    fmt = '{:.%dg}{}' % precision
    res = 10.0 ** mag
    return fmt.format(num / res, pre)


def si_string_to_float(si_str):
    """Converts the given string with SI prefix to float.

    Parameters
    ----------
    si_str : str
        the string to convert

    Returns
    -------
    ans : float
        the floating point value of the given string.
    """
    if si_str[-1] in si_pre:
        idx = si_pre.index(si_str[-1])
        return float(si_str[:-1]) * 10**si_mag[idx]
    else:
        return float(si_str)


def gcd(a, b):
    # type: (int, int) -> int
    """Compute greatest common divisor of two positive integers.

    Parameters
    ----------
    a : int
        the first number.
    b : int
        the second number.

    Returns
    -------
    ans : int
        the greatest common divisor of the two given integers.
    """
    while b:
        a, b = b, a % b
    return a


def lcm(arr, init=1):
    # type: (Iterable[int], int) -> int
    """Compute least common multiple of all numbers in the given list.

    Parameters
    ----------
    arr : Iterable[int]
        a list of integers.
    init : int
        the initial LCM.  Defaults to 1.

    Returns
    -------
    ans : int
        the least common multiple of all the given numbers.
    """
    cur_lcm = init
    for val in arr:
        cur_lcm = cur_lcm * val // gcd(cur_lcm, val)
    return cur_lcm
