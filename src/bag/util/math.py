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

from __future__ import annotations

from typing import Any, Mapping

import ast
import operator
from math import trunc, ceil, floor
from numbers import Integral, Real


class HalfInt(Integral):
    """A class that represents a half integer."""

    def __init__(self, dbl_val: Any) -> None:
        if isinstance(dbl_val, Integral):
            self._val = int(dbl_val)
        else:
            raise ValueError('HalfInt internal value must be an integer.')

    @classmethod
    def convert(cls, val: Any) -> HalfInt:
        if isinstance(val, HalfInt):
            return val
        elif isinstance(val, Integral):
            return HalfInt(2 * int(val))
        elif isinstance(val, Real):
            tmp = float(2 * val)
            if tmp.is_integer():
                return HalfInt(int(tmp))
        raise ValueError('Cannot convert {} type {} to HalfInt.'.format(val, type(val)))

    @property
    def value(self) -> float:
        q, r = divmod(self._val, 2)
        return q if r == 0 else q + 0.5

    @property
    def is_integer(self) -> bool:
        return self._val % 2 == 0

    @property
    def dbl_value(self) -> int:
        return self._val

    def div2(self, round_up: bool = False) -> HalfInt:
        q, r = divmod(self._val, 2)
        return HalfInt(q + (r and round_up))

    def to_string(self) -> str:
        q, r = divmod(self._val, 2)
        if r == 0:
            return '{:d}'.format(q)
        return '{:d}.5'.format(q)

    def up(self) -> HalfInt:
        return HalfInt(self._val + 1)

    def down(self) -> HalfInt:
        return HalfInt(self._val - 1)

    def up_even(self, flag: bool) -> HalfInt:
        return HalfInt(self._val + (self._val & flag))

    def down_even(self, flag: bool) -> HalfInt:
        return HalfInt(self._val - (self._val & flag))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'HalfInt({})'.format(self._val / 2)

    def __hash__(self):
        return hash(self._val / 2)

    def __eq__(self, other):
        if isinstance(other, HalfInt):
            return self._val == other._val
        return self._val == 2 * other

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        if isinstance(other, HalfInt):
            return self._val <= other._val
        return self._val <= 2 * other

    def __lt__(self, other):
        if isinstance(other, HalfInt):
            return self._val < other._val
        return self._val < 2 * other

    def __ge__(self, other):
        return not (self < other)

    def __gt__(self, other):
        return not (self <= other)

    def __add__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(self._val + other._val)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(self._val * other._val, 2)
        if r == 0:
            return HalfInt(q)

        raise ValueError('result is not a HalfInt.')

    def __truediv__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(2 * self._val, other._val)
        if r == 0:
            return HalfInt(q)

        raise ValueError('result is not a HalfInt.')

    def __floordiv__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(2 * (self._val // other._val))

    def __mod__(self, other):
        other = HalfInt.convert(other)
        return HalfInt(self._val % other._val)

    def __divmod__(self, other):
        other = HalfInt.convert(other)
        q, r = divmod(self._val, other._val)
        return HalfInt(2 * q), HalfInt(r)

    def __pow__(self, other, modulus=None):
        other = HalfInt.convert(other)
        if self.is_integer and other.is_integer:
            return HalfInt(2 * (self._val // 2)**(other._val // 2))
        raise ValueError('result is not a HalfInt.')

    def __lshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __rshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __and__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __xor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __or__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return HalfInt.convert(other) / self

    def __rfloordiv__(self, other):
        return HalfInt.convert(other) // self

    def __rmod__(self, other):
        return HalfInt.convert(other) % self

    def __rdivmod__(self, other):
        return HalfInt.convert(other).__divmod__(self)

    def __rpow__(self, other):
        return HalfInt.convert(other)**self

    def __rlshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __rrshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __rand__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __rxor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __ror__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __ifloordiv__(self, other):
        return self // other

    def __imod__(self, other):
        return self % other

    def __ipow__(self, other):
        return self ** other

    def __ilshift__(self, other):
        raise TypeError('Cannot lshift HalfInt')

    def __irshift__(self, other):
        raise TypeError('Cannot rshift HalfInt')

    def __iand__(self, other):
        raise TypeError('Cannot and HalfInt')

    def __ixor__(self, other):
        raise TypeError('Cannot xor HalfInt')

    def __ior__(self, other):
        raise TypeError('Cannot or HalfInt')

    def __neg__(self):
        return HalfInt(-self._val)

    def __pos__(self):
        return HalfInt(self._val)

    def __abs__(self):
        return HalfInt(abs(self._val))

    def __invert__(self):
        return -self

    def __complex__(self):
        raise TypeError('Cannot cast to complex')

    def __int__(self):
        if self._val % 2 == 1:
            raise ValueError('Not an integer.')
        return self._val // 2

    def __float__(self):
        return self._val / 2

    def __index__(self):
        return int(self)

    def __round__(self, ndigits=0):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(round(self._val / 2) * 2)

    def __trunc__(self):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(trunc(self._val / 2) * 2)

    def __floor__(self):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(floor(self._val / 2) * 2)

    def __ceil__(self):
        if self.is_integer:
            return HalfInt(self._val)
        else:
            return HalfInt(ceil(self._val / 2) * 2)


# noinspection PyPep8Naming,PyMethodMayBeStatic
class Calculator(ast.NodeVisitor):
    """A simple calculator.

    Modified from:
    https://stackoverflow.com/questions/33029168/how-to-calculate-an-equation-in-a-string-python

    user mgilson said in a comment that he agrees to distribute code with Apache 2.0 license.
    """
    _OP_MAP = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Invert: operator.neg,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
    }

    def __init__(self, namespace: Mapping[str, Any]) -> None:
        super().__init__()
        self._calc_namespace = namespace

    def __getitem__(self, name: str) -> Any:
        return self._calc_namespace[name]

    @property
    def namespace(self) -> Mapping[str, Any]:
        return self._calc_namespace

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._OP_MAP[type(node.op)](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return self._OP_MAP[type(node.op)](operand)

    def visit_Num(self, node):
        return node.n

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        return self._calc_namespace[node.id]

    def eval(self, expression: str):
        tree = ast.parse(expression)
        return self.visit(tree.body[0])

    @classmethod
    def evaluate(cls, expr: str, namespace: Mapping[str, Any]):
        return cls(namespace).eval(expr)
