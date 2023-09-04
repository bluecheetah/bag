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

from enum import IntFlag


class DrawTaps(IntFlag):
    LEFT = 1
    RIGHT = 2
    NONE = ~(LEFT | RIGHT)
    BOTH = LEFT & RIGHT

    @property
    def has_left(self) -> bool:
        return self is DrawTaps.LEFT or self is DrawTaps.BOTH

    @property
    def has_right(self) -> bool:
        return self is DrawTaps.RIGHT or self is DrawTaps.BOTH
