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

from typing import Awaitable, Any, List, Iterable

import asyncio


async def gather_err(coro_list: Iterable[Awaitable[Any]]) -> List[Any]:
    gatherer = GatherHelper()
    for coro in coro_list:
        gatherer.append(coro)

    return await gatherer.gather_err()


class GatherHelper:
    def __init__(self) -> None:
        self._tasks = []

    def __bool__(self) -> bool:
        return bool(self._tasks)

    def append(self, coro: Awaitable[Any]) -> None:
        self._tasks.append(asyncio.create_task(coro))

    async def gather_err(self) -> List[Any]:
        done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_EXCEPTION)
        if pending:
            # an error occurred, cancel and re-raise
            for task in pending:
                task.cancel()
            for task in done:
                err = task.exception()
                if err is not None:
                    raise err

        # all tasks completed
        return [task.result() for task in self._tasks]

    async def run(self) -> None:
        done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()

    def clear(self) -> None:
        self._tasks.clear()
