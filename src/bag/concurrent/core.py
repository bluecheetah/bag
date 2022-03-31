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

"""This module define utility classes for performing concurrent operations.
"""

from typing import Optional, Sequence, Dict, Union, Tuple, Callable, Any, Awaitable, List, Iterable

import asyncio
import subprocess
import collections
import multiprocessing
from pathlib import Path
from asyncio.subprocess import Process
from concurrent.futures import CancelledError

from .util import gather_err

ProcInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], Optional[str]]
FlowInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], Optional[str],
                 Callable[[Optional[int], str], Any]]


def batch_async_task(coro_list: Iterable[Awaitable[Any]]) -> List[Any]:
    """Execute a list of coroutines or futures concurrently.

    User may press Ctrl-C to cancel all given tasks.

    Parameters
    ----------
    coro_list : Iterable[Awaitable[Any]]
        a list of coroutines or futures to run concurrently.

    Returns
    -------
    results : Optional[Tuple[Any]]
        a list of return values or raised exceptions of given tasks.
    """
    return asyncio.run(gather_err(coro_list))


class Semaphore:
    """A modified asyncio Semaphore class that gets the running loop dynamically."""
    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")

        self._value = value
        self._waiters = collections.deque()

    async def __aenter__(self) -> None:
        await self.acquire()
        return None

    async def __aexit__(self, exc_type, exc, tb):
        self.release()

    def _wake_up_next(self):
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return

    def locked(self):
        return self._value == 0

    async def acquire(self):
        loop = asyncio.get_running_loop()
        while self._value <= 0:
            fut = loop.create_future()
            self._waiters.append(fut)
            try:
                await fut
            except Exception:
                # See the similar code in Queue.get.
                fut.cancel()
                if self._value > 0 and not fut.cancelled():
                    self._wake_up_next()
                raise
        self._value -= 1
        return True

    def release(self):
        self._value += 1
        self._wake_up_next()


class SubProcessManager:
    """A class that provides methods to run multiple subprocesses in parallel using asyncio.

    Parameters
    ----------
    max_workers : Optional[int]
        number of maximum allowed subprocesses.  If None, defaults to system
        CPU count.
    cancel_timeout : float
        Number of seconds to wait for a process to terminate once SIGTERM or
        SIGKILL is issued.  Defaults to 10 seconds.
    **kwargs: Any
        Optional keyword arguments.
    """

    def __init__(self, max_workers: int = 0, cancel_timeout: float = 10.0, **kwargs: Any) -> None:
        if max_workers == 0:
            max_workers = multiprocessing.cpu_count()

        self._cancel_timeout = cancel_timeout
        self._semaphore = Semaphore(max_workers)

    async def _kill_subprocess(self, proc: Optional[Process]) -> None:
        """Helper method; send SIGTERM/SIGKILL to a subprocess.

        This method first sends SIGTERM to the subprocess.  If the process hasn't terminated
        after a given timeout, it sends SIGKILL.

        Parameter
        ---------
        proc : Optional[Process]
            the process to attempt to terminate.  If None, this method does nothing.
        """
        if proc is not None:
            if proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.shield(asyncio.wait_for(proc.wait(), self._cancel_timeout))
                    except CancelledError:
                        pass

                    if proc.returncode is None:
                        proc.kill()
                        try:
                            await asyncio.shield(
                                asyncio.wait_for(proc.wait(), self._cancel_timeout))
                        except CancelledError:
                            pass
                except ProcessLookupError:
                    pass

    async def async_new_subprocess(self,
                                   args: Union[str, Sequence[str]],
                                   log: str,
                                   env: Optional[Dict[str, str]] = None,
                                   cwd: Optional[str] = None) -> Optional[int]:
        """A coroutine which starts a subprocess.

        If this coroutine is cancelled, it will shut down the subprocess gracefully using
        SIGTERM/SIGKILL, then raise CancelledError.

        Parameters
        ----------
        args : Union[str, Sequence[str]]
            command to run, as string or sequence of strings.
        log : str
            the log file name.
        env : Optional[Dict[str, str]]
            an optional dictionary of environment variables.  None to inherit from parent.
        cwd : Optional[str]
            the working directory.  None to inherit from parent.

        Returns
        -------
        retcode : Optional[int]
            the return code of the subprocess.
        """
        if isinstance(args, str):
            args = [args]

        # get log file name, make directory if necessary
        log_path = Path(log).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if cwd is not None:
            # make sure current working directory exists
            Path(cwd).mkdir(parents=True, exist_ok=True)

        async with self._semaphore:
            proc = None
            with open(log_path, 'w') as logf:
                logf.write(f'command: {" ".join(args)}\n')
                logf.flush()
                try:
                    proc = await asyncio.create_subprocess_exec(*args, stdout=logf,
                                                                stderr=subprocess.STDOUT,
                                                                env=env, cwd=cwd)
                    retcode = await proc.wait()
                    return retcode
                except CancelledError as err:
                    await self._kill_subprocess(proc)
                    raise err

    async def async_new_subprocess_flow(self,
                                        proc_info_list: Sequence[FlowInfo]) -> Any:
        """A coroutine which runs a series of subprocesses.

        If this coroutine is cancelled, it will shut down the current subprocess gracefully using
        SIGTERM/SIGKILL, then raise CancelledError.

        Parameters
        ----------
        proc_info_list : Sequence[FlowInfo]
            a list of processes to execute in series.  Each element is a tuple of:

            args : Union[str, Sequence[str]]
                command to run, as string or list of string arguments.
            log : str
                log file name.
            env : Optional[Dict[str, str]]
                environment variable dictionary.  None to inherit from parent.
            cwd : Optional[str]
                working directory path.  None to inherit from parent.
            vfun : Sequence[Callable[[Optional[int], str], Any]]
                a function to validate if it is ok to execute the next process.  The output of the
                last function is returned.  The first argument is the return code, the second
                argument is the log file name.

        Returns
        -------
        result : Any
            the return value of the last validate function.  None if validate function
            returns False.
        """
        num_proc = len(proc_info_list)
        if num_proc == 0:
            return None

        async with self._semaphore:
            for idx, (args, log, env, cwd, vfun) in enumerate(proc_info_list):
                if isinstance(args, str):
                    args = [args]

                log_path = Path(log).resolve()
                log_path.parent.mkdir(parents=True, exist_ok=True)

                if cwd is not None:
                    # make sure current working directory exists
                    Path(cwd).mkdir(parents=True, exist_ok=True)

                proc, retcode = None, None
                with open(log_path, 'w') as logf:
                    logf.write(f'command: {" ".join(args)}\n')
                    logf.flush()
                    try:
                        proc = await asyncio.create_subprocess_exec(*args, stdout=logf,
                                                                    stderr=subprocess.STDOUT,
                                                                    env=env, cwd=cwd)
                        retcode = await proc.wait()
                    except CancelledError as err:
                        await self._kill_subprocess(proc)
                        raise err

                fun_output = vfun(retcode, str(log_path))
                if idx == num_proc - 1:
                    return fun_output
                elif not fun_output:
                    return None

    def batch_subprocess(self, proc_info_list: Sequence[ProcInfo]
                         ) -> Optional[Sequence[Union[int, Exception]]]:
        """Run all given subprocesses in parallel.

        Parameters
        ----------
        proc_info_list : Sequence[ProcInfo]
            a list of process information.  Each element is a tuple of:

            args : Union[str, Sequence[str]]
                command to run, as string or list of string arguments.
            log : str
                log file name.
            env : Optional[Dict[str, str]]
                environment variable dictionary.  None to inherit from parent.
            cwd : Optional[str]
                working directory path.  None to inherit from parent.

        Returns
        -------
        results : Optional[Sequence[Union[int, Exception]]]
            if user cancelled the subprocesses, None is returned.  Otherwise, a list of
            subprocess return codes or exceptions are returned.
        """
        num_proc = len(proc_info_list)
        if num_proc == 0:
            return []

        coro_list = [self.async_new_subprocess(args, log, env, cwd) for args, log, env, cwd in
                     proc_info_list]

        return batch_async_task(coro_list)

    def batch_subprocess_flow(self, proc_info_list: Sequence[Sequence[FlowInfo]]) -> \
            Optional[Sequence[Union[int, Exception]]]:
        """Run all given subprocesses flow in parallel.

        Parameters
        ----------
        proc_info_list : Sequence[Sequence[FlowInfo]
            a list of process flow information.  Each element is a sequence of tuples of:

            args : Union[str, Sequence[str]]
                command to run, as string or list of string arguments.
            log : str
                log file name.
            env : Optional[Dict[str, str]]
                environment variable dictionary.  None to inherit from parent.
            cwd : Optional[str]
                working directory path.  None to inherit from parent.
            vfun : Sequence[Callable[[Optional[int], str], Any]]
                a function to validate if it is ok to execute the next process.  The output of the
                last function is returned.  The first argument is the return code, the second
                argument is the log file name.

        Returns
        -------
        results : Optional[Sequence[Any]]
            if user cancelled the subprocess flows, None is returned.  Otherwise, a list of
            flow return values or exceptions are returned.
        """
        num_proc = len(proc_info_list)
        if num_proc == 0:
            return []

        coro_list = [self.async_new_subprocess_flow(flow_info) for flow_info in proc_info_list]

        return batch_async_task(coro_list)
