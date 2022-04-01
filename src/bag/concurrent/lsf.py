# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
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

"""This module define utility classes for launching subprocesses via IBM Load Sharing Facility (LSF).
"""

from typing import Optional, Sequence, Dict, Union, Any, List

import asyncio
import subprocess
from pathlib import Path
from concurrent.futures import CancelledError

from .core import SubProcessManager, FlowInfo


class LSFSubProcessManager(SubProcessManager):
    """A class that provides methods to run multiple subprocesses in parallel using IBM Load Sharing Facility (LSF).

    Parameters
    ----------
    queue: str
        name of LSF queue to use for submitting jobs.
    options: Optional[List[str]]
        list of additional command line arguments to pass into the bsub command.
    max_workers : Optional[int]
        number of maximum allowed subprocesses.  If None, defaults to system
        CPU count.
    cancel_timeout : float
        Number of seconds to wait for a process to terminate once SIGTERM or
        SIGKILL is issued.  Defaults to 10 seconds.
    """

    def __init__(self, queue: str, options: Optional[List[str]] = None, max_workers: int = 0,
                 cancel_timeout: float = 10.0) -> None:
        self._queue = queue
        self._options = options or []
        super().__init__(max_workers, cancel_timeout)

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

        main_cmd = " ".join(args)

        cmd_args = ['bsub', '-K', '-q', self._queue, '-o', str(log_path)] + self._options + [f'"{main_cmd}"']
        cmd = " ".join(cmd_args)

        async with self._semaphore:
            proc = None
            with open(log_path, 'w') as logf:
                logf.write(f'command: {cmd}\n')
                logf.flush()
                try:
                    # shell must be used to preserve paths and environment variables on compute host
                    proc = await asyncio.create_subprocess_shell(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                                                 cwd=cwd)
                    retcode = await proc.wait()
                    return retcode
                except CancelledError as err:
                    await self._kill_subprocess(proc)
                    raise err

    async def async_new_subprocess_flow(self, proc_info_list: Sequence[FlowInfo]) -> Any:
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

                main_cmd = " ".join(args)

                cmd_args = ['bsub', '-K', '-q', self._queue, '-o', str(log_path)] + self._options + [f'"{main_cmd}"']
                cmd = " ".join(cmd_args)

                proc, retcode = None, None
                with open(log_path, 'w') as logf:
                    logf.write(f'command: {cmd}\n')
                    logf.flush()
                    try:
                        # shell must be used to preserve paths and environment variables on compute host
                        proc = await asyncio.create_subprocess_shell(cmd, stdout=logf, stderr=subprocess.STDOUT,
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

    # Some utility functions that are currently unused; could be useful in the future for job scheduling
    @staticmethod
    def get_njobs_per_user(queue):
        res = subprocess.Popen(f'bqueues {queue}', shell=True, stdout=subprocess.PIPE).communicate()[0]
        header, info = map(lambda s: s.decode('utf-8').split(), res.splitlines())
        return int(info[header.index('JL/U')])

    @staticmethod
    def get_njobs_running(queue):
        res = subprocess.Popen(f'bjobs -r -q {queue} | wc -l', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE).communicate()[0]
        num_lines = int(res.decode('utf-8').strip())
        # If there is a job running, there will be a header row, resulting in njobs + 1 lines in the prompt.
        # Otherwise, there will be 0 lines
        return max(num_lines - 1, 0)
