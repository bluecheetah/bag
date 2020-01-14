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

from pybag.enum import LogLevel
from pybag.core import FileLogger


class LoggingBase:
    def __init__(self, log_name: str, log_file: str, log_level: LogLevel = LogLevel.DEBUG) -> None:
        self._logger = FileLogger(log_name, log_file, log_level)
        self._logger.set_level(log_level)

    @property
    def log_file(self) -> str:
        return self._logger.log_basename

    @property
    def log_level(self) -> LogLevel:
        return self._logger.level

    @property
    def logger(self) -> FileLogger:
        return self._logger

    def log(self, msg: str, level: LogLevel = LogLevel.INFO) -> None:
        self._logger.log(level, msg)

    def error(self, msg: str) -> None:
        self._logger.log(LogLevel.ERROR, msg)
        raise ValueError(msg)

    def warn(self, msg: str) -> None:
        self._logger.log(LogLevel.WARN, msg)

    def set_log_level(self, level: LogLevel) -> None:
        self._logger.set_level(level)
