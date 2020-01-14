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

"""This module defines various wrapper around ZMQ sockets."""

import os
import zlib
import pprint

import zmq

from ..io.file import write_file
from ..io.common import to_bytes, fix_string
from ..io.string import read_yaml_str, to_yaml_str


class ZMQDealer(object):
    """A class that interacts with a ZMQ dealer socket.

    a dealer socket is an asynchronous socket that can issue multiple requests
    without needing to wait for an reply.  This class encapsulates the ZMQ
    socket details and provide more convenient API to use.

    Parameters
    ----------
    port : int
        the port to connect to.
    pipeline : int
        number of messages allowed in a pipeline.  Only affects file
        transfer performance.
    host : str
        the host to connect to.
    log_file : str or None
        the log file.  None to disable logging.
    """

    def __init__(self, port, pipeline=100, host='localhost', log_file=None):
        """Create a new ZMQDealer object.
        """
        context = zmq.Context.instance()
        # noinspection PyUnresolvedReferences
        self.socket = context.socket(zmq.DEALER)
        self.socket.hwm = pipeline
        self.socket.connect('tcp://%s:%d' % (host, port))
        self._log_file = log_file
        self.poller = zmq.Poller()
        # noinspection PyUnresolvedReferences
        self.poller.register(self.socket, zmq.POLLIN)

        if self._log_file is not None:
            self._log_file = os.path.abspath(self._log_file)
            # If log file directory does not exists, create it
            log_dir = os.path.dirname(self._log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # clears any existing log
            if os.path.exists(self._log_file):
                os.remove(self._log_file)

    def log_msg(self, msg):
        """Log the given message"""
        if self._log_file is not None:
            write_file(self._log_file, '%s\n' % msg, append=True)

    def log_obj(self, msg, obj):
        """Log the given object"""
        if self._log_file is not None:
            obj_str = pprint.pformat(obj)
            write_file(self._log_file, '%s\n%s\n' % (msg, obj_str), append=True)

    def close(self):
        """Close the underlying socket."""
        self.socket.close()

    def send_obj(self, obj):
        """Sends a python object using pickle serialization and zlib compression.

        Parameters
        ----------
        obj : any
            the object to send.
        """
        p = to_bytes(to_yaml_str(obj))
        z = zlib.compress(p)
        self.log_obj('sending data:', obj)
        self.socket.send(z)

    def recv_obj(self, timeout=None, enable_cancel=False):
        """Receive a python object, serialized with pickle and compressed with zlib.

        Parameters
        ----------
        timeout : int or None
            the timeout to wait in miliseconds.  If None, wait indefinitely.
        enable_cancel : bool
            If True, allows the user to press Ctrl-C to abort.  For this to work,
            the other end must know how to process the stop request dictionary.
        Returns
        -------
        obj : any
            the received object.  None if timeout reached.
        """
        try:
            events = self.poller.poll(timeout=timeout)
        except KeyboardInterrupt:
            if not enable_cancel:
                # re-raise exception if cancellation is not enabled.
                raise
            self.send_obj(dict(type='stop'))
            print('Stop signal sent, waiting for reply.  Press Ctrl-C again to force exit.')
            try:
                events = self.poller.poll(timeout=timeout)
            except KeyboardInterrupt:
                print('Force exiting.')
                return None

        if events:
            data = self.socket.recv()
            z = fix_string(zlib.decompress(data))
            obj = read_yaml_str(z)
            self.log_obj('received data:', obj)
            return obj
        else:
            self.log_msg('timeout with %d ms reached.' % timeout)
            return None

    def recv_msg(self):
        """Receive a string message.

        Returns
        -------
        msg : str
            the received object.
        """
        data = self.socket.recv()
        self.log_msg('received message:\n%s' % data)
        return data


class ZMQRouter(object):
    """A class that interacts with a ZMQ router socket.

    a router socket is an asynchronous socket that can receive multiple requests
    without needing to issue an reply.  This class encapsulates the ZMQ socket
    details and provide more convenient API to use.

    Parameters
    ----------
    port : int or None
        the port to connect to.  If None, then a random port between min_port and max_port
        will be chosen.
    min_port : int
        the minimum random port number (inclusive).
    max_port : int
        the maximum random port number (exclusive).
    pipeline : int
        number of messages allowed in a pipeline.  Only affects file
        transfer performance.
    log_file : str or None
        the log file.  None to disable logging.
    """

    def __init__(self, port=None, min_port=5000, max_port=9999, pipeline=100, log_file=None):
        """Create a new ZMQDealer object.
        """
        context = zmq.Context.instance()
        # noinspection PyUnresolvedReferences
        self.socket = context.socket(zmq.ROUTER)
        self.socket.hwm = pipeline
        if port is not None:
            self.socket.bind('tcp://*:%d' % port)
            self.port = port
        else:
            self.port = self.socket.bind_to_random_port('tcp://*', min_port=min_port, max_port=max_port)
        self.addr = None
        self._log_file = log_file

        if self._log_file is not None:
            self._log_file = os.path.abspath(self._log_file)
            # If log file directory does not exists, create it
            log_dir = os.path.dirname(self._log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # clears any existing log
            if os.path.exists(self._log_file):
                os.remove(self._log_file)

    def get_port(self):
        """Returns the port number."""
        return self.port

    def is_closed(self):
        """Returns True if this router is closed."""
        return self.socket.closed

    def close(self):
        """Close the underlying socket."""
        self.socket.close()

    def log_msg(self, msg):
        """Log the given message"""
        if self._log_file is not None:
            write_file(self._log_file, '%s\n' % msg, append=True)

    def log_obj(self, msg, obj):
        """Log the given object"""
        if self._log_file is not None:
            obj_str = pprint.pformat(obj)
            write_file(self._log_file, '%s\n%s\n' % (msg, obj_str), append=True)

    def send_msg(self, msg, addr=None):
        """Sends a string message

        Parameters
        ----------
        msg : str
            the message to send.
        addr : str or None
            the address to send the object to.  If None, send to last sender.
        """
        addr = addr or self.addr
        if addr is None:
            warn_msg = '*WARNING* No receiver address specified.  Message not sent:\n%s' % msg
            self.log_msg(warn_msg)
        else:
            self.log_msg('sending message:\n%s' % msg)
            self.socket.send_multipart([addr, msg])

    def send_obj(self, obj, addr=None):
        """Sends a python object using pickle serialization and zlib compression.

        Parameters
        ----------
        obj : any
            the object to send.
        addr : str or None
            the address to send the object to.  If None, send to last sender.
        """
        addr = addr or self.addr
        if addr is None:
            warn_msg = '*WARNING* No receiver address specified.  Message not sent:'
            self.log_obj(warn_msg, obj)
        else:
            p = to_bytes(to_yaml_str(obj))
            z = zlib.compress(p)
            self.log_obj('sending data:', obj)
            self.socket.send_multipart([addr, z])

    def poll_for_read(self, timeout):
        """Poll this socket for given timeout for read event.

        Parameters
        ----------
        timeout : int
            timeout in miliseconds.

        Returns
        -------
        status : int
            nonzero value means that this socket is ready for read.
        """
        return self.socket.poll(timeout=timeout)

    def recv_obj(self):
        """Receive a python object, serialized with pickle and compressed with zlib.

        Returns
        -------
        obj : any
            the received object.
        """
        self.addr, data = self.socket.recv_multipart()

        z = fix_string(zlib.decompress(data))
        obj = read_yaml_str(z)
        self.log_obj('received data:', obj)
        return obj

    def get_last_sender_addr(self):
        """Returns the address of the sender of last received message.

        Returns
        -------
        addr : str
            the last sender address
        """
        return self.addr
