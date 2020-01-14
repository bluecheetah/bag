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

import os
import sys
import subprocess
import json
import select

import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore

from .file import write_file, open_file
from .common import to_bytes

if os.name != 'posix':
    raise Exception('bag.io.gui module current only works for POSIX systems.')


class StdinThread(QtCore.QThread):
    """A QT worker thread that reads stdin."""
    update = QtCore.pyqtSignal('QString')

    def __init__(self, parent):
        QtCore.QThread.__init__(self, parent=parent)
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                stdin, _, _ = select.select([sys.stdin], [], [], 0.05)
                if stdin:
                    cmd = sys.stdin.readline().strip()
                else:
                    cmd = None
            except:
                cmd = 'exit'

            if cmd is not None:
                self.stop = (cmd == 'exit')
                self.update.emit(cmd)


class LogWidget(QtWidgets.QFrame):
    """A Logger window widget.

    Note: due to QPlainTextEdit always adding an extra newline when calling
    appendPlainText(), we keep track of internal buffer and only print output
    one line at a time.  This may cause some message to not display immediately.
    """

    def __init__(self, parent=None):
        QtWidgets.QFrame.__init__(self, parent=parent)

        self.logger = QtWidgets.QPlainTextEdit(parent=self)
        self.logger.setReadOnly(True)
        self.logger.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.logger.setMinimumWidth(1100)
        self.buffer = ''

        self.clear_button = QtWidgets.QPushButton('Clear Log', parent=self)
        self.clear_button.clicked.connect(self.clear_log)
        self.save_button = QtWidgets.QPushButton('Save Log As...', parent=self)
        self.save_button.clicked.connect(self.save_log)

        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.addWidget(self.logger)
        self.lay.addWidget(self.clear_button)
        self.lay.addWidget(self.save_button)

    def clear_log(self):
        self.logger.setPlainText('')
        self.buffer = ''

    def save_log(self):
        root_dir = os.getcwd()
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', root_dir)
        if fname:
            write_file(fname, self.logger.toPlainText() + '\n')

    def print_file(self, file_obj):
        # this code converts all types of newlines (such as '\r\n') to '\n',
        # and make sure any ending newlines are preserved.
        for line in file_obj:
            if self.buffer:
                line = self.buffer + line
                self.buffer = ''
            if line.endswith('\n'):
                self.logger.appendPlainText(line[:-1])
            else:
                self.buffer = line


class LogViewer(QtWidgets.QWidget):
    """A Simple window to see process log in real time.."""

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # combo box label
        self.label = QtWidgets.QLabel('Log File: ', parent=self)
        # populate log selection combo box.
        self.combo_box = QtWidgets.QComboBox(parent=self)
        self.log_files = []
        self.reader = None

        self.logger = LogWidget(parent=self)

        # setup GUI
        self.setWindowTitle('BAG Simulation Log Viewer')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.label, 0, 0, alignment=QtCore.Qt.AlignRight)
        self.layout.addWidget(self.combo_box, 0, 1, alignment=QtCore.Qt.AlignLeft)
        self.layout.addWidget(self.logger, 1, 0, -1, -1)
        self.layout.setRowStretch(0, 0.0)
        self.layout.setRowStretch(1, 1.0)
        self.layout.setColumnStretch(0, 0.0)
        self.layout.setColumnStretch(1, 0.0)

        # setup file watcher
        self.cur_paths = None
        self.watcher = QtCore.QFileSystemWatcher(parent=self)
        # setup signals
        self.watcher.fileChanged.connect(self.update_logfile)
        self.combo_box.currentIndexChanged.connect(self.change_log)

        # start thread
        self.thread = StdinThread(self)
        self.thread.update.connect(self.parse_cmd)
        self.thread.start()

    def closeEvent(self, evt):
        if not self.thread.stop:
            self.thread.stop = True
            self.thread.wait()
        QtWidgets.QWidget.closeEvent(self, evt)

    @QtCore.pyqtSlot('QString')
    def parse_cmd(self, cmd):
        if cmd == 'exit':
            self.close()
        else:
            try:
                cmd = json.loads(cmd)
                if cmd[0] == 'add':
                    self.add_log(cmd[1], cmd[2])
                elif cmd[0] == 'remove':
                    self.remove_log(cmd[1])
            except:
                pass

    @QtCore.pyqtSlot('int')
    def change_log(self, new_idx):
        # print('log change called, switching to index %d' % new_idx)
        if self.cur_paths is not None:
            self.watcher.removePaths(self.cur_paths)
        self.logger.clear_log()
        if self.reader is not None:
            self.reader.close()
            self.reader = None

        if new_idx >= 0:
            fname = os.path.abspath(self.log_files[new_idx])
            dname = os.path.dirname(fname)
            self.reader = open_file(fname, 'r')
            self.logger.print_file(self.reader)
            self.cur_paths = [dname, fname]
            self.watcher.addPaths(self.cur_paths)

    @QtCore.pyqtSlot('QString')
    def update_logfile(self, fname):
        # print('filechanged called, fname = %s' % fname)
        if self.reader is not None:
            self.logger.print_file(self.reader)

    def remove_log(self, log_tag):
        idx = self.combo_box.findText(log_tag)
        if idx >= 0:
            del self.log_files[idx]
            self.combo_box.removeItem(idx)

    def add_log(self, log_tag, log_file):
        self.remove_log(log_tag)
        if os.path.isfile(log_file):
            self.log_files.append(log_file)
            self.combo_box.addItem(log_tag)


def app_start():
    app = QtWidgets.QApplication([])

    window = LogViewer()
    app.window_reference = window
    window.show()
    app.exec_()


def start_viewer():
    cmd = [sys.executable, '-m', 'bag.io.gui']
    devnull = open(os.devnull, 'w')
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=devnull,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setpgrp)
    return proc


def add_log(proc, tag, fname):
    if proc is not None:
        if proc.poll() is not None or proc.stdin.closed:
            # process finished
            return False
        cmd_str = json.dumps(['add', tag, fname]) + '\n'
        proc.stdin.write(to_bytes(cmd_str))
        proc.stdin.flush()
    return True


def remove_log(proc, tag):
    if proc is not None:
        if proc.poll() is not None or proc.stdin.closed:
            # process finished
            return False
        cmd_str = json.dumps(['remove', tag]) + '\n'
        proc.stdin.write(to_bytes(cmd_str))
        proc.stdin.flush()
    return True


def close(proc):
    if proc is not None and proc.poll() is None:
        proc.stdin.close()

if __name__ == '__main__':
    app_start()
