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

"""netlist processing utilities."""

from __future__ import annotations

from typing import Union, List, Dict, Set, Callable, TextIO, Any

import abc
from pathlib import Path

from pybag.enum import DesignOutput

from ..env import get_bag_device_map
from ..util.search import get_new_name
from ..io.file import open_file
from ..io.string import wrap_string


def guess_netlist_type(netlist_in: Union[Path, str]) -> DesignOutput:
    if isinstance(netlist_in, str):
        netlist_in = Path(netlist_in)

    ext = netlist_in.suffix
    if ext == 'cdl' or ext == 'spf' or ext == 'sp':
        return DesignOutput.CDL
    elif ext == 'scs':
        return DesignOutput.SPECTRE

    with open_file(netlist_in, 'r') as f:
        for line in f:
            if 'simulator lang' in line or line.startswith('subckt'):
                return DesignOutput.SPECTRE
            if line.startswith('.subckt') or line.startswith('.SUBCKT'):
                return DesignOutput.CDL

    raise ValueError(f'Cannot guess netlist format of file {netlist_in}')


def parse_netlist(netlist_in: Union[Path, str], netlist_type: DesignOutput) -> Netlist:
    if isinstance(netlist_in, str):
        netlist_in = Path(netlist_in)

    lines = _read_lines(netlist_in)

    if netlist_type is DesignOutput.CDL:
        return ParserCDL.parse_netlist(lines)
    elif netlist_type is DesignOutput.SPECTRE:
        return ParserSpectre.parse_netlist(lines)
    else:
        raise ValueError(f'Unsupported netlist format: {netlist_type}')


def add_mismatch_offsets(netlist_in: Union[Path, str], netlist_out: Union[Path, str],
                         netlist_type: DesignOutput) -> Dict[str, Any]:
    return add_internal_sources(netlist_in, netlist_out, netlist_type, ['g'])


def add_internal_sources(netlist_in: Union[Path, str], netlist_out: Union[Path, str],
                         netlist_type: DesignOutput, ports: List[str]) -> Dict[str, Any]:
    netlist = parse_netlist(netlist_in, netlist_type)

    if isinstance(netlist_out, str):
        netlist_out: Path = Path(netlist_out)

    mos_map = get_bag_device_map('mos')
    bag_mos = set()
    pdk_mos = set()
    for k, v, in mos_map:
        bag_mos.add(k)
        pdk_mos.add(v)

    used_names = set()
    offset_map = {}
    with open_file(netlist_out, 'w') as f:
        netlist.netlist_with_offset(f, used_names, offset_map, netlist_type, ports, bag_mos, pdk_mos)
    return offset_map


class NetlistNode(abc.ABC):
    @abc.abstractmethod
    def netlist(self, stream: TextIO, netlist_type: DesignOutput) -> None:
        pass

    @abc.abstractmethod
    def netlist_with_offset(self, stream: TextIO, used_names: Set[str], offset_map: Dict[str, str],
                            netlist_type: DesignOutput, ports: List[str], bag_mos: Set[str], pdk_mos: Set[str]
                            ) -> None:
        pass


class Netlist(NetlistNode):
    def __init__(self, header: Header, subckts: List[Subcircuit]) -> None:
        self._header = header
        self._subckts = subckts
        self._used_names = set()
        for ckt in self._subckts:
            name = ckt.name
            if name in self._used_names:
                raise ValueError(f'Found duplicated subcircuit name: {name}')
            self._used_names.add(name)

    @property
    def used_names(self) -> Set[str]:
        return self._used_names

    def netlist(self, stream: TextIO, netlist_type: DesignOutput) -> None:
        self._header.netlist(stream, netlist_type)
        for ckt in self._subckts:
            ckt.netlist(stream, netlist_type)

    def netlist_with_offset(self, stream: TextIO, used_names: Set[str], offset_map: Dict[str, str],
                            netlist_type: DesignOutput, ports: List[str], bag_mos: Set[str], pdk_mos: Set[str]
                            ) -> None:
        self._header.netlist(stream, netlist_type)
        for idx in range(0, len(self._subckts) - 1):
            self._subckts[idx].netlist(stream, netlist_type)
        self._subckts[-1].netlist_with_offset(stream, used_names, offset_map, netlist_type,
                                              ports, bag_mos, pdk_mos)


class Header(NetlistNode):
    def __init__(self, lines: List[str]) -> None:
        self.lines = lines

    def netlist(self, stream: TextIO, netlist_type: DesignOutput) -> None:
        for line in self.lines:
            stream.write(line)
            stream.write('\n')

    def netlist_with_offset(self, stream: TextIO, used_names: Set[str], offset_map: Dict[str, str],
                            netlist_type: DesignOutput, ports: List[str], bag_mos: Set[str], pdk_mos: Set[str]
                            ) -> None:
        self.netlist(stream, netlist_type)


class Instance(NetlistNode):
    def __init__(self, inst_name: str, cell_name: str, ports: List[str], params: List[str]) -> None:
        self._inst_name = inst_name
        self._cell_name = cell_name
        self._ports = ports
        self._params = params

    def netlist(self, stream: TextIO, netlist_type: DesignOutput) -> None:
        if netlist_type is DesignOutput.CDL:
            if self._cell_name:
                # this is not a primitive instance
                tmp_list = [self._inst_name]
                tmp_list.extend(self._ports)
                tmp_list.append('/')
                tmp_list.append(self._cell_name)
                tmp_list.extend(self._params)
            else:
                # this is a primitive list
                tmp_list = [self._inst_name]
                tmp_list.extend(self._ports)
                tmp_list.extend(self._params)

            stream.write(wrap_string(tmp_list))
        elif netlist_type is DesignOutput.SPECTRE:
            tmp_list = [self._inst_name]
            tmp_list.extend(self._ports)
            tmp_list.append(self._cell_name)
            tmp_list.extend(self._params)
            stream.write(wrap_string(tmp_list))
        else:
            raise ValueError(f'unsupported netlist type: {netlist_type}')

    def netlist_with_offset(self, stream: TextIO, used_names: Set[str], offset_map: Dict[str, str],
                            netlist_type: DesignOutput, ports: List[str], bag_mos: Set[str], pdk_mos: Set[str]
                            ) -> None:
        if any(self._cell_name.startswith(mos_name) for mos_name in bag_mos):
            bag_mos = True
            body, drain, gate, source = self._ports
        elif any(self._cell_name.startswith(mos_name) for mos_name in pdk_mos):
            bag_mos = False
            drain, gate, source, body = self._ports
        else:
            return self.netlist(stream, netlist_type)

        old_mapping = dict(b=body, d=drain, g=gate, s=source)
        new_mapping = old_mapping.copy()

        # 1. modify gate connection of device
        for port in ports:
            new_mapping[port] = f'new___{old_mapping[port]}_{self._inst_name.replace("/", "_").replace("@", "_")}'

        if bag_mos:
            tmp_list = [self._inst_name, new_mapping['b'], new_mapping['d'], new_mapping['g'], new_mapping['s']]
        else:
            tmp_list = [self._inst_name, new_mapping['d'], new_mapping['g'], new_mapping['s'], new_mapping['b']]

        tmp_list.append(self._cell_name)
        tmp_list.extend(self._params)
        stream.write(wrap_string(tmp_list))

        # 2. add voltage source
        base_name, sep, index = self._inst_name.partition('@')
        for port in ports:
            base_name_port = f'{base_name}_{port}'
            if base_name_port in offset_map.keys():  # different finger of same transistor
                offset_v_port = offset_map[base_name_port]
            else:  # create unique name
                offset_v_port = get_new_name(f'v__{base_name_port.replace("/", "_")}', used_names)
                used_names.add(offset_v_port)
                offset_map[base_name_port] = offset_v_port

            vdc_name = f'V{offset_v_port}{sep}{index}'

            if netlist_type is DesignOutput.SPECTRE:
                tmp_list = [vdc_name, new_mapping[port], old_mapping[port], 'vsource', 'type=dc', f'dc={offset_v_port}']
            else:
                tmp_list = [vdc_name, new_mapping[port], old_mapping[port], offset_v_port]

            stream.write(wrap_string(tmp_list))


class Subcircuit(NetlistNode):
    def __init__(self, name: str, ports: List[str], items: List[Union[str, Instance]]) -> None:
        self._name = name
        self._ports = ports
        self._items = items

    @property
    def name(self) -> str:
        return self._name

    def netlist(self, stream: TextIO, netlist_type: DesignOutput) -> None:
        def inst_fun(inst: NetlistNode) -> None:
            inst.netlist(stream, netlist_type)

        return self._netlist_helper(stream, netlist_type, inst_fun)

    def netlist_with_offset(self, stream: TextIO, used_names: Set[str], offset_map: Dict[str, str],
                            netlist_type: DesignOutput, ports: List[str], bag_mos: Set[str], pdk_mos: Set[str]
                            ) -> None:
        def inst_fun(inst: NetlistNode) -> None:
            inst.netlist_with_offset(stream, used_names, offset_map, netlist_type, ports, bag_mos, pdk_mos)

        return self._netlist_helper(stream, netlist_type, inst_fun)

    def _netlist_helper(self, stream, netlist_type: DesignOutput, fun: Callable[[NetlistNode], None]
                        ) -> None:
        # header
        if netlist_type is DesignOutput.SPECTRE:
            tmp_list = ['subckt', self._name]
        else:
            tmp_list = ['.SUBCKT', self._name]
        tmp_list.extend(self._ports)
        stream.write(wrap_string(tmp_list))
        for item in self._items:
            if isinstance(item, str):
                stream.write(item)
                stream.write('\n')
            else:
                fun(item)

        # 3. end
        if netlist_type is DesignOutput.SPECTRE:
            stream.write(wrap_string(['ends', self._name]))
        else:
            stream.write('.ENDS\n')

        stream.write('\n')


class Parser(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def is_subckt_start(cls, line: str) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def is_subckt_end(cls, line: str) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def is_comment(cls, line) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def parse_instance(cls, tokens: List[str]) -> Instance:
        pass

    @classmethod
    def parse_netlist(cls, lines: List[str]) -> Netlist:
        subckts = []
        idx = 0
        num_lines = len(lines)
        while idx < num_lines and not cls.is_subckt_start(lines[idx]):
            idx += 1

        header = cls.parse_header(lines, 0, idx)
        while idx < num_lines:
            start_idx = idx
            while idx < num_lines and not cls.is_subckt_end(lines[idx]):
                idx += 1
            if idx == num_lines:
                raise ValueError('Did not find subcircuit end.')
            idx += 1
            subckts.append(cls.parse_subcircuit(lines, start_idx, idx))

            while idx < num_lines and not cls.is_subckt_start(lines[idx]):
                idx += 1

        return Netlist(header, subckts)

    @classmethod
    def parse_header(cls, lines: List[str], start: int, stop: int) -> Header:
        return Header(lines[start:stop])

    @classmethod
    def parse_subcircuit(cls, lines: List[str], start: int, stop: int) -> Subcircuit:
        header_tokens = lines[start].split()
        cell_name = header_tokens[1]
        ports = header_tokens[2:]
        items: List[Union[str, Instance]] = []
        # skip last line because it is end subcircuit line
        for idx in range(start + 1, stop - 1):
            cur_line = lines[idx]
            if cls.is_comment(cur_line):
                items.append(cur_line)
            else:
                tokens = cur_line.split()
                if tokens:
                    if tokens[0] == 'parameters':
                        items.append(' '.join(tokens))
                    else:
                        items.append(cls.parse_instance(tokens))

        return Subcircuit(cell_name, ports, items)


class ParserCDL(Parser):
    prim_prefix = {'R', 'C', 'V', 'I'}

    @classmethod
    def is_subckt_start(cls, line: str) -> bool:
        return line.startswith('.SUBCKT ') or line.startswith('.subckt ')

    @classmethod
    def is_subckt_end(cls, line: str) -> bool:
        return line.startswith('.ENDS') or line.startswith('.ends')

    @classmethod
    def is_comment(cls, line) -> bool:
        return line.startswith('*')

    @classmethod
    def parse_instance(cls, tokens: List[str]) -> Instance:
        inst_name = tokens[0]
        if inst_name[0] in cls.prim_prefix:
            # this is a resistor
            ports = tokens[1:3]
            params = tokens[3:]
            return Instance(inst_name, '', ports, params)

        ports = None
        end_idx = 0
        for end_idx in range(1, len(tokens)):
            if tokens[end_idx] == '/':
                # detect separator
                ports = tokens[1:end_idx]
                end_idx += 1
                break
            elif '=' in tokens[end_idx]:
                # detect parameters
                ports = tokens[1:end_idx - 1]
                end_idx -= 1
                break

        if ports is None:
            # did not hit separator or parameters, assume last index is cell name
            ports = tokens[1:end_idx]
            end_idx -= 1

        cell_name = tokens[end_idx]
        params = tokens[end_idx+1:]
        return Instance(inst_name, cell_name, ports, params)


class ParserSpectre(Parser):
    @classmethod
    def is_subckt_start(cls, line: str) -> bool:
        return line.startswith('subckt ')

    @classmethod
    def is_subckt_end(cls, line: str) -> bool:
        return line.startswith('ends')

    @classmethod
    def is_comment(cls, line) -> bool:
        return line.startswith('*') or line.startswith('//')

    @classmethod
    def parse_instance(cls, tokens: List[str]) -> Instance:
        inst_name = tokens[0]
        ports = None
        end_idx = 0
        for end_idx in range(1, len(tokens)):
            if '=' in tokens[end_idx]:
                # detect parameters
                ports = tokens[1:end_idx - 1]
                end_idx -= 1
                break

        if ports is None:
            # did not hit parameters, assume last index is cell name
            ports = tokens[1:end_idx]

        cell_name = tokens[end_idx]
        params = tokens[end_idx+1:]
        return Instance(inst_name, cell_name, ports, params)


def _read_lines(netlist: Path) -> List[str]:
    """Reads the given Spectre or CDL netlist.

    This function process line continuation and comments so we don't have to worry about it.
    """
    lines = []
    with open_file(netlist, 'r') as f:
        for line in f:
            if line.startswith('+'):
                lines[-1] += line[1:-1]
            else:
                lines.append(line[:-1])

    return lines
