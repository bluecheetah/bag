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

from typing import Optional, Union, List, Any, Dict
from pathlib import Path
from dataclasses import dataclass

from lark.lark import Lark
from lark.visitors import Transformer
from lark.tree import pydot__tree_to_png

from ..util.search import BinaryIterator
from ..io.file import open_file
from ..io.string import wrap_string
from ..design.netlist import read_spectre_cdl_unwrap

grammar_cdl = """
    start: headers subckts+

    headers: HEADER*
    subckts.2: ".SUBCKT" CELL PORTS+ NEWLINE instances* ".ENDS"

    instances: DEV PORTS* "/"* PORTS PARAMS* NEWLINE

    HEADER: ("*." | ".") ("_" | LETTER | NUMBER)+ (("=" | " = " | WS)? ("_" | LETTER | NUMBER)+)?
    CELL: ("_" | LETTER | NUMBER)+
    PORTS: ("_" | LETTER | NUMBER)+
    DEV: ("_" | LETTER | NUMBER | "@" | "/")+
    PARAMS: ("_" | LETTER | NUMBER)+ "=" PAR_VAL

    SC_NUM: (NUMBER | ".")+ "e-" NUMBER
    PAR_VAL: (("_" | LETTER | NUMBER)+ | SC_NUM | " * " | "*")+

    %import common.ESCAPED_STRING   -> STRING
    %import common.LETTER
    %import common.SIGNED_NUMBER    -> NUMBER
    %import common.WS
    %import common.NEWLINE
    %ignore WS
"""

grammar_scs = r"""
    start: headers subckts+

    headers: HEADER*
    subckts: "subckt" CELL PORTS+ NEWLINE instances* "ends" CELL

    instances: DEV PORTS* PARAMS* NEWLINE

    HEADER: "include " PATH | "simulator lang=spectre"
    CELL: ("_" | LETTER | NUMBER)+
    PORTS: ("_" | LETTER | NUMBER)+
    DEV: ("_" | LETTER | NUMBER | "@" | "/")+
    PARAMS: ("_" | LETTER | NUMBER)+ "=" PAR_VAL

    SC_NUM: (NUMBER | ".")+ "e-" NUMBER
    PAR_VAL: (("_" | LETTER | NUMBER)+ | SC_NUM | " * " | "*")+
    PATH: /"[\w\.\/]+"/

    %import common.ESCAPED_STRING   -> STRING
    %import common.LETTER
    %import common.SIGNED_NUMBER    -> NUMBER
    %import common.WS
    %import common.NEWLINE
    %ignore WS
"""


@dataclass
class Instance:
    inst_name: str
    ports: List[str]
    params: List[str]
    prim: Optional[str] = None
    is_transistor: bool = False
    is_BAG_prim: bool = False
    netlist_str: str = ''

    def __init__(self, items: List[Any]):
        self.params = []
        self.ports = []

        for item in items:
            if item.type == 'DEV':
                self.inst_name = item.value
            elif item.type == 'PARAMS':
                self.params.append(item.value)
            elif item.type == 'PORTS':
                if item.value.startswith('nmos4') or item.value.startswith('pmos4'):
                    self.prim = item.value
                    self.is_transistor = True
                    self.is_BAG_prim = True
                # TODO: add conditions to check for transistor in extracted netlist
                else:
                    self.ports.append(item.value)
        if self.prim is None:
            self.prim = self.ports.pop()

    def netlist(self, used_names: List[str], offset_map: Dict[str, str], scs: bool, last: bool
                ) -> str:
        if self.is_transistor:
            if self.is_BAG_prim:
                body, drain, gate, source = self.ports
            else:
                drain, gate, source, body = self.ports

            # 1. modify gate connection of device
            new_gate = f'new___{gate}_{self.inst_name.replace("/", "_").replace("@", "_")}' if \
                last else gate
            if self.is_BAG_prim:
                new_ports = [body, drain, new_gate, source]
            else:
                new_ports = [drain, new_gate, source, body]
            self.netlist_str = wrap_string([self.inst_name] + new_ports + [self.prim] + self.params)

            if last:
                # 2. add voltage source
                base_name, sep, index = self.inst_name.partition('@')
                if base_name in offset_map.keys():  # different finger of same transistor
                    offset_v = offset_map[base_name]
                else:  # create unique name
                    offset_v = f'v__{base_name.replace("/", "_")}'
                    if offset_v in used_names:  # not unique; find unique by Binary Iteration
                        bin_iter = BinaryIterator(1, None)
                        while bin_iter.has_next():
                            new_offset_v = f'{offset_v}_{bin_iter.get_next()}'
                            if new_offset_v in used_names:
                                bin_iter.up()
                            else:
                                bin_iter.save_info(new_offset_v)
                                bin_iter.down()

                        offset_v = f'{offset_v}_{bin_iter.get_last_save_info()}'
                    used_names.append(offset_v)
                    offset_map[base_name] = offset_v

                vdc_name = f'V{offset_v}{sep}{index}'
                if scs:
                    str_list = [vdc_name, new_gate, gate, 'vsource', 'type=dc', f'dc={offset_v}']
                else:
                    str_list = [vdc_name, new_gate, gate, offset_v]
                self.netlist_str += wrap_string(str_list)
        else:
            tmp_list = [self.inst_name]
            tmp_list.extend(self.ports)
            tmp_list.append(self.prim)
            tmp_list.extend(self.params)
            self.netlist_str = wrap_string(tmp_list)

        return self.netlist_str


@dataclass
class SubCKT:
    subckt_name: str
    ports: List[str]
    instances: List[Instance]
    netlist_str: str = ''
    last: bool = False

    def __init__(self, items: List[Any]):
        self.ports = []
        self.instances = []

        for item in items:
            if isinstance(item, Instance):
                self.instances.append(item)
            elif item.type == 'CELL':
                self.subckt_name = item.value
            elif item.type == 'PORTS':
                self.ports.append(item.value)

    def netlist(self, used_names: List[str], offset_map: Dict[str, str], scs: bool) -> str:
        # Construct sub-circuit netlist
        # 1. begin sub-circuit
        str_list = ['subckt' if scs else '.SUBCKT', self.subckt_name] + self.ports
        self.netlist_str = wrap_string(str_list)

        # 2. write instances
        for inst in self.instances:
            net = inst.netlist(used_names=used_names, offset_map=offset_map, scs=scs,
                               last=self.last)
            self.netlist_str += net

        # 3. end
        str_list = ['ends', self.subckt_name] if scs else ['.ENDS']
        self.netlist_str += wrap_string(str_list)
        return self.netlist_str + '\n'


@dataclass
class Header:
    netlist_str: str

    def __init__(self, items: List[Any]):
        self.netlist_str = ''
        for item in items:
            self.netlist_str += f'{item.value}\n'

    # noinspection PyUnusedLocal
    def netlist(self, used_names: List[str], offset_map: Dict[str, str], scs: bool) -> str:
        return self.netlist_str + '\n'


class CktTransformer(Transformer):
    @classmethod
    def instances(cls, items):
        return Instance(items)

    @classmethod
    def subckts(cls, items):
        return SubCKT(items)

    @classmethod
    def headers(cls, items):
        return Header(items)


def add_mismatch_offsets(netlist_in: Union[Path, str],
                         netlist_out: Optional[Union[Path, str]] = None, debug: bool = False,
                         ) -> None:
    if isinstance(netlist_in, str):
        netlist_in = Path(netlist_in)

    if netlist_in.suffix in ['.cdl', '.sp', '.spf']:
        parser = Lark(grammar_cdl, parser='lalr')
        scs = False
    elif netlist_in.suffix in ['.scs', '.net']:
        parser = Lark(grammar_scs, parser='lalr')
        scs = True
    else:
        raise ValueError(f'Unknown netlist suffix={netlist_in.suffix}. Use ".cdl" or ".scs".')

    lines = read_spectre_cdl_unwrap(netlist_in)

    lines[-1] += '\n'
    tree = parser.parse('\n'.join(lines))

    if debug:
        pydot__tree_to_png(tree, "test0.png")
    obj_list = CktTransformer().transform(tree).children
    obj_list[-1].last = True

    if netlist_out is None:
        netlist_out: Path = netlist_in.with_name(netlist_in.stem + 'out')
    if isinstance(netlist_out, str):
        netlist_out: Path = Path(netlist_out)
    full_netlist = ''
    used_names = []
    offset_map = {}
    for obj in obj_list:
        full_netlist += obj.netlist(used_names, offset_map, scs)
    for key, val in offset_map.items():
        print(f'{val}: 0.0')

    with open_file(netlist_out, 'w') as f:
        f.write(full_netlist)
