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

"""Generate setup yaml files for various netlist outputs

Please run this script through the generate_netlist_config.sh shell script, which will setup
the PYTHONPATH correctly.
"""

from typing import Dict, Any, Tuple, List, Union

import copy
import argparse
from pathlib import Path

from jinja2 import Environment, DictLoader

from pybag.enum import DesignOutput

from bag.io.file import read_yaml, write_yaml, open_file

netlist_map_default = {
    'basic': {
        'cds_thru': {
            'lib_name': 'basic',
            'cell_name': 'cds_thru',
            'in_terms': [],
            'io_terms': ['src', 'dst'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {},
            'ignore': False,
        },
        'noConn': {
            'lib_name': 'basic',
            'cell_name': 'noConn',
            'in_terms': [],
            'io_terms': ['noConn'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {},
            'ignore': True,
        },
    },
    'analogLib': {
        'cap': {
            'lib_name': 'analogLib',
            'cell_name': 'cap',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'c': [3, ''],
                'l': [3, ''],
                'm': [3, ''],
                'w': [3, ''],
            }
        },
        'cccs': {
            'lib_name': 'analogLib',
            'cell_name': 'cccs',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'fgain': [3, '1.0'],
                'maxm': [3, ''],
                'minm': [3, ''],
                'vref': [3, ''],
            }
        },
        'ccvs': {
            'lib_name': 'analogLib',
            'cell_name': 'ccvs',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'hgain': [3, '1.0'],
                'maxm': [3, ''],
                'minm': [3, ''],
                'vref': [3, ''],
            }
        },
        'dcblock': {
            'lib_name': 'analogLib',
            'cell_name': 'dcblock',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'c': [3, '1u'],
            }
        },
        'dcfeed': {
            'lib_name': 'analogLib',
            'cell_name': 'dcfeed',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'l': [3, '1u'],
            }
        },
        'idc': {
            'lib_name': 'analogLib',
            'cell_name': 'idc',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'acm': [3, ''],
                'acp': [3, ''],
                'idc': [3, ''],
                'pacm': [3, ''],
                'pacp': [3, ''],
                'srcType': [3, 'dc'],
                'xfm': [3, ''],
            }
        },
        'ideal_balun': {
            'lib_name': 'analogLib',
            'cell_name': 'ideal_balun',
            'in_terms': [],
            'io_terms': ['d', 'c', 'p', 'n'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {}
        },
        'ind': {
            'lib_name': 'analogLib',
            'cell_name': 'ind',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'l': [3, ''],
                'm': [3, ''],
                'r': [3, ''],
            }
        },
        'iprobe': {
            'lib_name': 'analogLib',
            'cell_name': 'iprobe',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {}
        },
        'ipwlf': {
            'lib_name': 'analogLib',
            'cell_name': 'ipwlf',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'fileName': [3, ''],
                'srcType': [3, 'pwl'],
            }
        },
        'ipulse': {
            'lib_name': 'analogLib',
            'cell_name': 'ipulse',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'i1': [3, ''],
                'i2': [3, ''],
                'idc': [3, ''],
                'per': [3, ''],
                'pw': [3, ''],
                'srcType': [3, 'pulse'],
                'td': [3, ''],
            }
        },
        'isin': {
            'lib_name': 'analogLib',
            'cell_name': 'isin',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'freq': [3, ''],
                'ia': [3, ''],
                'idc': [3, ''],
                'srcType': [3, 'sine'],
            }
        },
        'gnd': {
            'lib_name': 'analogLib',
            'cell_name': 'gnd',
            'in_terms': [],
            'io_terms': ['gnd!'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {},
            'ignore': True,
        },
        'mind': {
            'lib_name': 'analogLib',
            'cell_name': 'mind',
            'in_terms': [],
            'io_terms': [],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'ind1': [3, ''],
                'ind2': [3, ''],
                'k': [3, '0'],
            },
        },
        'n1port': {
            'lib_name': 'analogLib',
            'cell_name': 'n1port',
            'in_terms': [],
            'io_terms': ['t1', 'b1'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'n2port': {
            'lib_name': 'analogLib',
            'cell_name': 'n2port',
            'in_terms': [],
            'io_terms': ['t1', 'b1', 't2', 'b2'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'n3port': {
            'lib_name': 'analogLib',
            'cell_name': 'n3port',
            'in_terms': [],
            'io_terms': ['t1', 'b1', 't2', 'b2', 't3', 'b3'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'n4port': {
            'lib_name': 'analogLib',
            'cell_name': 'n4port',
            'in_terms': [],
            'io_terms': ['t1', 'b1', 't2', 'b2', 't3', 'b3', 't4', 'b4'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'n6port': {
            'lib_name': 'analogLib',
            'cell_name': 'n6port',
            'in_terms': [],
            'io_terms': ['t1', 'b1', 't2', 'b2', 't3', 'b3', 't4', 'b4', 't5', 'b5', 't6', 'b6'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'n8port': {
            'lib_name': 'analogLib',
            'cell_name': 'n8port',
            'in_terms': [],
            'io_terms': ['t1', 'b1', 't2', 'b2', 't3', 'b3', 't4', 'b4', 't5', 'b5', 't6', 'b6', 't7', 'b7',
                         't8', 'b8'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'n12port': {
            'lib_name': 'analogLib',
            'cell_name': 'n12port',
            'in_terms': [],
            'io_terms': ['t1', 'b1', 't2', 'b2', 't3', 'b3', 't4', 'b4', 't5', 'b5', 't6', 'b6', 't7', 'b7',
                         't8', 'b8', 't9', 'b9', 't10', 'b10', 't11', 'b11', 't12', 'b12'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'dataFile': [3, ''],
                'interp': [3, 'linear'],
                'thermalnoise': [3, 'yes'],
            },
        },
        'port': {
            'lib_name': 'analogLib',
            'cell_name': 'port',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'num': [3, ''],
                'r': [3, ''],
                'srcType': [3, 'sine'],
            }
        },
        'res': {
            'lib_name': 'analogLib',
            'cell_name': 'res',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'l': [3, ''],
                'm': [3, ''],
                'r': [3, ''],
                'w': [3, ''],
            }
        },
        'switch': {
            'lib_name': 'analogLib',
            'cell_name': 'switch',
            'in_terms': [],
            'io_terms': ['N+', 'N-', 'NC+', 'NC-'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'rc': [3, ''],
                'ro': [3, ''],
                'vt1': [3, ''],
                'vt2': [3, ''],
            }
        },
        'vccs': {
            'lib_name': 'analogLib',
            'cell_name': 'vccs',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS', 'NC+', 'NC-'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'ggain': [3, '1.0'],
                'maxm': [3, ''],
                'minm': [3, ''],
            }
        },
        'vcvs': {
            'lib_name': 'analogLib',
            'cell_name': 'vcvs',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS', 'NC+', 'NC-'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'egain': [3, '1.0'],
                'maxm': [3, ''],
                'minm': [3, ''],
            }
        },
        'vdc': {
            'lib_name': 'analogLib',
            'cell_name': 'vdc',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'acm': [3, ''],
                'acp': [3, ''],
                'pacm': [3, ''],
                'pacp': [3, ''],
                'srcType': [3, 'dc'],
                'vdc': [3, ''],
                'xfm': [3, ''],
            }
        },
        'vpulse': {
            'lib_name': 'analogLib',
            'cell_name': 'vpulse',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'per': [3, ''],
                'pw': [3, ''],
                'srcType': [3, 'pulse'],
                'td': [3, ''],
                'v1': [3, ''],
                'v2': [3, ''],
                'vdc': [3, ''],
            }
        },
        'vpwlf': {
            'lib_name': 'analogLib',
            'cell_name': 'vpwlf',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'fileName': [3, ''],
                'srcType': [3, 'pwl'],
            }
        },
        'vsin': {
            'lib_name': 'analogLib',
            'cell_name': 'vsin',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'freq': [3, ''],
                'srcType': [3, 'sine'],
                'va': [3, ''],
                'vdc': [3, ''],
            }
        },
    },
    'ahdlLib': {
        'comparator': {
            'lib_name': 'ahdlLib',
            'cell_name': 'comparator',
            'in_terms': ['sigin', 'sigref'],
            'io_terms': [],
            'is_prim': True,
            'nets': [],
            'out_terms': ['sigout'],
            'props': {
                'sigout_high': [3, ''],
                'sigout_low': [3, ''],
                'sigin_offset': [3, ''],
                'comp_slope': [3, ''],
            },
            'va': '${CDSHOME}/tools/dfII/samples/artist/ahdlLib/comparator/veriloga/veriloga.va',
        },
        'rand_bit_stream': {
            'lib_name': 'ahdlLib',
            'cell_name': 'rand_bit_stream',
            'in_terms': [],
            'io_terms': [],
            'is_prim': True,
            'nets': [],
            'out_terms': ['vout'],
            'props': {
                'tperiod': [3, ''],
                'seed': [3, ''],
                'vlogic_high': [3, ''],
                'vlogic_low': [3, ''],
                'tdel': [3, ''],
                'trise': [3, ''],
                'tfall': [3, ''],
            },
            'va': '${CDSHOME}/tools/dfII/samples/artist/ahdlLib/rand_bit_stream/veriloga/veriloga.va',
        },
    },
}

mos_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['B', 'D', 'G', 'S'],
    'nets': [],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
        'nf': [3, ''],
    },
}

mos3_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['D', 'G', 'S'],
    'nets': [],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
        'nf': [3, ''],
    },
}

dio_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['MINUS', 'PLUS'],
    'nets': [],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
    },
}

res_metal_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['MINUS', 'PLUS'],
    'nets': [],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
    },
}

res_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['BULK', 'MINUS', 'PLUS'],
    'nets': [],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
    },
}

mim_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['BOT', 'TOP'],
    'nets': [],
    'is_prim': True,
    'props': {
        'unit_width': [3, ''],
        'unit_height': [3, ''],
        'num_rows': [3, ''],
        'num_cols': [3, ''],
    },
}

mos_cdl_fmt = """.SUBCKT {{ cell_name }} B D G S
*.PININFO B:B D:B G:B S:B
{{ prefix }}M0 D G S B {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

mos3_cdl_fmt = """.SUBCKT {{ cell_name }} D G S
*.PININFO D:B G:B S:B
{{ prefix }}M0 D G S {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

dio_cdl_fmt = """.SUBCKT {{ cell_name }}{% if ports|length == 3 %} GUARD_RING{% endif %} MINUS PLUS
*.PININFO{% if ports|length == 3 %} GUARD_RING:B{% endif %} MINUS:B PLUS:B
{{ prefix }}D0 {{ ports[0] }} {{ ports[1] }}{% if ports|length == 3 %} {{ ports[2] }}{% endif %} {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

dio_cdl_fmt_static = """.SUBCKT {{ cell_name }}{% if ports|length == 3 %} GUARD_RING{% endif %} MINUS PLUS
*.PININFO{% if ports|length == 3 %} GUARD_RING:B{% endif %} MINUS:B PLUS:B
{{ prefix }}D0 {{ ports[0] }} {{ ports[1] }}{% if ports|length == 3 %} {{ ports[2] }}{% endif %} {{ model_name }}
.ENDS
"""

res_metal_cdl_fmt = """.SUBCKT {{ cell_name }} MINUS PLUS
*.PININFO MINUS:B PLUS:B
{{ prefix }}R0 PLUS MINUS {{ model_name }} {% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

res_cdl_fmt = """.SUBCKT {{ cell_name }}{% if num_ports == 3 %} BULK{% endif %} MINUS PLUS
*.PININFO{% if num_ports == 3 %} BULK:B{% endif %} MINUS:B PLUS:B
{{ prefix }}R0 PLUS MINUS{% if num_ports == 3 %} BULK{% endif %} {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

mim_cdl_fmt = """.SUBCKT {{ cell_name }} BOT TOP
*.PININFO BOT:B TOP:B
{{ prefix }}C0 TOP BOT {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

mos_spectre_fmt = """subckt {{ cell_name }} B D G S
parameters l w nf
{{ prefix }}M0 D G S B {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

mos3_spectre_fmt = """subckt {{ cell_name }} D G S
parameters l w nf
{{ prefix }}M0 D G S {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

dio_spectre_fmt = """subckt {{ cell_name }}{% if ports|length == 3 %} GUARD_RING{% endif %} MINUS PLUS
parameters l w
{{ prefix }}D0 {{ ports[0] }} {{ ports[1] }}{% if ports|length == 3 %} {{ ports[2] }}{% endif %} {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

dio_spectre_fmt_static = """subckt {{ cell_name }}{% if ports|length == 3 %} GUARD_RING{% endif %} MINUS PLUS
{{ prefix }}D0 {{ ports[0] }} {{ ports[1] }}{% if ports|length == 3 %} {{ ports[2] }}{% endif %} {{ model_name }}
ends {{ cell_name }}
"""

res_metal_spectre_fmt = """subckt {{ cell_name }} MINUS PLUS
parameters l w
{{ prefix }}R0 PLUS MINUS {{ model_name }} {% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

res_spectre_fmt = """subckt {{ cell_name }}{% if num_ports == 3 %} BULK{% endif %} MINUS PLUS
parameters l w
{{ prefix }}R0 PLUS MINUS{% if num_ports == 3 %} BULK{% endif %} {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

mim_spectre_fmt = """subckt {{ cell_name }} BOT TOP
parameters unit_width unit_height num_rows num_cols
{{ prefix }}C0 TOP BOT {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

mos_verilog_fmt = """module {{ cell_name }}(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule
"""

mos3_verilog_fmt = """module {{ cell_name }}(
    inout D,
    inout G,
    inout S
);
endmodule
"""

scs_ideal_balun = """subckt ideal_balun d c p n
    K0 d 0 p c transformer n1=2
    K1 d 0 c n transformer n1=2
ends ideal_balun
"""

supported_formats = {
    DesignOutput.CDL: {
        'fname': 'bag_prim.cdl',
        'mos': 'mos_cdl',
        'mos3': 'mos3_cdl',
        'diode': 'diode_cdl',
        'diode_static': 'diode_cdl_static',
        'res_metal': 'res_metal_cdl',
        'res': 'res_cdl',
        'mim': 'mim_cdl',
    },
    DesignOutput.SPECTRE: {
        'fname': 'bag_prim.scs',
        'mos': 'mos_scs',
        'mos3': 'mos3_scs',
        'diode': 'diode_scs',
        'diode_static': 'diode_scs_static',
        'res_metal': 'res_metal_scs',
        'res': 'res_scs',
        'mim': 'mim_scs',
    },
    DesignOutput.VERILOG: {
        'fname': 'bag_prim.v',
        'mos': '',
        'mos3': '',
        'diode': '',
        'diode_static': '',
        'res_metal': '',
        'res': '',
        'mim': '',
    },
    DesignOutput.SYSVERILOG: {
        'fname': 'bag_prim.sv',
        'mos': '',
        'mos3': '',
        'diode': '',
        'diode_static': '',
        'res_metal': '',
        'res': '',
        'mim': '',
    },
}

jinja_env = Environment(
    loader=DictLoader(
        {'mos_cdl': mos_cdl_fmt,
         'mos3_cdl': mos3_cdl_fmt,
         'mos_scs': mos_spectre_fmt,
         'mos3_scs': mos3_spectre_fmt,
         'mos_verilog': mos_verilog_fmt,
         'diode_cdl': dio_cdl_fmt,
         'diode_scs': dio_spectre_fmt,
         'diode_cdl_static': dio_cdl_fmt_static,
         'diode_scs_static': dio_spectre_fmt_static,
         'res_metal_cdl': res_metal_cdl_fmt,
         'res_metal_scs': res_metal_spectre_fmt,
         'res_cdl': res_cdl_fmt,
         'res_scs': res_spectre_fmt,
         'mim_cdl': mim_cdl_fmt,
         'mim_scs': mim_spectre_fmt}),
    keep_trailing_newline=True,
)

prefix_dict = {
    'mos_cdl': 'M',
    'mos3_cdl': 'M',
    'mos_scs': 'M',
    'mos3_scs': 'M',
    'diode_cdl': 'X',
    'diode_scs': 'X',
    'diode_cdl_static': 'X',
    'diode_scs_static': 'X',
    'res_metal_cdl': 'R',
    'res_metal_scs': 'R',
    'res_cdl': 'R',
    'res_scs': 'R',
    'mim_cdl': 'C',
    'mim_scs': 'C',
}


def populate_header(config: Dict[str, Any], inc_lines: Dict[DesignOutput, List[str]],
                    inc_list: Dict[int, List[str]]) -> None:
    for v, lines in inc_lines.items():
        inc_list[v.value] = config[v.name]['includes']


def populate_mos(config: Dict[str, Any], netlist_map: Dict[str, Any],
                 inc_lines: Dict[DesignOutput, List[str]], key: str = 'mos') -> None:
    if key == 'mos':
        _default = mos_default
    elif key == 'mos3':
        _default = mos3_default
    else:
        raise ValueError(f'Unknown key = {key}')
    for cell_name, model_name in config['types']:
        # populate netlist_map
        cur_info = copy.deepcopy(_default)
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            param_list = config[v.name]
            template_name = supported_formats[v][key]
            if template_name:
                mos_template = jinja_env.get_template(template_name)
                lines.append('\n')
                lines.append(
                    mos_template.render(
                        cell_name=cell_name,
                        model_name=_get_model_name(model_name, v.name),
                        param_list=param_list,
                        prefix=_get_prefix(config, v, template_name),
                    ))


def populate_diode(config: Dict[str, Any], netlist_map: Dict[str, Any],
                   inc_lines: Dict[DesignOutput, List[str]]) -> None:
    static: bool = config.get('static', False)
    template_key = 'diode_static' if static else 'diode'

    if 'types' not in config:
        return
    for cell_name, model_name in config['types']:
        # populate netlist_map
        cur_info = copy.deepcopy(dio_default)
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info
        ports = config['port_order'][cell_name]
        if len(ports) == 3:
            cur_info['io_terms'].insert(0, 'GUARD_RING')
        if static:
            cur_info['props'] = {}

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            param_list = config[v.name]
            template_name = supported_formats[v][template_key]
            if template_name:
                jinja_template = jinja_env.get_template(template_name)
                lines.append('\n')
                lines.append(
                    jinja_template.render(
                        cell_name=cell_name,
                        model_name=_get_model_name(model_name, v.name),
                        ports=ports,
                        param_list=param_list,
                        prefix=_get_prefix(config, v, template_name),
                    ))


def populate_res_metal(config: Dict[str, Any], netlist_map: Dict[str, Any],
                       inc_lines: Dict[DesignOutput, List[str]]) -> None:
    for idx, (cell_name, model_name) in enumerate(config['types']):
        # populate netlist_map
        cur_info = copy.deepcopy(res_metal_default)
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            param_list = config[v.name]
            template_name = supported_formats[v]['res_metal']
            write_res_val: Union[bool, Dict[str, bool]] = config.get('write_res_val', False)
            if isinstance(write_res_val, dict):
                write_res_val: bool = write_res_val.get(v.name, False)
            new_param_list = param_list.copy()
            if write_res_val:
                res_val = config['res_map'][idx + 1]
                new_param_list.append(['r', '{}*l/w'.format(res_val)])
            if template_name:
                res_metal_template = jinja_env.get_template(template_name)
                lines.append('\n')
                lines.append(
                    res_metal_template.render(
                        cell_name=cell_name,
                        model_name=_get_model_name(model_name, v.name),
                        param_list=new_param_list,
                        prefix=_get_prefix(config, v, template_name),
                    ))


def populate_res(config: Dict[str, Any], netlist_map: Dict[str, Any], inc_lines: Dict[DesignOutput, List[str]]) -> None:
    num_ports_dict: Dict[str, int] = config.get('num_ports', {})
    for idx, (cell_name, model_name) in enumerate(config['types']):
        # populate netlist_map
        cur_info = copy.deepcopy(res_default)
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info
        num_ports: int = num_ports_dict.get(cell_name, 3)
        if num_ports == 2:
            cur_info['io_terms'].remove('BULK')

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            param_list = config[v.name]
            template_name = supported_formats[v]['res']
            if template_name:
                res_template = jinja_env.get_template(template_name)
                lines.append('\n')
                lines.append(
                    res_template.render(
                        cell_name=cell_name,
                        model_name=_get_model_name(model_name, v.name),
                        num_ports=num_ports,
                        param_list=param_list,
                        prefix=_get_prefix(config, v, template_name),
                    ))


def populate_mim(config: Dict[str, Any], netlist_map: Dict[str, Any], inc_lines: Dict[DesignOutput, List[str]]) -> None:
    for idx, (cell_name, model_name) in enumerate(config['types']):
        # populate netlist_map
        cur_info = copy.deepcopy(mim_default)
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            param_list = config[v.name]
            template_name = supported_formats[v]['mim']
            if template_name:
                mim_template = jinja_env.get_template(template_name)
                lines.append('\n')
                lines.append(
                    mim_template.render(
                        cell_name=cell_name,
                        model_name=_get_model_name(model_name, v.name),
                        param_list=param_list,
                        prefix=_get_prefix(config, v, template_name),
                    ))


def _get_model_name(model_name: Union[str, Dict[str, str]], key: str) -> str:
    if isinstance(model_name, str):
        return model_name
    else:
        return model_name[key]


def _get_prefix(config: Dict[str, Any], key: DesignOutput, template_name: str) -> str:
    if key is DesignOutput.CDL or key is DesignOutput.SPECTRE:
        prefix = config.get('prefix')
        if prefix and key.name in prefix:
            return prefix[key.name]
        return prefix_dict[template_name]
    return ''


def populate_custom_cells(inc_lines: Dict[DesignOutput, List[str]]):
    scs_lines = inc_lines[DesignOutput.SPECTRE]
    scs_lines.append('\n')
    scs_lines.append(scs_ideal_balun)


def get_info(config: Dict[str, Any], output_dir: Path
             ) -> Tuple[Dict[str, Any], Dict[int, List[str]], Dict[int, str]]:
    netlist_map = {}
    inc_lines = {v: [] for v in supported_formats}

    inc_list: Dict[int, List[str]] = {}
    populate_header(config['header'], inc_lines, inc_list)
    populate_mos(config['mos'], netlist_map, inc_lines)
    if 'mos_rf' in config:
        populate_mos(config['mos_rf'], netlist_map, inc_lines)
    if 'mos3' in config:
        populate_mos(config['mos3'], netlist_map, inc_lines, 'mos3')
    populate_diode(config['diode'], netlist_map, inc_lines)
    populate_res_metal(config['res_metal'], netlist_map, inc_lines)
    populate_res(config['res'], netlist_map, inc_lines)
    if 'mim' in config:
        populate_mim(config['mim'], netlist_map, inc_lines)
    populate_custom_cells(inc_lines)

    prim_files: Dict[int, str] = {}
    for v, lines in inc_lines.items():
        fpath = output_dir / supported_formats[v]['fname']
        if lines:
            prim_files[v.value] = str(fpath)
            with open_file(fpath, 'w') as f:
                f.writelines(lines)
        else:
            prim_files[v.value] = ''

    return {'BAG_prim': netlist_map}, inc_list, prim_files


def parse_options() -> Tuple[str, Path]:
    parser = argparse.ArgumentParser(description='Generate netlist setup file.')
    parser.add_argument(
        'config_fname', type=str, help='YAML file containing technology information.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    args = parser.parse_args()
    return args.config_fname, Path(args.output_dir)


def main() -> None:
    config_fname, output_dir = parse_options()

    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(config_fname)

    netlist_map, inc_list, prim_files = get_info(config, output_dir)
    netlist_map.update(netlist_map_default)
    result = {
        'prim_files': prim_files,
        'inc_list': inc_list,
        'netlist_map': netlist_map,
    }

    write_yaml(output_dir / 'netlist_setup.yaml', result)


if __name__ == '__main__':
    main()
