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

from typing import Dict, Any

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag_test__net_bus(Module):
    """Module for library bag_test cell net_bus.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'net_bus.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            mult='number of bits in the intermediate bus'
        )

    def design(self, mult: int) -> None:

        self.instances['X0'].design(mult=1)
        self.instances['X1'].design(mult=mult)

        if mult > 1:
            bus_name = f'<{mult - 1}:0>'
            mid_name = f'mid{bus_name}'

            self.rename_pin('mid', mid_name)

            self.rename_instance('X0', f'X0{bus_name}', [('vout', mid_name)])
            self.reconnect_instance_terminal('X1', f'vin{bus_name}', mid_name)
