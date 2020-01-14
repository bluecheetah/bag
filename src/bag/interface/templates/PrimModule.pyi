# -*- coding: utf-8 -*-

from typing import Any


from bag.design.module import {{ module_name }}
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class {{ lib_name }}__{{ cell_name }}({{ module_name }}):
    """design module for {{ lib_name }}__{{ cell_name }}.
    """

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        {{ module_name }}.__init__(self, '', database, params, **kwargs)
