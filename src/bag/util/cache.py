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

"""This module defines classes used to cache existing design masters
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Sequence, Dict, Set, Any, Optional, TypeVar, Type, Tuple, Iterator,
    List, Mapping, Iterable
)

import abc
import time
from collections import OrderedDict
from itertools import chain
from pybag.enum import DesignOutput, SupplyWrapMode, LogLevel
from pybag.core import (
    implement_yaml, implement_netlist, implement_gds, SUPPLY_SUFFIX, PySchCellView, read_gds, make_tr_colors
)

from ..env import get_netlist_setup_file, get_gds_layer_map, get_gds_object_map
from .search import get_new_name
from .immutable import Param, to_immutable
from .logging import LoggingBase

if TYPE_CHECKING:
    from ..core import BagProject
    from ..layout.tech import TechInfo

MasterType = TypeVar('MasterType', bound='DesignMaster')
DBType = TypeVar('DBType', bound='MasterDB')


def format_cell_name(cell_name: str, rename_dict: Dict[str, str], name_prefix: str,
                     name_suffix: str, exact_cell_names: Set[str],
                     supply_wrap_mode: SupplyWrapMode) -> str:
    ans = rename_dict.get(cell_name, cell_name)
    if ans not in exact_cell_names:
        ans = name_prefix + ans + name_suffix
    if supply_wrap_mode is not SupplyWrapMode.NONE:
        ans += SUPPLY_SUFFIX

    return ans


class DesignMaster(LoggingBase, metaclass=abc.ABCMeta):
    """A design master instance.

    This class represents a design master in the design database.

    Parameters
    ----------
    master_db : MasterDB
        the master database.
    params : Param
        the parameters dictionary.
    log_file: str
        the log file path.
    log_level : LogLevel
        the logging level.
    key: Any
        If not None, the unique ID for this master instance.
    copy_state : Optional[Dict[str, Any]]
        If not None, set content of this master from this dictionary.

    Attributes
    ----------
    params : Param
        the parameters dictionary.
    """

    def __init__(self, master_db: MasterDB, params: Param, log_file: str,
                 log_level: LogLevel = LogLevel.DEBUG, *,
                 key: Any = None, copy_state: Optional[Dict[str, Any]] = None) -> None:
        LoggingBase.__init__(self, self.__class__.__name__, log_file, log_level=log_level)
        self._master_db = master_db
        self._cell_name = ''

        if copy_state:
            self._children = copy_state['children']
            self._finalized = copy_state['finalized']
            self._params = copy_state['params']
            self._cell_name = copy_state['cell_name']
            self._key = copy_state['key']
        else:
            # use ordered dictionary so we have deterministic dependency order
            self._children = OrderedDict()
            self._finalized = False

            # set parameters
            self._params = params
            self._key = self.compute_unique_key(params) if key is None else key

            # update design master signature
            self._cell_name = get_new_name(self.get_master_basename(),
                                           self.master_db.used_cell_names)

    @classmethod
    def get_qualified_name(cls) -> str:
        """Returns the qualified name of this class."""
        my_module = cls.__module__
        if my_module is None or my_module == str.__class__.__module__:
            return cls.__name__
        else:
            return my_module + '.' + cls.__name__

    @classmethod
    def populate_params(cls, table: Mapping[str, Any], params_info: Mapping[str, str],
                        default_params: Mapping[str, Any]) -> Param:
        """Fill params dictionary with values from table and default_params"""
        hidden_params = cls.get_hidden_params()

        result = {}
        for key, desc in params_info.items():
            if key not in table:
                if key not in default_params:
                    raise ValueError('Parameter {} not specified.  '
                                     'Description:\n{}'.format(key, desc))
                else:
                    result[key] = default_params[key]
            else:
                result[key] = table[key]

        # add hidden parameters
        for name, value in hidden_params.items():
            result[name] = table.get(name, value)

        return Param(result)

    @classmethod
    def compute_unique_key(cls, params: Param) -> Any:
        """Returns a unique hashable object (usually tuple or string) that represents this instance.

        Parameters
        ----------
        params : Param
            the parameters object.  All default and hidden parameters have been processed already.

        Returns
        -------
        unique_id : Any
            a hashable unique ID representing the given parameters.
        """
        return cls.get_qualified_name(), params

    @classmethod
    def process_params(cls, params: Mapping[str, Any]) -> Tuple[Param, Any]:
        """Process the given parameters dictionary.

        This method computes the final parameters dictionary from the user given one by
        filling in default and hidden parameter values, and also compute the unique ID of
        this master instance.

        Parameters
        ----------
        params : Mapping[str, Any]
            the parameter dictionary specified by the user.

        Returns
        -------
        unique_id : Any
            a hashable unique ID representing the given parameters.
        """
        params_info = cls.get_params_info()
        default_params = cls.get_default_param_values()
        params = cls.populate_params(params, params_info, default_params)
        return params, cls.compute_unique_key(params)

    def update_signature(self, key: Any) -> None:
        self._key = key
        self._cell_name = get_new_name(self.get_master_basename(), self.master_db.used_cell_names)

    def get_copy_state_with(self, new_params: Param) -> Dict[str, Any]:
        return {
            'children': self._children.copy(),
            'finalized': self._finalized,
            'params': new_params,
            'cell_name': self._cell_name,
            'key': self._key,
        }

    def get_copy_with(self: MasterType, new_params: Param) -> MasterType:
        """Returns a copy of this master instance."""
        copy_state = self.get_copy_state_with(new_params)
        return self.__class__(self._master_db, None, copy_state=copy_state)

    @classmethod
    def to_immutable_id(cls, val: Any) -> Any:
        """Convert the given object to an immutable type for use as keys in dictionary.
        """
        try:
            return to_immutable(val)
        except ValueError:
            if hasattr(val, 'get_immutable_key') and callable(val.get_immutable_key):
                return val.get_immutable_key()
            else:
                raise Exception('Unrecognized value %s with type %s' % (str(val), type(val)))

    @classmethod
    @abc.abstractmethod
    def get_params_info(cls) -> Mapping[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Mapping[str, str]
            dictionary from parameter names to descriptions.
        """
        return {}

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : Mapping[str, Any]
            dictionary of default parameter values.
        """
        return {}

    @classmethod
    def get_hidden_params(cls) -> Dict[str, Any]:
        """Returns a dictionary of hidden parameter values.

        hidden parameters are parameters are invisible to the user and only used
        and computed internally.

        Returns
        -------
        hidden_params : Dict[str, Any]
            dictionary of hidden parameter values.
        """
        return {}

    @abc.abstractmethod
    def get_master_basename(self) -> str:
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return ''

    @abc.abstractmethod
    def get_content(self, output_type: DesignOutput, rename_dict: Dict[str, str], name_prefix: str,
                    name_suffix: str, shell: bool, exact_cell_names: Set[str],
                    supply_wrap_mode: SupplyWrapMode) -> Tuple[str, Any]:
        """Returns the content of this master instance.

        Parameters
        ----------
        output_type : DesignOutput
            the output type.
        rename_dict : Dict[str, str]
            the renaming dictionary.
        name_prefix : str
            the name prefix.
        name_suffix : str
            the name suffix.
        shell : bool
            True if we're just producing a shell content (i.e. just top level block).
        exact_cell_names : Set[str]
            set of cell names to keep exact (don't add prefix and suffix)
        supply_wrap_mode : SupplyWrapMode
            the netlisting supply wrap mode.

        Returns
        -------
        cell_name : str
            the master cell name.
        content : Any
            the master content data structure.
        """
        return '', None

    @property
    def master_db(self) -> MasterDB:
        """Returns the database used to create design masters."""
        return self._master_db

    @property
    def lib_name(self) -> str:
        """The master library name"""
        return self._master_db.lib_name

    @property
    def cell_name(self) -> str:
        """The master cell name"""
        return self._cell_name

    @property
    def key(self) -> Optional[Any]:
        """A unique key representing this master."""
        return self._key

    @property
    def finalized(self) -> bool:
        """Returns True if this DesignMaster is finalized."""
        return self._finalized

    @property
    def params(self) -> Param:
        return self._params

    def finalize(self) -> None:
        """Finalize this master instance.
        """
        self._finalized = True

    def add_child_key(self, child_key: object) -> None:
        """Registers the given child key."""
        self._children[child_key] = None

    def clear_children_key(self) -> None:
        """Remove all children keys."""
        self._children.clear()

    def children(self) -> Iterator[object]:
        """Iterate over all children's key."""
        return iter(self._children)


class MasterDB(LoggingBase, metaclass=abc.ABCMeta):
    """A database of existing design masters.

    This class keeps track of existing design masters and maintain design dependency hierarchy.

    Parameters
    ----------
    lib_name : str
        the library to put all generated templates in.
    log_file: str
        the log file path.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated master name prefix.
    name_suffix : str
        generated master name suffix.
    log_level : LogLevel
        the logging level.
    """

    def __init__(self, lib_name: str, log_file: str, prj: Optional[BagProject] = None,
                 name_prefix: str = '', name_suffix: str = '', log_level: LogLevel = LogLevel.DEBUG) -> None:
        LoggingBase.__init__(self, self.__class__.__name__, log_file, log_level=log_level)

        self._prj = prj
        self._lib_name = lib_name
        self._name_prefix = name_prefix
        self._name_suffix = name_suffix

        self._used_cell_names: Set[str] = set()
        self._key_lookup: Dict[Any, Any] = {}
        self._master_lookup: Dict[Any, DesignMaster] = {}

    @property
    def prj(self) -> BagProject:
        return self._prj

    @property
    @abc.abstractmethod
    def tech_info(self) -> TechInfo:
        """TechInfo: the TechInfo object."""
        pass

    @property
    def lib_name(self) -> str:
        """Returns the master library name."""
        return self._lib_name

    @property
    def cell_prefix(self) -> str:
        """Returns the cell name prefix."""
        return self._name_prefix

    @cell_prefix.setter
    def cell_prefix(self, new_val: str) -> None:
        """Change the cell name prefix."""
        self._name_prefix = new_val

    @property
    def cell_suffix(self) -> str:
        """Returns the cell name suffix."""
        return self._name_suffix

    @property
    def used_cell_names(self) -> Set[str]:
        return self._used_cell_names

    @cell_suffix.setter
    def cell_suffix(self, new_val: str) -> None:
        """Change the cell name suffix."""
        self._name_suffix = new_val

    def create_masters_in_db(self, output: DesignOutput, lib_name: str, content_list: List[Any],
                             top_list: List[str],
                             supply_wrap_mode: SupplyWrapMode = SupplyWrapMode.NONE,
                             debug: bool = False, **kwargs: Any) -> None:
        """Create the masters in the design database.

        Parameters
        ----------
        output : DesignOutput
            the output type.
        lib_name : str
            library to create the designs in.
        content_list : Sequence[Any]
            a list of the master contents.  Must be created in this order.
        top_list : List[str]
            list of top level cells.
        supply_wrap_mode : SupplyWrapMode
            the supply wrapping mode.
        debug : bool
            True to print debug messages
        **kwargs : Any
            parameters associated with the given output type.
        """
        start = time.time()
        if output is DesignOutput.LAYOUT:
            if self._prj is None:
                raise ValueError('BagProject is not defined.')

            # create layouts
            self._prj.instantiate_layout(lib_name, content_list)
        elif output is DesignOutput.GDS:
            fname = kwargs['fname']
            square_bracket = kwargs.get('square_bracket', False)

            if square_bracket:
                raise ValueError('square bracket GDS export not supported yet.')

            lay_map = get_gds_layer_map()
            obj_map = get_gds_object_map()
            implement_gds(fname, lib_name, lay_map, obj_map, content_list)
        elif output is DesignOutput.SCHEMATIC:
            if self._prj is None:
                raise ValueError('BagProject is not defined.')

            self._prj.instantiate_schematic(lib_name, content_list)
        elif output is DesignOutput.YAML:
            fname = kwargs['fname']

            implement_yaml(fname, content_list)
        elif output.is_netlist or output.is_model:
            fname = kwargs['fname']
            flat = kwargs.get('flat', False)
            shell = kwargs.get('shell', False)
            top_subckt = kwargs.get('top_subckt', True)
            square_bracket = kwargs.get('square_bracket', False)
            rmin = kwargs.get('rmin', 2000)
            precision = kwargs.get('precision', 6)
            cv_info_list = kwargs.get('cv_info_list', [])
            va_cvinfo_list = kwargs.get('va_cvinfo_list', [])
            cv_info_out = kwargs.get('cv_info_out', None)
            cv_netlist_list = kwargs.get('cv_netlist_list', [])

            prim_fname = get_netlist_setup_file()
            if bool(cv_info_list) != bool(cv_netlist_list):
                raise ValueError('cv_netlist_list and cv_info_list must be given together.')

            implement_netlist(fname, content_list, top_list, output, flat, shell, top_subckt,
                              square_bracket, rmin, precision, supply_wrap_mode, prim_fname,
                              cv_info_list, cv_netlist_list, cv_info_out, va_cvinfo_list)
        else:
            raise ValueError('Unknown design output type: {}'.format(output.name))
        end = time.time()

        if debug:
            print('design instantiation took %.4g seconds' % (end - start))

    def clear(self):
        """Clear all existing schematic masters."""
        self._key_lookup.clear()
        self._master_lookup.clear()

    def new_master(self: MasterDB, gen_cls: Type[MasterType],
                   params: Optional[Mapping[str, Any]] = None, debug: bool = False,
                   **kwargs) -> MasterType:
        """Create a generator instance.

        Parameters
        ----------
        gen_cls : Type[MasterType]
            the generator class to instantiate.  Overrides lib_name and cell_name.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.
        debug : bool
            True to print debug messages.
        **kwargs :
            optional arguments for generator.

        Returns
        -------
        master : MasterType
            the generator instance.
        """
        if params is None:
            params = {}

        master_params, key = gen_cls.process_params(params)
        test = self.find_master(key)
        if test is not None:
            if debug:
                print('master cached')
            return test

        if debug:
            print('finalizing master')
        master = gen_cls(self, master_params, log_file=self.log_file, key=key, log_level=self.log_level, **kwargs)
        start = time.time()
        master.finalize()
        end = time.time()
        self.register_master(key, master)
        if debug:
            print('finalizing master took %.4g seconds' % (end - start))

        return master

    def find_master(self, key: Any) -> Optional[MasterType]:
        return self._master_lookup.get(key, None)

    def register_master(self, key: Any, master: MasterType) -> None:
        self._master_lookup[key] = master
        self._used_cell_names.add(master.cell_name)

    def instantiate_master(self, output: DesignOutput, master: DesignMaster,
                           top_cell_name: str = '', **kwargs) -> None:
        """Instantiate the given master.

        Parameters
        ----------
        output : DesignOutput
            the design output type.
        master : DesignMaster
            the :class:`~bag.layout.template.TemplateBase` to instantiate.
        top_cell_name : str
            name of the top level cell.  If empty, a default name is used.
        **kwargs : Any
            optional arguments for batch_output().
        """
        self.batch_output(output, [(master, top_cell_name)], **kwargs)

    def batch_output(self, output: DesignOutput, info_list: Sequence[Tuple[DesignMaster, str]],
                     debug: bool = False, rename_dict: Optional[Dict[str, str]] = None,
                     **kwargs: Any) -> None:
        """create all given masters in the database.

        Parameters
        ----------
        output : DesignOutput
            The output type.
        info_list : Sequence[Tuple[DesignMaster, str]]
            Sequence of (master, cell_name) tuples to instantiate.
            Use empty string cell_name to use default names.
        debug : bool
            True to print debugging messages
        rename_dict : Optional[Dict[str, str]]
            optional master cell renaming dictionary.
        **kwargs : Any
            parameters associated with the given output type.
        """
        supply_wrap_mode: SupplyWrapMode = kwargs.pop('supply_wrap_mode', SupplyWrapMode.NONE)
        cv_info_list: List[PySchCellView] = kwargs.get('cv_info_list', [])
        va_cvinfo_list: List[PySchCellView] = kwargs.get('va_cvinfo_list', [])
        shell: bool = kwargs.get('shell', False)
        exact_cell_names: Set[str] = kwargs.get('exact_cell_names', set())
        prefix: str = kwargs.get('name_prefix', self._name_prefix)
        suffix: str = kwargs.get('name_suffix', self._name_suffix)
        empty_dict = {}

        if cv_info_list or va_cvinfo_list:
            # need to avoid name collision
            cv_netlist_names = set((cv.cell_name for cv in chain(cv_info_list, va_cvinfo_list)))

            # check that exact cell names won't collide with existing names in netlist
            for name in exact_cell_names:
                if name in cv_netlist_names or (SupplyWrapMode is not SupplyWrapMode.NONE and
                                                (name + SUPPLY_SUFFIX) in cv_netlist_names):
                    raise ValueError(f'Cannot use name {name}, as it is already used by netlist.')

            # get list of names already used by netlist, that we need to avoid
            netlist_used_names = set(_netlist_used_names_iter(cv_netlist_names, prefix, suffix,
                                                              supply_wrap_mode))
        else:
            netlist_used_names = set()

        # configure renaming dictionary.  Verify that renaming dictionary is one-to-one.
        rename: Dict[str, str] = {}
        reverse_rename: Dict[str, str] = {}
        if rename_dict:
            # make sure user renaming won't cause conflicts
            for key, val in rename_dict.items():
                if key != val:
                    if val in reverse_rename:
                        # user renaming is not one-to-one
                        raise ValueError(f'Both {key} and {reverse_rename[val]} are '
                                         f'renamed to {val}')
                    if val in netlist_used_names:
                        # user renaming will conflict with a name in the included netlist file
                        raise ValueError(f'Cannot rename {key} to {val}, name {val} used '
                                         f'by the netlist.')
                    rename[key] = val
                    reverse_rename[val] = key

        # compute names of generated blocks
        top_list: List[str] = []
        for m, name in info_list:
            m_name = m.cell_name
            if name and name != m_name:
                # user wants to rename a generated block
                if name in reverse_rename:
                    # we don't have one-to-one renaming
                    raise ValueError(f'Both {m_name} and {reverse_rename[name]} are '
                                     f'renamed to {name}')
                rename[m_name] = name
                reverse_rename[name] = m_name
                if name in netlist_used_names:
                    # user wants to rename to a name that will conflict with the netlist
                    raise ValueError(f'Cannot use name {name}, as it is already used by netlist.')

                top_list.append(format_cell_name(name, empty_dict, prefix, suffix, exact_cell_names,
                                                 supply_wrap_mode))
                if name in self._used_cell_names:
                    # name is an already used name, so we need to rename other blocks using
                    # this name to something else
                    name2 = get_new_name(name, self._used_cell_names, reverse_rename,
                                         netlist_used_names)
                    rename[name] = name2
                    reverse_rename[name2] = name
            else:
                if m_name in netlist_used_names:
                    if name:
                        raise ValueError(f'Cannot use name {m_name}, '
                                         f'as it is already used by netlist.')
                    else:
                        name2 = get_new_name(m_name, self._used_cell_names, reverse_rename,
                                             netlist_used_names)
                        rename[m_name] = name2
                        reverse_rename[name2] = m_name
                        print(f'renaming {m_name} to {name2}')
                        top_list.append(format_cell_name(name2, empty_dict, prefix, suffix,
                                                         exact_cell_names, supply_wrap_mode))
                else:
                    top_list.append(format_cell_name(m_name, empty_dict, prefix, suffix,
                                                     exact_cell_names, supply_wrap_mode))

        if debug:
            print('Retrieving master contents')

        # use ordered dict so that children are created before parents.
        info_dict = OrderedDict()
        start = time.time()
        for master, _ in info_list:
            self._batch_output_helper(info_dict, master, rename, reverse_rename, netlist_used_names)
        end = time.time()

        content_list = []
        for master in info_dict.values():
            if output is DesignOutput.GDS:
                lay_map = get_gds_layer_map()
                obj_map = get_gds_object_map()
                tr_colors = make_tr_colors(self.tech_info)
                if master.blackbox_gds:
                    for _path in master.blackbox_gds:
                        if _path.is_file():
                            _gds_in = read_gds(str(_path), lay_map, obj_map, self._prj.grid, tr_colors)
                        else:
                            raise ValueError(f'Non existent gds file: {str(_path)}')
                        for _item in _gds_in:
                            _tuple = (_item.cell_name, _item)
                            if _tuple not in content_list:
                                content_list.append(_tuple)
            content_list.append(master.get_content(output, rename, prefix, suffix, shell,
                                                   exact_cell_names, supply_wrap_mode))

        if debug:
            print(f'master content retrieval took {end - start:.4g} seconds')

        self.create_masters_in_db(output, self.lib_name, content_list, top_list,
                                  supply_wrap_mode=supply_wrap_mode, debug=debug, **kwargs)

    def _batch_output_helper(self, info_dict: Dict[str, DesignMaster], master: DesignMaster,
                             rename: Dict[str, str], rev_rename: Dict[str, str],
                             used_names: Set[str]) -> None:
        """Helper method for batch_layout().

        Parameters
        ----------
        info_dict : Dict[str, DesignMaster]
            dictionary from existing master cell name to master objects.
        master : DesignMaster
            the master object to create.
        """
        # get template master for all children
        for master_key in master.children():
            child_temp = self._master_lookup[master_key]
            if child_temp.cell_name not in info_dict:
                self._batch_output_helper(info_dict, child_temp, rename, rev_rename, used_names)

        # get template master for this cell.
        cur_name = master.cell_name
        if cur_name not in rename and cur_name in used_names:
            name2 = get_new_name(cur_name, self._used_cell_names, rev_rename, used_names)
            rename[cur_name] = name2
            rev_rename[name2] = cur_name
        info_dict[cur_name] = self._master_lookup[master.key]

    def exclude_model(self, lib_name: str, cell_name: str) -> bool:
        """True to exclude the given schematic generator when generating behavioral models."""
        if self._prj is None:
            raise ValueError('BagProject is not defined.')
        return self._prj.exclude_model(lib_name, cell_name)


def _netlist_used_names_iter(used_names: Set[str], prefix: str, suffix: str,
                             sup_wrap_mode: SupplyWrapMode) -> Iterable[str]:
    pre_len = len(prefix)
    suf_len = len(suffix)
    sup_suffix = suffix + SUPPLY_SUFFIX
    sup_len = len(sup_suffix)
    for name in used_names:
        # Error checking with exact_names
        if name.startswith(prefix):
            if name.endswith(suffix):
                yield name[pre_len:len(name) - suf_len]
            if sup_wrap_mode is not SupplyWrapMode.NONE and name.endswith(sup_suffix):
                yield name[pre_len:len(name) - sup_len]
