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

"""This module defines classes representing various design instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, Optional, Any

from ..util.cache import Param

from pybag.core import PySchInstRef

if TYPE_CHECKING:
    from .database import ModuleDB
    from .module import Module


class SchInstance:
    """This class represents an instance inside a schematic.

    Parameters
    ----------
    db : ModuleDB
        the design database.
    inst_ptr : PySchInstRef
        a reference to the actual schematic instance object.
    """

    def __init__(self, db: ModuleDB, inst_ptr: PySchInstRef,
                 master: Optional[Module] = None) -> None:
        self._db: ModuleDB = db
        self._master: Optional[Module] = master
        self._ptr: PySchInstRef = inst_ptr

        # get schematic class object from master
        if master is None:
            lib_name = self._ptr.lib_name
            static = self._ptr.is_primitive and lib_name != 'BAG_prim'
            if static:
                sch_cls = None
            else:
                cell_name = self._ptr.cell_name
                sch_cls = db.get_schematic_class(lib_name, cell_name)
        else:
            sch_cls = master.__class__

        self._sch_cls: Optional[Type[Module]] = sch_cls

    @property
    def database(self) -> ModuleDB:
        """ModuleDB: the schematic database."""
        return self._db

    @property
    def master(self) -> Optional[Module]:
        """Optional[Module]: the master object of this instance."""
        return self._master

    @property
    def master_class(self) -> Optional[Type[Module]]:
        """Optional[Type[Module]]: the class object of the master of this instance."""
        return self._sch_cls

    @property
    def lib_name(self) -> str:
        """str: the generator library name."""
        return self._ptr.lib_name

    @property
    def cell_name(self) -> str:
        """str: the generator cell name."""
        return self._ptr.cell_name

    @property
    def master_cell_name(self) -> str:
        """str: the cell name of the master object"""
        return self.cell_name if self.master is None else self.master.cell_name

    @property
    def static(self) -> bool:
        """bool: True if this instance points to a static/fixed schematic."""
        return self._sch_cls is None

    @property
    def width(self) -> int:
        """int: the instance symbol width."""
        return self._ptr.width

    @property
    def height(self) -> int:
        """int: the instance symbol height."""
        return self._ptr.height

    @property
    def is_valid(self) -> bool:
        """bool: True if this instance is valid (i.e. static or has a master."""
        return self._sch_cls is None or self.master is not None

    @property
    def is_primitive(self) -> bool:
        """bool: True if this is a primitive (static or in BAG_prim) schematic instance."""
        return self._sch_cls is None or self.master.is_primitive()

    @property
    def should_delete(self) -> bool:
        """bool: True if this instance should be deleted by the parent."""
        return self.master is not None and self.master.should_delete_instance()

    @property
    def master_key(self) -> Optional[Any]:
        """Optional[Any]: A unique key identifying the master object."""
        if self.master is None:
            raise ValueError('Instance {} has no master; cannot get key')
        return self.master.key

    def design(self, **kwargs: Any) -> None:
        """Call the design method on master."""
        if self._sch_cls is None:
            raise RuntimeError('Cannot call design() method on static instances.')

        self._master = self._db.new_master(self._sch_cls, params=kwargs)
        if self._master.is_primitive():
            # update parameters
            for key, val in self._master.get_schematic_parameters().items():
                self.set_param(key, val)
        else:
            self._ptr.lib_name = self._master.lib_name
        self._ptr.cell_name = self._master.cell_name

    def design_model(self, model_params: Param) -> None:
        """Call design_model method on master."""
        if self._sch_cls is None:
            # static instance; assume model is defined in include files
            return

        self._master = self._db.new_model(self._master, model_params)
        self._ptr.cell_name = self._master.cell_name

    def change_generator(self, gen_lib_name: str, gen_cell_name: str,
                         static: bool = False, keep_connections: bool = False) -> None:
        """Change the circuit generator responsible for producing this instance.

        Parameter
        ---------
        gen_lib_name : str
            new generator library name.
        gen_cell_name : str
            new generator cell name.
        static : bool
            True if this is actually a fixed schematic, not a generator.
        keep_connections : bool
            True to keep the old connections when the instance master changed.
        """
        self._master = None
        if static:
            self._sch_cls = None
            prim = True
        else:
            self._sch_cls = self._db.get_schematic_class(gen_lib_name, gen_cell_name)
            prim = self._sch_cls.is_primitive()
        self._ptr.update_master(gen_lib_name, gen_cell_name, prim=prim,
                                keep_connections=keep_connections)

    def set_param(self, key: str, val: Any) -> None:
        """Sets the parameters of this instance.

        Parameters
        ----------
        key : str
            the parameter name.
        val : Any
            the parameter value.
        """
        self._ptr.set_param(key, val)

    def update_connection(self, inst_name: str, term_name: str, net_name: str) -> None:
        """Update connections of this schematic instance.

        Parameters
        ----------
        inst_name : str
            The instance name.
        term_name : str
            The terminal (in other words, port) of the instance.
        net_name : str
            The net to connect the terminal to.
        """
        self._ptr.update_connection(inst_name, term_name, net_name)

    def check_connections(self):
        """Check that the connections of this instance is valid.

        This method is called by the finalize() method, and checks that the user
        connected every port of this instance.
        """
        if self._master is not None:
            self._ptr.check_connections(self._master.pins.keys())

    def get_connection(self, term_name: str) -> str:
        """Get the net name connected to the given terminal.

        Parameters
        ----------
        term_name : str
            the terminal name.

        Returns
        -------
        net_name : str
            the resulting net name.  Empty string if given terminal is not found.
        """
        return self._ptr.get_connection(term_name)

    def get_master_lib_name(self, impl_lib: str) -> str:
        """Returns the master library name.

        the master library could be different than the implementation library in
        the case of static schematic.

        Parameters
        ----------
        impl_lib : str
            implementation library name.

        Returns
        -------
        master_lib : str
            the master library name.

        """
        return self.lib_name if self.is_primitive else impl_lib
