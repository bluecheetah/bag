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

"""This module defines various OpenMDAO component classes.
"""

import numpy as np
import openmdao.api as omdao


class VecFunComponent(omdao.Component):
    """A component based on a list of functions.

    A component that evaluates multiple functions on the given inputs, then
    returns the result as an 1D array.  Each of the inputs may be a scalar or
    a vector with the same size as the output.  If a vector input is given,
    each function will use a different element of the vector.

    Parameters
    ----------
    output_name : str
        output name.
    fun_list : list[bag.math.dfun.DiffFunction]
        list of interpolator functions, one for each dimension.
    params : list[str]
        list of parameter names.  Parameter names may repeat, in which case the
        same parameter will be used for multiple arguments of the function.
    vector_params : set[str]
        set of parameters that are vector instead of scalar.  If a parameter
        is a vector, it will be the same size as the output, and each function
        only takes in the corresponding element of the parameter.
    """

    def __init__(self, output_name, fun_list, params,
                 vector_params=None):
        omdao.Component.__init__(self)

        vector_params = vector_params or set()

        self._output = output_name
        self._out_dim = len(fun_list)
        self._in_dim = len(params)
        self._params = params
        self._unique_params = {}
        self._fun_list = fun_list

        for par in params:
            adj = par in vector_params
            shape = self._out_dim if adj else 1

            if par not in self._unique_params:
                # linear check, but small list so should be fine.
                self.add_param(par, val=np.zeros(shape))
                self._unique_params[par] = len(self._unique_params), adj

        # construct chain rule jacobian matrix
        self._chain_jacobian = np.zeros((self._in_dim, len(self._unique_params)))
        for idx, par in enumerate(params):
            self._chain_jacobian[idx, self._unique_params[par][0]] = 1

        self.add_output(output_name, val=np.zeros(self._out_dim))

    def __call__(self, **kwargs):
        """Evaluate on the given inputs.

        Parameters
        ----------
        kwargs : dict[str, np.array or float]
            the inputs as a dictionary.

        Returns
        -------
        out : np.array
            the output array.
        """
        tmp = {}
        self.solve_nonlinear(kwargs, tmp)
        return tmp[self._output]

    def _get_inputs(self, params):
        """Given parameter values, construct inputs for functions.

        Parameters
        ----------
        params : VecWrapper, optional
            VecWrapper containing parameters. (p)

        Returns
        -------
        ans : list[list[float]]
            input lists.
        """
        ans = np.empty((self._out_dim, self._in_dim))
        for idx, name in enumerate(self._params):
            ans[:, idx] = params[name]
        return ans

    def solve_nonlinear(self, params, unknowns, resids=None):
        """Compute the output parameter.

        Parameters
        ----------
        params : VecWrapper, optional
            VecWrapper containing parameters. (p)

        unknowns : VecWrapper, optional
            VecWrapper containing outputs and states. (u)

        resids : VecWrapper, optional
            VecWrapper containing residuals. (r)
        """
        xi_mat = self._get_inputs(params)

        tmp = np.empty(self._out_dim)
        for idx in range(self._out_dim):
            tmp[idx] = self._fun_list[idx](xi_mat[idx, :])

        unknowns[self._output] = tmp

    def linearize(self, params, unknowns=None, resids=None):
        """Compute the Jacobian of the parameter.

        Parameters
        ----------
        params : VecWrapper, optional
            VecWrapper containing parameters. (p)

        unknowns : VecWrapper, optional
            VecWrapper containing outputs and states. (u)

        resids : VecWrapper, optional
            VecWrapper containing residuals. (r)
        """
        # print('rank {} computing jac for {}'.format(self.comm.rank, self._outputs))

        xi_mat = self._get_inputs(params)

        jf = np.empty((self._out_dim, self._in_dim))
        for k, fun in enumerate(self._fun_list):
            jf[k, :] = fun.jacobian(xi_mat[k, :])

        jmat = np.dot(jf, self._chain_jacobian)
        jdict = {}
        for par, (pidx, adj) in self._unique_params.items():
            tmp = jmat[:, pidx]
            if adj:
                tmp = np.diag(tmp)
            jdict[self._output, par] = tmp

        return jdict
