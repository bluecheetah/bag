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

"""This module defines various interpolation classes.
"""

from typing import List, Tuple, Union, Sequence

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage.interpolation as imag_interp

from ..math.dfun import DiffFunction

__author__ = 'erichang'
__all__ = ['interpolate_grid', 'LinearInterpolator']


def _scales_to_points(scale_list,  # type: List[Tuple[float, float]]
                      values,  # type: np.multiarray.ndarray
                      delta=1e-4,  # type: float
                      ):
    # type: (...) -> Tuple[List[np.multiarray.ndarray], List[float]]
    """convert scale_list to list of point values and finite difference deltas."""

    ndim = len(values.shape)
    # error checking
    if ndim == 1:
        raise ValueError('This class only works for dimension >= 2.')
    elif ndim != len(scale_list):
        raise ValueError('input and output dimension mismatch.')

    points = []
    delta_list = []
    for idx in range(ndim):
        num_pts = values.shape[idx]  # type: int
        if num_pts < 2:
            raise ValueError('Every dimension must have at least 2 points.')
        offset, scale = scale_list[idx]
        points.append(np.linspace(offset, (num_pts - 1) * scale + offset, num_pts))
        delta_list.append(scale * delta)

    return points, delta_list


def interpolate_grid(scale_list,  # type: List[Tuple[float, float]]
                     values,  # type: np.multiarray.ndarray
                     method='spline',  # type: str
                     extrapolate=False,  # type: bool
                     delta=1e-4,  # type: float
                     num_extrapolate=3,  # type: int
                     ):
    # type: (...) -> DiffFunction
    """Interpolates multidimensional data on a regular grid.

    returns an Interpolator for the given dataset.

    Parameters
    ----------
    scale_list : List[Tuple[float, float]]
        a list of (offset, spacing).
    values : np.multiarray.ndarray
        The output data in N dimensions.  The length in each dimension must
        be at least 2.
    method : str
        The interpolation method.  Either 'linear', or 'spline'.
        Defaults to 'spline'.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    delta : float
        the finite difference step size.  Finite difference is only used for
        linear interpolation and spline interpolation on 3D data or greater.
        Defaults to 1e-4 of the grid spacing.
    num_extrapolate: int
        If spline interpolation is selected on 3D data or greater, we linearly
        extrapolate the given data by this many points to fix behavior near
        input boundaries.

    Returns
    -------
    fun : DiffFunction
        the interpolator function.
    """
    ndim = len(values.shape)
    if method == 'linear':
        points, delta_list = _scales_to_points(scale_list, values, delta)
        return LinearInterpolator(points, values, delta_list, extrapolate=extrapolate)
    elif ndim == 1:
        return Interpolator1D(scale_list, values, method=method, extrapolate=extrapolate)
    elif method == 'spline':
        if ndim == 2:
            return Spline2D(scale_list, values, extrapolate=extrapolate)
        else:
            return MapCoordinateSpline(scale_list, values, delta=delta, extrapolate=extrapolate,
                                       num_extrapolate=num_extrapolate)
    else:
        raise ValueError('Unsupported interpolation method: %s' % method)


class LinearInterpolator(DiffFunction):
    """A linear interpolator on a regular grid for arbitrary dimensions.

    This class is backed by scipy.interpolate.RegularGridInterpolator.
    Derivatives are calculated using finite difference.

    Parameters
    ----------
    points : Sequence[np.multiarray.ndarray]
        list of points of each dimension.
    values : np.multiarray.ndarray
        The output data in N dimensions.
    delta_list : List[float]
        list of finite difference step size for each axis.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    """

    def __init__(self, points, values, delta_list, extrapolate=False):
        # type: (Sequence[np.multiarray.ndarray], np.multiarray.ndarray, List[float], bool) -> None
        input_range = [(pvec[0], pvec[-1]) for pvec in points]
        DiffFunction.__init__(self, input_range, delta_list=delta_list)
        self._points = points
        self._extrapolate = extrapolate
        self.fun = interp.RegularGridInterpolator(points, values, method='linear',
                                                  bounds_error=not extrapolate,
                                                  fill_value=None)

    def get_input_points(self, idx):
        # type: (int) -> np.multiarray.ndarray
        """Returns the input points for the given dimension."""
        return self._points[idx]

    def __call__(self, xi):
        """Interpolate at the given coordinate.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        ans = self.fun(xi)
        if ans.size == 1:
            return ans[0]
        return ans

    def integrate(self, xstart, xstop, axis=-1, logx=False, logy=False, raw=False):
        # type: (float, float, int, bool, bool, bool) -> Union[LinearInterpolator, np.ndarray]
        """Integrate away the given axis.

        if logx/logy is True, that means this LinearInterpolator is actually used
        to do linear interpolation on the logarithm of the actual data.  This method
        will returns the integral of the actual data.

        Parameters
        ----------
        xstart : float
            the X start value.
        xstop : float
            the X stop value.
        axis : int
            the axis of integration.
            If unspecified, this will be the last axis.
        logx : bool
            True if the values on the given axis are actually the logarithm of
            the real values.
        logy : bool
            True if the Y values are actually the logarithm of the real values.
        raw : bool
            True to return the raw data points instead of a LinearInterpolator object.

        Returns
        -------
        result : Union[LinearInterpolator, np.ndarray]
            float if this interpolator has only 1 dimension, otherwise a new
            LinearInterpolator is returned.
        """
        if self.delta_list is None:
            raise ValueError("Finite differences must be enabled")

        if logx != logy:
            raise ValueError('Currently only works for linear or log-log relationship.')

        ndim = self.ndim
        if axis < 0:
            axis = ndim - 1
        if axis < 0 or axis >= ndim:
            raise IndexError('index out of range.')

        if len(self._points) < ndim:
            raise ValueError("len(self._points) != ndim")

        def calculate_integ_x() -> np.ndarray:
            # find data points between xstart and xstop
            vec_inner = self._points[axis]
            start_idx, stop_idx = np.searchsorted(vec_inner, [xstart, xstop])

            cur_len = stop_idx - start_idx
            if vec_inner[start_idx] > xstart:
                cur_len += 1
                istart = 1
            else:
                istart = 0
            if vec_inner[stop_idx - 1] < xstop:
                cur_len += 1
                istop = cur_len - 1
            else:
                istop = cur_len

            integ_x_inner = np.empty(cur_len)
            integ_x_inner[istart:istop] = vec_inner[start_idx:stop_idx]
            if istart != 0:
                integ_x_inner[0] = xstart

            if istop != cur_len:
                integ_x_inner[cur_len - 1] = xstop

            return integ_x_inner

        # get all input sample points we need to integrate.
        plist = []
        integ_x = calculate_integ_x()  # type: np.ndarray
        new_points = []
        new_deltas = []
        for axis_idx, vec in enumerate(self._points):
            if axis == axis_idx:
                plist.append(integ_x)
            else:
                plist.append(vec)
                new_points.append(vec)
                new_deltas.append(self.delta_list[axis_idx])

        fun_arg = np.stack(np.meshgrid(*plist, indexing='ij'), axis=-1)
        values = self.fun(fun_arg)

        if logx:
            if axis != ndim - 1:
                # transpose values so that broadcasting/slicing is easier
                new_order = [idx for idx in range(ndim) if idx != axis]
                new_order.append(axis)
                values = np.transpose(values, axes=new_order)

            # integrate given that log-log plot is piece-wise linear
            ly1 = values[..., :-1]
            ly2 = values[..., 1:]
            lx1 = np.broadcast_to(integ_x[:-1], ly1.shape)
            lx2 = np.broadcast_to(integ_x[1:], ly1.shape)
            m = (ly2 - ly1) / (lx2 - lx1)

            x1 = np.exp(lx1)
            y1 = np.exp(ly1)

            log_idx = np.abs(m + 1) < 1e-6
            log_idxb = np.invert(log_idx)
            area = np.empty(m.shape)
            area[log_idx] = (y1[log_idx] / np.power(x1[log_idx], m[log_idx]) * (lx2[log_idx] -
                                                                                lx1[log_idx]))

            mp1 = m[log_idxb] + 1
            x2 = np.exp(lx2[log_idxb])
            x1 = x1[log_idxb]
            area[log_idxb] = y1[log_idxb] / mp1 * (np.power(x2 / x1, m[log_idxb]) * x2 - x1)
            new_values = np.sum(area, axis=-1)  # type: np.multiarray.ndarray
        else:
            # just use trapezoid integration
            # noinspection PyTypeChecker
            new_values = np.trapz(values, x=integ_x, axis=axis)

        if not raw and new_points:
            return LinearInterpolator(new_points, new_values, new_deltas,
                                      extrapolate=self._extrapolate)
        else:
            return new_values


class Interpolator1D(DiffFunction):
    """An interpolator on a regular grid for 1 dimensional data.

    This class is backed by scipy.interpolate.InterpolatedUnivariateSpline.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data.  Must be 1 dimension.
    method : str
        extrapolation method.  Either 'linear' or 'spline'.  Defaults to spline.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    """

    def __init__(self, scale_list, values, method='spline', extrapolate=False):
        # error checking
        if len(values.shape) != 1:
            raise ValueError('This class only works for 1D data.')
        elif len(scale_list) != 1:
            raise ValueError('input and output dimension mismatch.')

        if method == 'linear':
            k = 1
        elif method == 'spline':
            k = 3
        else:
            raise ValueError('Unsuppoorted interpolation method: %s' % method)

        offset, scale = scale_list[0]
        num_pts = values.shape[0]
        points = np.linspace(offset, (num_pts - 1) * scale + offset,
                             num_pts)  # type: np.multiarray.ndarray

        DiffFunction.__init__(self, [(points[0], points[-1])], delta_list=None)

        ext = 0 if extrapolate else 2
        self.fun = interp.InterpolatedUnivariateSpline(points, values, k=k, ext=ext)

    def __call__(self, xi):
        """Interpolate at the given coordinate.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        ans = self.fun(xi)
        if ans.size == 1:
            return ans.item()
        return ans

    def deriv(self, xi, idx):
        """Calculate the derivative of the spline along the given index.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)
        idx : int
            The index to calculate the derivative on.

        Returns
        -------
        val : numpy.array
            The derivatives at the given coordinates.
        """
        if idx != 0:
            raise ValueError('Invalid derivative index: %d' % idx)

        ans = self.fun(xi, 1)
        if ans.size == 1:
            return ans[0]
        return ans


class Spline2D(DiffFunction):
    """A spline interpolator on a regular grid for 2D data.

    This class is backed by scipy.interpolate.RectBivariateSpline.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data.  Must be 2D.
    extrapolate : bool
        True to extrapolate data output of given bounds.  Defaults to False.
    """

    def __init__(self, scale_list, values, extrapolate=False):
        # error checking
        if len(values.shape) != 2:
            raise ValueError('This class only works for 2D data.')
        elif len(scale_list) != 2:
            raise ValueError('input and output dimension mismatch.')

        nx, ny = values.shape
        offset, scale = scale_list[0]
        x = np.linspace(offset, (nx - 1) * scale + offset, nx)  # type: np.multiarray.ndarray
        offset, scale = scale_list[1]
        y = np.linspace(offset, (ny - 1) * scale + offset, ny)  # type: np.multiarray.ndarray

        self._min = x[0], y[0]
        self._max = x[-1], y[-1]

        DiffFunction.__init__(self, [(x[0], x[-1]), (y[0], y[-1])], delta_list=None)

        self.fun = interp.RectBivariateSpline(x, y, values)
        self._extrapolate = extrapolate

    def _get_xy(self, xi):
        """Get X and Y array from given coordinates."""
        xi = np.asarray(xi, dtype=float)
        if xi.shape[-1] != 2:
            raise ValueError("The requested sample points xi have dimension %d, "
                             "but this interpolator has dimension 2" % (xi.shape[-1]))

        # check input within bounds.
        x = xi[..., 0]  # type: np.multiarray.ndarray
        y = xi[..., 1]  # type: np.multiarray.ndarray
        if not self._extrapolate and not np.all((self._min[0] <= x) & (x <= self._max[0]) &
                                                (self._min[1] <= y) & (y <= self._max[1])):
            raise ValueError('some inputs are out of bounds.')

        return x, y

    def __call__(self, xi):
        """Interpolate at the given coordinates.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        x, y = self._get_xy(xi)
        return self.fun(x, y, grid=False)

    def deriv(self, xi, idx):
        """Calculate the derivative of the spline along the given index.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)
        idx : int
            The index to calculate the derivative on.

        Returns
        -------
        val : numpy.array
            The derivatives at the given coordinates.
        """
        if idx < 0 or idx > 1:
            raise ValueError('Invalid derivative index: %d' % idx)

        x, y = self._get_xy(xi)
        if idx == 0:
            return self.fun(x, y, dx=1, grid=False)
        else:
            return self.fun(x, y, dy=1, grid=False)


class MapCoordinateSpline(DiffFunction):
    """A spline interpolator on a regular grid for multidimensional data.

    The spline interpolation is done using map_coordinate method in the
    scipy.ndimage.interpolation package.  The derivative is done using
    finite difference.

    if extrapolate is True, we use linear interpolation for values outside of
    bounds.

    Note: By default, map_coordinate uses the nearest value for all points
    outside the boundary.  This will cause undesired interpolation
    behavior near boundary points.  To solve this, we linearly
    extrapolates the given data for a fixed number of points.

    Parameters
    ----------
    scale_list : list[(float, float)]
        a list of (offset, spacing) for each input dimension.
    values : numpy.array
        The output data.
    extrapolate : bool
        True to linearly extrapolate outside of bounds.
    num_extrapolate : int
        number of points to extrapolate in each dimension in each direction.
    delta : float
        the finite difference step size.  Defaults to 1e-4 (relative to a spacing of 1).
    """

    def __init__(self, scale_list, values, extrapolate=False, num_extrapolate=3,
                 delta=1e-4):
        shape = values.shape
        ndim = len(shape)

        # error checking
        if ndim < 3:
            raise ValueError('Data must have 3 or more dimensions.')
        elif ndim != len(scale_list):
            raise ValueError('input and output dimension mismatch.')

        self._scale_list = scale_list
        self._max = [n - 1 + num_extrapolate for n in shape]
        self._extrapolate = extrapolate
        self._ext = num_extrapolate

        # linearly extrapolate given values
        ext_points = [np.arange(num_extrapolate, n + num_extrapolate) for n in shape]
        points, delta_list = _scales_to_points(scale_list, values, delta)
        input_ranges = [(pvec[0], pvec[-1]) for pvec in points]
        self._extfun = LinearInterpolator(ext_points, values, [delta] * ndim, extrapolate=True)

        xi_ext = np.stack(np.meshgrid(*(np.arange(0, n + 2 * num_extrapolate) for n in shape),
                                      indexing='ij', copy=False), axis=-1)

        values_ext = self._extfun(xi_ext)
        self._filt_values = imag_interp.spline_filter(values_ext)

        DiffFunction.__init__(self, input_ranges, delta_list=delta_list)

    def _normalize_inputs(self, xi):
        """Normalize the inputs."""
        xi = np.asarray(xi, dtype=float)
        if xi.shape[-1] != self.ndim:
            raise ValueError("The requested sample points xi have dimension %d, "
                             "but this interpolator has dimension %d" % (xi.shape[-1], self.ndim))

        xi = np.atleast_2d(xi.copy())
        for idx, (offset, scale) in enumerate(self._scale_list):
            xi[..., idx] -= offset
            xi[..., idx] /= scale

        # take extension input account.
        xi += self._ext

        return xi

    def __call__(self, xi):
        """Interpolate at the given coordinate.

        Parameters
        ----------
        xi : numpy.array
            The coordinates to evaluate, with shape (..., ndim)

        Returns
        -------
        val : numpy.array
            The interpolated values at the given coordinates.
        """
        ext = self._ext
        ndim = self.ndim
        xi = self._normalize_inputs(xi)
        ans_shape = xi.shape[:-1]
        xi = xi.reshape(-1, ndim)

        ext_idx_vec = False
        for idx in range(self.ndim):
            ext_idx_vec = ext_idx_vec | (xi[:, idx] < ext) | (xi[:, idx] > self._max[idx])

        int_idx_vec = ~ext_idx_vec
        xi_ext = xi[ext_idx_vec, :]
        xi_int = xi[int_idx_vec, :]
        ans = np.empty(xi.shape[0])
        ans[int_idx_vec] = imag_interp.map_coordinates(self._filt_values, xi_int.T,
                                                       mode='nearest', prefilter=False)
        if xi_ext.size > 0:
            if not self._extrapolate:
                raise ValueError('some inputs are out of bounds.')
            ans[ext_idx_vec] = self._extfun(xi_ext)

        if ans.size == 1:
            return ans[0]
        return ans.reshape(ans_shape)
