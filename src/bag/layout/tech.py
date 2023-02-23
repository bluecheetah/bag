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

"""This module defines BAG's technology related classes"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any, Sequence

import math
from itertools import chain

# try to import cython classes
# noinspection PyUnresolvedReferences
from pybag.core import BBox, PyTech, Transform
from pybag.enum import Orient2D, Orientation, Direction

from ..util.search import BinaryIterator
from ..util.immutable import ImmutableSortedDict, Param

from .data import MaxSpaceFillInfo

if TYPE_CHECKING:
    from .core import PyLayInstance
    from .template import TemplateBase


# Note: pybind11 classes do not work with ABCs, so we throw NotImplementedError as compromise.
class TechInfo(PyTech):
    """The base technology class.

    This class provides various methods for querying technology-specific information.

    Parameters
    ----------
    tech_params : Dict[str, Any]
        process specific parameters.
    config : Dict[str, Any]
        the configuration dictionary corresponding to config_fname.
    config_fname : str
        the configuration file name.

    Attributes
    ----------
    tech_params : Dict[str, Any]
        technology specific parameters.
    """

    def __init__(self, tech_params: Dict[str, Any], config: Dict[str, Any],
                 config_fname: str) -> None:
        PyTech.__init__(self, config_fname)
        self._tech_params = tech_params
        self._config = config
        self._tech_cls_dict: Dict[str, Any] = {}
        self._tech_cls_cache: Dict[Tuple[str, ImmutableSortedDict], Any] = {}

    def get_margin(self, is_vertical: bool, edge1: Param, edge2: Optional[Param]) -> int:
        raise NotImplementedError('Not implemented.')

    def add_cell_boundary(self, template: TemplateBase, box: BBox) -> None:
        """Adds a cell boundary object to the given template.

        This is usually the PR boundary.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the cell boundary in.
        box : BBox
            the cell boundary bounding box.
        """
        raise NotImplementedError('Not implemented.')

    def draw_device_blockage(self, template: TemplateBase) -> None:
        """Draw device blockage layers on the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the device block layers on
        """
        raise NotImplementedError('Not implemented.')

    def get_metal_em_specs(self, layer: str, purpose: str, w: int, length: int = -1,
                           vertical: bool = False, dc_temp: int = -1000, rms_dt: int = -1000
                           ) -> Tuple[float, float, float]:
        """Returns a tuple of EM current/resistance specs of the given wire.

        Parameters
        ----------
        layer : str
            the layer name.
        purpose : str
            the purpose name.
        w : int
            the width of the metal in resolution units (dimension perpendicular to current flow).
        length : int
            the length of the metal in resolution units (dimension parallel to current flow).
            If negative, disable length enhancement.
        vertical : bool
            True to compute vertical current.
        dc_temp : int
            the temperature (in Celsius) to calculate DC current EM spec with.
            If equal to -1000, use technology default.
        rms_dt : int
            the temperature delta (in Celsius) to target for when computing AC RMS current
            EM spec.  If equal to -1000, use technology default.

        Returns
        -------
        idc : float
            maximum DC current, in Amperes.
        iac_rms : float
            maximum AC RMS current, in Amperes.
        iac_peak : float
            maximum AC peak current, in Amperes.
        """
        raise NotImplementedError('Not implemented.')

    def get_via_em_specs(self, layer_dir: int, layer: str, purpose: str, adj_layer: str,
                         adj_purpose: str, cut_w: int, cut_h: int, m_w: int = -1, m_l: int = -1,
                         adj_m_w: int = -1, adj_m_l: int = -1, array: bool = False,
                         dc_temp: int = -1000, rms_dt: int = -1000) -> Tuple[float, float, float]:
        """Returns a tuple of EM current/resistance specs of the given via.

        Parameters
        ----------
        layer_dir : int
            the direction value of the first specified layer.  LOWER if the first layer is the
            bottom layer, UPPER if the first layer is the top layer.
        layer : str
            the first layer name.
        purpose : str
            the first layer purpose name.
        adj_layer : str
            the second layer name.
        adj_purpose : str
            the second layer purpose name.
        cut_w : int
            the via cut width.
        cut_h : int
            the via cut height.
        m_w : int
            the first layer wire width, used for EM enhancement calculations.
            Negative numbers has no effect.
        m_l : int
            the first layer wire length, used for EM enhancement calculations.
            Negative numbers has no effect.
        adj_m_w : int
            the second layer wire width, used for EM enhancement calculations.
            Negative numbers has no effect.
        adj_m_l : int
            the second layer wire length, used for EM enhancement calculations.
            Negative numbers has no effect.
        array : bool
            True if this via is in a via array.
        dc_temp : int
            the temperature (in Celsius) to calculate DC current EM spec with.
            If equal to -1000, use technology default.
        rms_dt : int
            the temperature delta (in Celsius) to target for when computing AC RMS current
            EM spec.  If equal to -1000, use technology default.

        Returns
        -------
        idc : float
            maximum DC current per via, in Amperes.
        iac_rms : float
            maximum AC RMS current per via, in Amperes.
        iac_peak : float
            maximum AC peak current per via, in Amperes.
        """
        raise NotImplementedError('Not implemented.')

    def get_res_em_specs(self, res_type: str, w: int, *, length: int = -1,
                         dc_temp: int = -1000, rms_dt: int = -1000) -> Tuple[float, float, float]:
        """Returns a tuple of EM current/resistance specs of the given resistor.

        Parameters
        ----------
        res_type : str
            the resistor type string.
        w : int
            the width of the metal in resolution units (dimension perpendicular to current flow).
        length : int
            the length of the metal in resolution units (dimension parallel to current flow).
            If negative, disable length enhancement.
        dc_temp : int
            the temperature (in Celsius) to calculate DC current EM spec with.
            If equal to -1000, use technology default.
        rms_dt : int
            the temperature delta (in Celsius) to target for when computing AC RMS current
            EM spec.  If equal to -1000, use technology default.

        Returns
        -------
        idc : float
            maximum DC current, in Amperes.
        iac_rms : float
            maximum AC RMS current, in Amperes.
        iac_peak : float
            maximum AC peak current, in Amperes.
        """
        raise NotImplementedError('Not implemented.')

    @property
    def tech_params(self) -> Dict[str, Any]:
        """Dict[str, Any]: the technology parameters dictionary."""
        return self._tech_params

    @property
    def config(self) -> Dict[str, Any]:
        """Dict[str, Any]: The configuration dictionary used to compute various DRC rules."""
        return self._config

    def register_device_tech(self, dev_name: str, obj: Any) -> None:
        """Register the A technology class for the given device."""
        self._tech_cls_dict[dev_name] = obj

    def get_device_tech(self, dev_name: str, **kwargs: Any) -> Any:
        """Get an instance of the technology class for the given device."""
        cache_key = (dev_name, ImmutableSortedDict(kwargs))
        ans = self._tech_cls_cache.get(cache_key, None)
        if ans is None:
            # make the technology class instance.
            cls = self._tech_cls_dict.get(dev_name, None)
            if cls is None:
                raise ValueError(
                    'Technology class {} not found.  Is it registered?'.format(dev_name))

            ans = self._tech_cls_cache[cache_key] = cls(tech_info=self, **kwargs)
        return ans

    def get_dc_temp(self, dc_temp: int = -1000) -> int:
        """Returns the temperature at which to evaluate DC electro-migration rules."""
        if dc_temp == -1000:
            return self._tech_params['em']['dc_temp']
        return dc_temp

    def get_rms_dt(self, rms_dt: int = -1000) -> int:
        """Returns the delta-temperature requirement for RMS electro-migration rules."""
        if rms_dt == -1000:
            return self._tech_params['em']['rms_dt']
        return rms_dt

    def get_well_layers(self, mos_type: str) -> Sequence[Tuple[str, str]]:
        """Returns a list of well layers associated with the given transistor type.

        """
        return self._config['well_layers'][mos_type]

    def get_implant_layers(self, mos_type: str, res_type: str = '') -> Sequence[Tuple[str, str]]:
        """Returns a list of implant layers associated with the given device type.

        Parameters
        ----------
        mos_type : str
            one of 'nch', 'pch', 'ntap', or 'ptap'
        res_type : str
            If given, the return layers will be for the substrate of the given resistor type.

        Returns
        -------
        imp_list : List[Tuple[str, str]]
            list of implant layers.
        """
        entry_name = 'res_{}'.format(res_type) if res_type else mos_type
        return self.config['imp_layers'][entry_name]

    def get_threshold_layers(self, mos_type: str, threshold: str,
                             res_type: str = '') -> Sequence[Tuple[str, str]]:
        """Returns a list of threshold layers."""
        entry_name = 'res_{}'.format(res_type) if res_type else mos_type
        return self.config['thres_layers'][entry_name][threshold]

    def get_exclude_layer(self, layer_id: int) -> Tuple[str, str]:
        """Returns the metal exclude layer"""
        return self.config['metal_exclude_table'][layer_id]

    def get_dnw_margin(self, dnw_mode: str) -> int:
        """Returns the required DNW margin given the DNW mode.

        Parameters
        ----------
        dnw_mode : str
            the DNW mode string.

        Returns
        -------
        dnw_margin : int
            the DNW margin in resolution units.
        """
        return self.config['dnw_margins'][dnw_mode]

    def get_dnw_layers(self) -> List[Tuple[str, str]]:
        """Returns a list of layers that defines DNW.

        Returns
        -------
        lay_list : List[Tuple[str, str]]
            list of DNW layers.
        """
        return self.config['dnw_layers']

    def has_res_metal(self) -> bool:
        """Returns True if res_metal layers exist in this process"""
        res_metal_layer_table = self.config.get('res_metal_layer_table', {})
        if res_metal_layer_table:
            return True
        return False

    def get_res_metal_layers(self, layer_id: int) -> List[Tuple[str, str]]:
        """Returns a list of layers associated with the given metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.

        Returns
        -------
        res_list : List[Tuple[str, str]]
            list of resistor layers.
        """
        if self.has_res_metal():
            return self.config['res_metal_layer_table'][layer_id]
        raise ValueError('res_metal does not exist in the process.')

    def get_res_rsquare(self, res_type: str) -> float:
        """Returns R-square for the given resistor type.

        This is used to do some approximate resistor dimension calculation.

        Parameters
        ----------
        res_type : str
            the resistor type.

        Returns
        -------
        rsquare : float
            resistance in Ohms per unit square of the given resistor type.
        """
        return self.config['resistor']['info'][res_type]['rsq']

    def get_res_width_bounds(self, res_type: str) -> Tuple[int, int]:
        """Returns the maximum and minimum resistor width for the given resistor type.

        Parameters
        ----------
        res_type : str
            the resistor type.

        Returns
        -------
        wmin : int
            minimum resistor width, in layout units.
        wmax : int
            maximum resistor width, in layout units.
        """
        return self.config['resistor']['info'][res_type]['w_bounds']

    def get_res_length_bounds(self, res_type: str) -> Tuple[int, int]:
        """Returns the maximum and minimum resistor length for the given resistor type.

        Parameters
        ----------
        res_type : str
            the resistor type.

        Returns
        -------
        lmin : int
            minimum resistor length, in layout units.
        lmax : int
            maximum resistor length, in layout units.
        """
        return self.config['resistor']['info'][res_type]['l_bounds']

    def get_res_min_nsquare(self, res_type: str) -> float:
        """Returns the minimum allowable number of squares for the given resistor type.

        Parameters
        ----------
        res_type : str
            the resistor type.

        Returns
        -------
        nsq_min : float
            minimum number of squares needed.
        """
        return self.config['resistor']['info'][res_type]['min_nsq']

    def get_max_space_fill_info(self, layer_id: int) -> MaxSpaceFillInfo:
        fill_info: Tuple[int, int, int, int, float] = self._tech_params['fill'][layer_id]
        return MaxSpaceFillInfo(fill_info)

    def get_idc_scale_factor(self, layer: str, purpose: str, temp: float,
                             is_res: bool = False) -> float:
        """Return the Idc EM specs temperature scale factor.

        Parameters
        ----------
        layer : str
            the layer name.
        purpose : str
            the purpose name.
        temp : float
            the temperature, in Celsius.
        is_res : bool
            True to get scale factor for resistor.

        Returns
        -------
        scale : float
            the scale factor.
        """
        if is_res:
            key = 'res'
        else:
            key = (layer, purpose)
        idc_em_scale = self.config['idc_em_scale']

        idc_params = idc_em_scale.get(key, None)
        if idc_params is None:
            idc_params = idc_em_scale['default']

        temp_list = idc_params['temp']
        scale_list = idc_params['scale']

        for temp_test, scale in zip(temp_list, scale_list):
            if temp <= temp_test:
                return scale
        return scale_list[-1]

    def merge_well(self, template: TemplateBase, inst_list: List[PyLayInstance], sub_type: str, *,
                   threshold: str = '', res_type: str = '', merge_imp: bool = False) -> None:
        """Merge the well of the given instances together."""

        if threshold is not None:
            lay_iter = chain(self.get_well_layers(sub_type),
                             self.get_threshold_layers(sub_type, threshold, res_type=res_type))
        else:
            lay_iter = self.get_well_layers(sub_type)
        if merge_imp:
            lay_iter = chain(lay_iter, self.get_implant_layers(sub_type, res_type=res_type))

        for lay_purp in lay_iter:
            tot_box = BBox.get_invalid_bbox()
            for inst in inst_list:
                cur_box = inst.master.get_rect_bbox(lay_purp)
                tot_box.merge(inst.transform_master_object(cur_box))
            if tot_box.is_physical():
                template.add_rect(lay_purp, tot_box)

    # noinspection PyMethodMayBeStatic
    def finalize_template(self, template: TemplateBase) -> None:
        """Perform any operations necessary on the given layout template before finalizing it.

        By default we add the cell boundary.

        Parameters
        ----------
        template : TemplateBase
            the template object.
        """
        template.add_cell_boundary(template.bound_box)

    def get_res_info(self, res_type: str, w: int, l: int, **kwargs: Any) -> Dict[str, Any]:
        """Returns a dictionary containing EM information of the given resistor.

        Parameters
        ----------
        res_type : str
            the resistor type.
        w : int
            the resistor width in resolution units (dimension perpendicular to current flow).
        l : int
            the resistor length in resolution units (dimension parallel to current flow).
        **kwargs : Any
            optional parameters for EM rule calculations, such as nominal temperature,
            AC rms delta-T, etc.

        Returns
        -------
        info : Dict[str, Any]
            A dictionary of wire information.  Should have the following:

            resistance : float
                The resistance, in Ohms.
            idc : float
                The maximum allowable DC current, in Amperes.
            iac_rms : float
                The maximum allowable AC RMS current, in Amperes.
            iac_peak : float
                The maximum allowable AC peak current, in Amperes.
        """
        rsq = self.get_res_rsquare(res_type)
        res = l / w * rsq
        idc, irms, ipeak = self.get_res_em_specs(res_type, w, length=l, **kwargs)

        return dict(
            resistance=res,
            idc=idc,
            iac_rms=irms,
            iac_peak=ipeak,
        )

    def get_via_info(self, bbox: BBox, layer_dir: Direction, layer: str, adj_layer: str,
                     ex_dir: Orient2D, *, purpose: str = '', adj_purpose: str = '',
                     wlen: int = -1, adj_wlen: int = -1, extend: bool = True,
                     adj_ex_dir: Optional[Orient2D] = None,
                     **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Create a via on the routing grid given the bounding box.

        Parameters
        ----------
        bbox : BBox
            the bounding box of the via.
        layer_dir : Direction
            the direction of the first specified layer.  LOWER if the first layer is the
            bottom layer, UPPER if the first layer is the top layer.
        layer : str
            the first layer name.
        adj_layer : str
            the second layer name.
        ex_dir : Orient2D
            the first layer extension direction.
        purpose : str
            first layer purpose name.
        adj_purpose : str
            second layer purpose name.
        wlen : int
            length of first layer wire connected to this Via, in resolution units.
            Used for length enhancement EM calculation.
        adj_wlen : int
            length of second layer wire connected to this Via, in resolution units.
            Used for length enhancement EM calculation.
        extend : bool
            True if via extension can be drawn outside of bounding box.
        adj_ex_dir : Optional[Orient2D]
            second layer extension direction.  Can force to extend in same direction as bottom.
        **kwargs : Any
            optional parameters for EM rule calculations, such as nominal temperature,
            AC rms delta-T, etc.

        Returns
        -------
        info : Optional[Dict[str, Any]]
            A dictionary of via information, or None if no solution.  Should have the following:

            resistance : float
                The total via array resistance, in Ohms.
            idc : float
                The total via array maximum allowable DC current, in Amperes.
            iac_rms : float
                The total via array maximum allowable AC RMS current, in Amperes.
            iac_peak : float
                The total via array maximum allowable AC peak current, in Amperes.
            params : Dict[str, Any]
                A dictionary of via parameters.
        """
        if adj_ex_dir is None:
            adj_ex_dir = ex_dir.perpendicular()

        via_id = self.get_via_id(layer_dir, layer, purpose, adj_layer, adj_purpose)
        via_param = self.get_via_param(bbox.w, bbox.h, via_id, layer_dir,
                                       ex_dir, adj_ex_dir, extend)

        if via_param.empty:
            # no solution found
            return None

        xform = Transform(bbox.xm, bbox.ym, Orientation.R0)
        m_box = via_param.get_box(xform, layer_dir)
        adj_m_box = via_param.get_box(xform, layer_dir.flip())
        w = m_box.get_dim(ex_dir.perpendicular())
        adj_w = adj_m_box.get_dim(adj_ex_dir.perpendicular())
        cut_dim = via_param.cut_dim
        nx = via_param.nx
        ny = via_param.ny
        idc, irms, ipeak = self.get_via_em_specs(layer_dir, layer, purpose, adj_layer,
                                                 adj_purpose, cut_dim[0], cut_dim[1],
                                                 m_w=w, m_l=wlen, adj_m_w=adj_w, adj_m_l=adj_wlen,
                                                 array=nx > 1 or ny > 1, **kwargs)

        params = {'id': via_id,
                  'xform': Transform(bbox.xm, bbox.ym, Orientation.R0),
                  'via_param': via_param,
                  }

        ntot = nx * ny
        box_list = [None, None]
        box_list[layer_dir] = m_box
        box_list[layer_dir.flip()] = adj_m_box
        return dict(
            resistance=0.0,
            idc=idc * ntot,
            iac_rms=irms * ntot,
            iac_peak=ipeak * ntot,
            params=params,
            metal_box=box_list,
        )

    def design_resistor(self, res_type: str, res_targ: float, idc: float = 0.0,
                        iac_rms: float = 0.0, iac_peak: float = 0.0, num_even: bool = True,
                        **kwargs: Any) -> Tuple[int, int, int, int]:
        """Finds the optimal resistor dimension that meets the given specs.

        Assumes resistor length does not effect EM specs.

        Parameters
        ----------
        res_type : str
            the resistor type.
        res_targ : float
            target resistor, in Ohms.
        idc : float
            maximum DC current spec, in Amperes.
        iac_rms : float
            maximum AC RMS current spec, in Amperes.
        iac_peak : float
            maximum AC peak current spec, in Amperes.
        num_even : int
            True to return even number of resistors.
        **kwargs :
            optional EM spec calculation parameters.

        Returns
        -------
        num_par : int
            number of resistors needed in parallel.
        num_ser : int
            number of resistors needed in series.
        w : int
            width of a unit resistor, in resolution units.
        l : int
            length of a unit resistor, in resolution units.
        """
        rsq = self.get_res_rsquare(res_type)
        wmin_unit, wmax_unit = self.get_res_width_bounds(res_type)
        lmin_unit, lmax_unit = self.get_res_length_bounds(res_type)
        min_nsq = self.get_res_min_nsquare(res_type)

        # make sure width is always even
        wmin_unit = -2 * (-wmin_unit // 2)
        wmax_unit = 2 * (wmax_unit // 2)

        # step 1: find number of parallel resistors and minimum resistor width.
        if num_even:
            npar_iter = BinaryIterator(2, None, step=2)
        else:
            npar_iter = BinaryIterator(1, None, step=1)
        while npar_iter.has_next():
            npar = npar_iter.get_next()
            res_targ_par = res_targ * npar
            idc_par = idc / npar
            iac_rms_par = iac_rms / npar
            iac_peak_par = iac_peak / npar
            res_idc, res_irms, res_ipeak = self.get_res_em_specs(res_type, wmax_unit, **kwargs)
            if (0.0 < res_idc < idc_par or 0.0 < res_irms < iac_rms_par or
                    0.0 < res_ipeak < iac_peak_par):
                npar_iter.up()
            else:
                # This could potentially work, find width solution
                w_iter = BinaryIterator(wmin_unit, wmax_unit + 1, step=2)
                while w_iter.has_next():
                    wcur_unit = w_iter.get_next()
                    lcur_unit = int(math.ceil(res_targ_par / rsq * wcur_unit))
                    if lcur_unit < max(lmin_unit, int(math.ceil(min_nsq * wcur_unit))):
                        w_iter.down()
                    else:
                        tmp = self.get_res_em_specs(res_type, wcur_unit, length=lcur_unit, **kwargs)
                        res_idc, res_irms, res_ipeak = tmp
                        if (0.0 < res_idc < idc_par or 0.0 < res_irms < iac_rms_par or
                                0.0 < res_ipeak < iac_peak_par):
                            w_iter.up()
                        else:
                            w_iter.save_info((wcur_unit, lcur_unit))
                            w_iter.down()

                w_info = w_iter.get_last_save_info()
                if w_info is None:
                    # no solution; we need more parallel resistors
                    npar_iter.up()
                else:
                    # solution!
                    npar_iter.save_info((npar, w_info[0], w_info[1]))
                    npar_iter.down()

        # step 3: fix maximum length violation by having resistor in series.
        num_par, wopt_unit, lopt_unit = npar_iter.get_last_save_info()
        if lopt_unit > lmax_unit:
            num_ser = -(-lopt_unit // lmax_unit)
            lopt_unit = -(-lopt_unit // num_ser)
        else:
            num_ser = 1

        # step 4: return answer
        return num_par, num_ser, wopt_unit, lopt_unit

    @property
    def has_guard_ring(self) -> bool:
        """Returns True if this process has guard rings"""
        return True
