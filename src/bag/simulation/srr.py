# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
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

from typing import Union

from pathlib import Path

import re
import time
import numpy as np

from pybag.enum import DesignOutput
from pybag.core import get_bag_logger

from .data import AnalysisData, SimData, _check_is_md, combine_ana_sim_envs

from srr_python.pysrr import pysrrDataBase, pysrrDataSet


# The following SRR analysis types are mapped to different analysis type names for SimData
_ANA_TYPE_MAP = {
    'td.pss': 'pss_td',
    'fd.pss': 'pss_fd',
    'tran.pss': 'pss_tran',
    'timedomain.pnoise': 'pnoise',
    'pac_timepoint': 'pac',
}


# Some custom exception classes for SRR database parsing
class SRRtoSimDataError(Exception):
    pass


class SRRDatabaseNotReady(SRRtoSimDataError):
    pass


def srr_dataset_to_analysis_data(ds: pysrrDataSet, rtol: float, atol: float) -> AnalysisData:
    """Parse SRR data set and convert into BAG-specific data structure AnalysisData.

    Parameters
    ----------
    ds : pysrrDataSet
        SRR data set.
    rtol: float
        relative tolerance for checking if 2 simulation values are the same.
    atol: float
        absolute tolerance for checking if 2 simulation values are the same.

    Returns
    -------
    ana_data : AnalysisData
        the parsed analysis data.
    """

    data = {}

    # Get sweep information
    swp_vars = ds.getVariableNameList()
    new_swp_vars = list(swp_vars)
    swp_len = ds.getParametricSize()
    swp_combo_arr = np.array([ds.getParametricValueList(i) for i in range(swp_len)])
    swp_combo_list = [swp_combo_arr[:, i] for i in range(len(swp_vars))]
    swp_shape, swp_vals = _check_is_md(1, swp_combo_list, rtol, atol, None)  # each data set only contains single corner
    is_md = swp_shape is not None
    if is_md:
        swp_combo = {var: swp_vals[i] for i, var in enumerate(swp_vars)}
    else:
        raise NotImplementedError("Parametric sweeps must be formatted multi-dimensionally")
    data.update(swp_combo)

    # Parse each signal
    sig_names = ds.getSignalNameList()
    for sig_name in sig_names:
        sig_data = ds.evalSignal(sig_name).getValue()
        if isinstance(sig_data, float):  # Non-parametric data, no sweep
            sig_y = sig_data
            data_shape = swp_shape
        elif isinstance(sig_data, dict):  # Non-parametric data, with sweep
            sig_xname = sig_data['xname']
            sig_y = sig_data['y']
            if sig_xname in swp_vars:  # In some cases, the last sweep variable becomes the x axis of this signal
                assert sig_xname == swp_vars[-1]
                data_shape = swp_shape
            else:
                data_shape = tuple((*swp_shape, sig_y.shape[-1]))
                if sig_xname not in data:
                    data[sig_xname] = sig_data['x']
                    new_swp_vars.append(sig_xname)
        elif isinstance(sig_data, list):  # Parametric data, with sweep
            sig_xname = sig_data[0]['xname']
            len_sig_data = len(sig_data)
            yvecs = [sig_data[i]['y'] for i in range(len_sig_data)]
            sub_dims = tuple(yvec.shape[0] for yvec in yvecs)
            max_dim = max(sub_dims)
            is_same_len = all((sub_dims[i] == sub_dims[0] for i in range(1, len_sig_data)))
            data_shape = tuple((*swp_shape, max_dim))
            if not is_same_len:
                yvecs_padded = [np.pad(yvec, (0, max_dim - dim), constant_values=np.nan)
                                for yvec, dim in zip(yvecs, sub_dims)]
                sig_y = np.stack(yvecs_padded)
                if sig_xname not in data:
                    new_swp_vars.append(sig_xname)
                    xvecs_padded = [np.pad(sub_sig_data['x'], (0, max_dim - dim), constant_values=np.nan)
                                    for sub_sig_data, dim in zip(sig_data, sub_dims)]
                    data[sig_xname] = np.reshape(np.stack(xvecs_padded), data_shape)
            else:
                sig_y = np.stack(yvecs)
                if sig_xname in swp_vars:  # In some cases, the last sweep variable becomes the x axis of this signal
                    assert sig_xname == swp_vars[-1]
                    data_shape = swp_shape
                else:
                    xvecs = np.array([sig_data[i]['x'] for i in range(len_sig_data)])
                    same_xvec = True
                    for xvec in xvecs[1:]:
                        if not np.allclose(xvecs[0], xvec, rtol=rtol, atol=atol):
                            same_xvec = False
                            break
                    if same_xvec:
                        xvecs = xvecs[0]
                    if sig_xname not in data:
                        new_swp_vars.append(sig_xname)
                        data[sig_xname] = xvecs
        else:
            raise TypeError(f"Unexpected signal data type ({type(sig_data)}) for signal {sig_name}")
        try:
            sig_y_reshaped = np.reshape(sig_y, data_shape)
        except ValueError as e:  # Missing some data so reshaping fails
            raise SRRDatabaseNotReady from e
        data[sig_name.replace('/', '.')] = sig_y_reshaped
    return AnalysisData(['corner'] + new_swp_vars, data, is_md)


def get_sim_env(ds: pysrrDataSet) -> str:
    """Get the corner for the given data set.

    Parameters
    ----------
    ds : pysrrDataSet
        SRR data set.

    Returns
    -------
    sim_env : str
        the parsed corner.
    """

    ds_name = ds._name
    ana_type = ds.getAnalysisType()
    sim_env_fmt = r'[a-zA-Z0-9]+_[a-zA-Z0-9]+'
    if ana_type.endswith(('.pss', '.pnoise')):
        ana_type_end = ana_type.split('.')[-1]
        matched = re.search(rf'__+{ana_type_end}__+({sim_env_fmt})__+.*-', ds_name)
    elif ana_type.startswith('pac_'):
        ana_type_start = ana_type.split('_')[0]
        matched = re.search(rf'__+{ana_type_start}__+({sim_env_fmt})__+.*-{ana_type}', ds_name)
    else:
        matched = re.search(rf'__+{ana_type}__+({sim_env_fmt})__+.*', ds_name)
    if not matched:
        raise ValueError(f"Unmatched dataset name {ds_name} of analysis type {ana_type}")
    return matched.group(1)


def srr_to_sim_data(srr_path: Union[str, Path], rtol: float, atol: float) -> SimData:
    """Parse simulation data and convert into BAG-specific data structure SimData.

    Parameters
    ----------
    srr_path : Union[str, Path]
        simulation data directory path.
    rtol: float
        relative tolerance for checking if 2 simulation values are the same.
    atol: float
        absolute tolerance for checking if 2 simulation values are the same.

    Returns
    -------
    sim_data : SimData
        the parsed simulation data.
    """

    logger = get_bag_logger()

    # Due to IO latency, the SRR database may not be fully populated at time of conversion.
    # Try to parse until successful (or timed out)
    num_tries = 0
    max_tries = 10
    while num_tries < max_tries:
        db = pysrrDataBase(srr_path)
        if not db.isValid():
            raise ValueError(f"SRR database {srr_path} is invalid")
        ds_names = sorted(db.dataSetNameList())
        ana_dict = {}
        sim_netlist_type = DesignOutput[db.getAttribute('simulator').upper()]
        try:
            # A simulation database contains a separate data set for each analysis.
            # For multi-corner simulations, there is a separate data set per corner
            for name in ds_names:
                ds = db.getDataSet(name)
                ana_type = ds.getAnalysisType()
                ana_type = _ANA_TYPE_MAP.get(ana_type, ana_type)
                cur_ana = srr_dataset_to_analysis_data(ds, rtol, atol)
                sim_env = get_sim_env(ds)
                if ana_type not in ana_dict:
                    ana_dict[ana_type] = {}
                ana_dict[ana_type][sim_env] = cur_ana

            # Get all corners
            sim_envs = sorted(next(iter(ana_dict.values())))

            # For each analysis type, combine data sets per corner into the same data set
            for ana_type, sub_ana_dict in ana_dict.items():
                ana_dict[ana_type] = combine_ana_sim_envs(sub_ana_dict, sim_envs)

            del db
            return SimData(sim_envs, ana_dict, sim_netlist_type)

        except SRRDatabaseNotReady as e:
            num_tries += 1
            logger.info(f'Error occurred while converting SRR dataset (attempt {num_tries}): {e}\nRestarting...')
            del db
            time.sleep(10)

    raise SRRtoSimDataError('Error occurred while converting SRR dataset. Maximum number of tries reached.')
