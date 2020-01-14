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

from typing import List, Dict, Any, Tuple

from pathlib import Path

import h5py
import numpy as np

from pybag.enum import DesignOutput
from pybag.core import get_bag_logger

from ..util.search import BinaryIterator
from .data import AnalysisData, SimData

try:
    # register the blosc filter on load
    import blosc_filter_pybind11
    BLOSC_FILTER = blosc_filter_pybind11.register_blosc_filter()
except ImportError:
    print('WARNING: Error registering BLOSC filter for HDF5.  Default to LZF')
    blosc_filter_pybind11 = None
    BLOSC_FILTER = None

MB_SIZE = 1024**2


def _set_chunk_args(kwargs: Dict[str, Any], chunk_size_mb: int, shape: Tuple[int, ...],
                    unit_size: int) -> None:
    if chunk_size_mb == 0:
        return

    ndim = len(shape)
    num_max = chunk_size_mb * MB_SIZE // unit_size
    chunk_shape = [1] * ndim
    num_cum = 1
    for cur_idx in range(ndim - 1, -1, -1):
        size_cur = shape[cur_idx]
        num_cur = num_cum * size_cur
        if num_cur > num_max:
            # binary search on divisor
            bin_iter = BinaryIterator(2, size_cur + 1)
            while bin_iter.has_next():
                div = bin_iter.get_next()
                q, r = divmod(size_cur, div)
                q += (r != 0)
                num_test = num_cum * q
                if num_test <= num_max:
                    bin_iter.save_info(q)
                    bin_iter.down()
                elif num_test > num_max:
                    bin_iter.up()
                else:
                    bin_iter.save_info(q)
                    break
            chunk_shape[cur_idx] = bin_iter.get_last_save_info()
            break
        else:
            # we can take all values from this dimension
            chunk_shape[cur_idx] = size_cur
            if num_cur == num_max:
                # we're done
                break

    kwargs['chunks'] = tuple(chunk_shape)


def save_sim_data_hdf5(data: SimData, hdf5_path: Path, compress: bool = True,
                       chunk_size_mb: int = 2, cache_size_mb: int = 20,
                       cache_modulus: int = 2341) -> None:
    """Saves the given MDArray as a HDF5 file.

    The simulation environments are stored as fixed length byte strings,
    and the sweep parameters are stored as dimension label for each data.

    Parameters
    ----------
    data: SimData
        the data.
    hdf5_path: Path
        the hdf5 file path.
    compress : str
        HDF5 compression method.  Defaults to 'lzf' for speed (use 'gzip' for space).
    chunk_size_mb : int
        HDF5 data chunk size, in megabytes.  0 to disable.
    cache_size_mb : int
        HDF5 file chunk cache size, in megabytes.
    cache_modulus : int
        HDF5 file chunk cache modulus.
    """
    # create parent directory
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    str_kwargs: Dict[str, Any] = {}
    dset_kwargs: Dict[str, Any] = {}
    if compress:
        if chunk_size_mb == 0:
            raise ValueError('Compression can only be done with chunk storage')
        if BLOSC_FILTER is None:
            dset_kwargs['compression'] = 'lzf'
            dset_kwargs['shuffle'] = True
        else:
            dset_kwargs['compression'] = BLOSC_FILTER
            dset_kwargs['compression_opts'] = (0, 0, 0, 0, 5, 1, 0)
            dset_kwargs['shuffle'] = False

    with h5py.File(str(hdf5_path), 'w', libver='latest', rdcc_nbytes=cache_size_mb * MB_SIZE,
                   rdcc_w0=1.0, rdcc_nslots=cache_modulus) as f:
        arr = np.array(data.sim_envs, dtype='S')
        _set_chunk_args(str_kwargs, chunk_size_mb, arr.shape, arr.dtype.itemsize)
        f.create_dataset('__corners', data=arr, **str_kwargs)
        f.attrs['netlist_type'] = data.netlist_type.value
        for group in data.group_list:
            data.open_group(group)
            grp = f.create_group(group)
            grp.attrs['is_md'] = data.is_md
            arr = np.array(data.sweep_params, dtype='S')
            _set_chunk_args(str_kwargs, chunk_size_mb, arr.shape, arr.dtype.itemsize)
            grp.create_dataset('__sweep_params', data=arr, **str_kwargs)
            for name, arr in data.items():
                _set_chunk_args(dset_kwargs, chunk_size_mb, arr.shape, arr.dtype.itemsize)
                grp.create_dataset(name, data=arr, **dset_kwargs)


def load_sim_data_hdf5(path: Path, cache_size_mb: int = 20, cache_modulus: int = 2341) -> SimData:
    """Read simulation results from HDF5 file.

    Parameters
    ----------
    path : Path
        the file to read.
    cache_size_mb : int
        HDF5 file chunk cache size, in megabytes.
    cache_modulus : int
        HDF5 file chunk cache modulus.

    Returns
    -------
    results : SimData
        the data.
    """
    if not path.is_file():
        raise FileNotFoundError(f'{path} is not a file.')

    with h5py.File(str(path), 'r', rdcc_nbytes=cache_size_mb * MB_SIZE, rdcc_nslots=cache_modulus,
                   rdcc_w0=1.0) as f:
        corners: List[str] = []
        ana_dict: Dict[str, AnalysisData] = {}
        for ana, obj in f.items():
            if ana == '__corners':
                corners = obj[:].astype('U').tolist()
            else:
                sweep_params: List[str] = []
                sig_dict: Dict[str, np.ndarray] = {}
                is_md: bool = bool(obj.attrs['is_md'])
                for sig, dset in obj.items():
                    if sig == '__sweep_params':
                        sweep_params = dset[:].astype('U').tolist()
                    else:
                        sig_dict[sig] = dset[:]
                ana_dict[ana] = AnalysisData(sweep_params, sig_dict, is_md)

        netlist_code = f.attrs.get('netlist_type', None)
        if netlist_code is None:
            logger = get_bag_logger()
            logger.warn('Old HDF5 file: cannot find attribute "netlist_type".  Assuming SPECTRE.')
            netlist_type = DesignOutput.SPECTRE
        else:
            netlist_type = DesignOutput(netlist_code)

        ans = SimData(corners, ana_dict, netlist_type)

    return ans
