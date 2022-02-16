import numpy as np
from astropy.nddata import NDData, VarianceUncertainty

from .ndcombine import ndcombine

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

DATA_t = np.float32
MASK_t = np.uint16


def combine_arrays(
    data,
    mask=None,
    variance=None,
    clipping_limits=(3, 3),
    clipping_method='none',
    max_iters=100,
    method='mean',
    num_threads=0,
    # weights=None,
):
    """
    Parameters:
    -----------
    data : list of ndarray or list of NDData
        Data arrays.
    mask : list of ndarray, optional
        Mask arrays.
    variance : list of ndarray, optional
        Variance arrays.
    clipping_limits : tuple of int
        For sigma clipping, the lower and upper bounds: (sigma_lower,
        sigma_upper).
    clipping_method : str, {'sigclip', 'varclip', 'none'}
        Clipping method.
    max_iters : int
        Maximum number of iterations (for sigma clipping).
    method : str, {'mean', 'median', 'sum'}
        Combination method.
    num_threads : int
        Number of threads.

    """

    def flatten_arr(arr, dtype):
        return arr.astype(dtype, order='C', copy=False).ravel()

    if isinstance(data[0], NDData):
        ndds = data
        input_shape = ndds[0].data.shape
        data, mask, variance = [], [], []

        for nd in ndds:
            data.append(flatten_arr(nd.data, DATA_t))
            if nd.mask is not None:
                mask.append(flatten_arr(nd.mask, MASK_t))
            if nd.uncertainty is not None:
                if not isinstance(nd.uncertainty, VarianceUncertainty):
                    raise ValueError('TODO')
                variance.append(flatten_arr(nd.uncertainty.array, DATA_t))

        # Ensure mask and variance are set to None if empty
        mask = mask or None
        variance = variance or None
    else:
        input_shape = data[0].shape
        data = [flatten_arr(arr, DATA_t) for arr in data]
        if mask is not None:
            mask = [flatten_arr(arr, MASK_t) for arr in mask]
        if variance is not None:
            variance = [flatten_arr(arr, DATA_t) for arr in variance]

    if mask is None:
        mask = list(np.zeros_like(data, dtype=MASK_t))

    lsigma, hsigma = clipping_limits

    outdata, outvar, outmask = ndcombine(
        data,
        mask,
        list_of_var=variance,
        combine_method=method,
        hsigma=hsigma,
        lsigma=lsigma,
        max_iters=max_iters,
        num_threads=num_threads,
        reject_method=clipping_method,
    )

    outdata = outdata.reshape(input_shape)
    if outvar is not None:
        outvar = VarianceUncertainty(outvar.reshape(input_shape))

    out = NDData(outdata, uncertainty=outvar)
    out.meta['REJMAP'] = len(data) - outmask.reshape(input_shape)
    return out
