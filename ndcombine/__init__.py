import numpy as np
from astropy.nddata import NDData, VarianceUncertainty

from .ndcombine import ndcombine
from .sigma_clip import sigma_clip  # noqa

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''


def combine_arrays(
    data,
    mask=None,
    variance=None,
    clipping_limits=(3, 3),
    clipping_method='none',
    method='mean',
    num_threads=0,
    # weights=None,
    # clipping methods
    # clip_extrema=False, nlow=1, nhigh=1,
    # minmax_clip=False, minmax_clip_min=None, minmax_clip_max=None,
    # sigma_clip=False, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
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
    clipping_method : str, {'minmax', 'extrema', 'sigmaclip', 'none'}
        Clipping method.
    method : str, {'mean', 'median', 'sum'}
        Combination method.
    num_threads : int
        Number of threads.

    """
    if isinstance(data[0], NDData):
        ndds = data
        shape = ndds[0].shape
        data = [nd.data.astype('float32', copy=False).ravel() for nd in ndds]

        if ndds[0].mask is not None:
            # For now suppose that all NDData objects have a mask if the
            # first object has one.
            mask = [
                nd.mask.astype('uint16', copy=False).ravel() for nd in ndds
            ]
        else:
            mask = None

        if ndds[0].uncertainty is not None:
            if not isinstance(ndds[0].uncertainty, VarianceUncertainty):
                raise ValueError('TODO')
            # For now suppose that all NDData objects have a mask if the
            # first object has one.
            variance = [
                nd.uncertainty.array.astype('float32', copy=False).ravel()
                for nd in ndds
            ]
        else:
            variance = None
    else:
        raise ValueError
        # data = np.asarray(data, dtype=np.float32)
        # shape = data.shape[1:]
        # data = data.reshape(data.shape[0], -1)
        # if mask is not None:
        #     mask = np.asarray(mask, dtype=np.uint16)
        #     mask = mask.reshape(mask.shape[0], -1)
        # if variance is not None:
        #     variance = np.asarray(variance, dtype=np.float32)
        #     variance = variance.reshape(variance.shape[0], -1)

    if mask is None:
        mask = list(np.zeros_like(data, dtype=np.uint16))

    lsigma, hsigma = clipping_limits
    max_iters = 100

    outdata, outvar, outmask = ndcombine(
        data,
        mask,
        combine_method=method,
        hsigma=hsigma,
        lsigma=lsigma,
        max_iters=max_iters,
        num_threads=num_threads,
        reject_method=clipping_method,
        list_of_var=variance,
    )

    outdata = outdata.reshape(shape)
    if outvar is not None:
        outvar = VarianceUncertainty(outvar.reshape(shape))

    out = NDData(outdata, uncertainty=outvar)
    out.meta['REJMASK'] = outmask.reshape((-1, ) + shape)
    return out
