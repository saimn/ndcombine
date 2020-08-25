import time

import numpy as np
from astropy.nddata import NDData

from .ndcombine import ndcombine
from .sigma_clip import sigma_clip


def combine_arrays(
    data,
    mask=None,
    variance=None,
    method='mean',
    weights=None,
    clipping_method='none',
    clip_limits=(-3, 3),
    # clipping methods
    # clip_extrema=False, nlow=1, nhigh=1,
    # minmax_clip=False, minmax_clip_min=None, minmax_clip_max=None,
    # sigma_clip=False, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
):
    """
    Parameters:
    -----------
    data : list of ndarray or list of NDData
        Data.
    method : str, {'average', 'median', 'sum'}
        Combination method.
    clipping_method : str, {'minmax', 'extrema', 'sigmaclip', 'none'}
        Clipping method.

    """
    if isinstance(data[0], NDData):
        ndds = data
        data = np.asarray([nd.data for nd in ndds], dtype=np.float32)
        mask = np.asarray([(nd.mask if nd.mask is not None
                            else np.zeros(nd.shape, dtype=np.uint16))
                           for nd in ndds],
                          dtype=np.uint16)
    else:
        data = np.asarray(data, dtype=np.float32)
        if mask is not None:
            mask = np.asarray(mask, dtype=np.uint16)
        else:
            mask = np.zeros_like(data, dtype=np.uint16)

    shape = data.shape
    data = data.reshape(data.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1)

    t0 = time.time()
    out, outmask = ndcombine(data,
                             mask,
                             combine_method=method,
                             reject_method=clipping_method)
    print(time.time() - t0)

    out = out.reshape(shape[1:])
    out = NDData(out)
    out.meta['REJMASK'] = outmask.reshape(shape)
    return out
