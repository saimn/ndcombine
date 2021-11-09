import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.stats import sigma_clip as sigma_clip_ast
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ndcombine import combine_arrays, sigma_clip

VALUES_1D = [1., 2, 3, 2, 3, 2, 1, 4, 2, 100]


def test_sigclip():
    """Compare sigma_clip with Astropy."""

    data = np.array(VALUES_1D, dtype=np.float32)
    mask1 = sigma_clip_ast(data).mask.astype(int)
    mask2 = sigma_clip(data, lsigma=3, hsigma=3, max_iters=10)
    assert_array_equal(mask1, mask2)

    mask = np.zeros_like(data, dtype=np.uint16)
    mask[7] = 1
    mask1 = sigma_clip_ast(np.ma.array(data, mask=mask)).mask.astype(int)
    mask2 = sigma_clip(data, mask=mask, lsigma=3, hsigma=3, max_iters=10)
    assert_array_equal(mask1, mask2)


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_combine_array(dtype):
    data = np.array([VALUES_1D], dtype=dtype).T
    out = combine_arrays(data, method='mean', clipping_method='sigclip')

    assert out.data.dtype == np.float32
    assert out.mask is None
    assert out.uncertainty is None
    assert_array_almost_equal(out.data.ravel(), [2.2222223])
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_combine_nddata(dtype):
    data = [NDData(data=np.array([val], dtype=dtype)) for val in VALUES_1D]
    out = combine_arrays(data, method='mean', clipping_method='sigclip')

    assert out.data.dtype == np.float32
    assert out.mask is None
    assert out.uncertainty is None
    assert_array_almost_equal(out.data.ravel(), [2.2222223])
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
