import numpy as np
from astropy.stats import sigma_clip as sigma_clip_ast
from ndcombine import combine_arrays, sigma_clip
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_sigclip():
    data = np.array([1., 2, 3, 2, 3, 2, 1, 4, 2, 100], dtype=np.float32)
    mask1 = sigma_clip_ast(data).mask.astype(int)
    mask2 = sigma_clip(data, lsigma=3, hsigma=3, max_iters=10)
    assert_array_equal(mask1, mask2)

    mask = np.zeros_like(data, dtype=np.uint16)
    mask[7] = 1
    mask1 = sigma_clip_ast(np.ma.array(data, mask=mask)).mask.astype(int)
    mask2 = sigma_clip(data, mask=mask, lsigma=3, hsigma=3, max_iters=10)
    assert_array_equal(mask1, mask2)


def test_simple():
    data = np.array([[1., 2, 3, 2, 3, 2, 1, 4, 2, 100]],
                    dtype=np.float32).T
    out = combine_arrays(data, method='mean', clipping_method='sigclip')
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert_array_almost_equal(out.data.ravel(), [2.2222223])
