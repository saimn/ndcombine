import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.stats import sigma_clip as sigma_clip_ast
from numpy.testing import assert_array_equal

from ndcombine import combine_arrays, sigma_clip

# Test values:
# - without outlier: mean=2.2, median=2.0, std=0.87, sum=22.0, len=10
# - with outlier: mean=11.09, median=2.0, std=28.13, sum=122.0, len=11
TEST_VALUES = [1, 2, 3, 2, 3, 2, 1, 4, 2, 2, 100]


def test_sigclip():
    """Compare sigma_clip with Astropy."""
    data = np.array(TEST_VALUES, dtype=np.float32)
    mask1 = sigma_clip_ast(data).mask.astype(int)
    mask2 = sigma_clip(data, lsigma=3, hsigma=3, max_iters=10)
    assert_array_equal(mask1, mask2)

    mask2 = sigma_clip(data, lsigma=3, hsigma=3, max_iters=0)
    assert_array_equal(mask1, mask2)


def test_sigclip_with_mask():
    data = np.array(TEST_VALUES, dtype=np.float32)
    mask = np.zeros_like(data, dtype=np.uint16)
    mask[7] = 1
    mask1 = sigma_clip_ast(np.ma.array(data, mask=mask)).mask.astype(int)
    mask2 = sigma_clip(data, mask=mask, lsigma=3, hsigma=3, max_iters=10)
    assert_array_equal(mask1, mask2)


def test_sigclip_with_var():
    data = np.array(TEST_VALUES, dtype=np.float32)

    var = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000], dtype=np.float32)
    mask = sigma_clip(data, variance=var, lsigma=3, hsigma=3, max_iters=10,
                      use_variance=True)
    assert_array_equal(mask, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    var = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100_000], dtype=np.float32)
    mask = sigma_clip(data, variance=var, lsigma=3, hsigma=3, max_iters=10,
                      use_variance=True)
    assert_array_equal(mask, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    var = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100_000], dtype=np.float32)
    mask = sigma_clip(data, variance=var, lsigma=3, hsigma=3, max_iters=10,
                      use_variance=False)
    assert_array_equal(mask, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_combine_array(dtype):
    data = np.array([TEST_VALUES], dtype=dtype).T
    out = combine_arrays(data, method='mean', clipping_method='sigclip')

    assert out.data.dtype == np.float32
    assert out.mask is None
    assert out.uncertainty is None
    assert np.isclose(out.data[0], 2.2)
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_combine_nddata(dtype):
    data = [NDData(data=np.array([val], dtype=dtype)) for val in TEST_VALUES]
    out = combine_arrays(data, method='mean', clipping_method='sigclip')

    assert out.data.dtype == np.float32
    assert out.mask is None
    assert out.uncertainty is None
    assert np.isclose(out.data[0], 2.2)
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_no_clipping(dtype):
    data = np.array([TEST_VALUES], dtype=dtype).T
    out = combine_arrays(data, method='mean', clipping_method='none')
    assert np.isclose(out.data[0], 11.09, atol=1e-2)
    assert_array_equal(out.meta['REJMASK'].ravel(), 0)


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_array_with_mask(dtype):
    data = np.array([TEST_VALUES], dtype=dtype).T
    mask = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]], dtype=bool).T
    out = combine_arrays(data,
                         mask=mask,
                         method='mean',
                         clipping_method='sigclip')

    assert np.isclose(out.data[0], 2.)
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1])


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_nddata_with_mask(dtype):
    mask_values = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    data = [
        NDData(data=np.array([val], dtype=dtype),
               mask=np.array([mask], dtype=bool))
        for val, mask in zip(TEST_VALUES, mask_values)
    ]
    out = combine_arrays(data, method='mean', clipping_method='sigclip')

    assert np.isclose(out.data[0], 2.)
    assert_array_equal(out.meta['REJMASK'].ravel(),
                       [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1])
