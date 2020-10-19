import os
import sys
import time
import timeit
from pathlib import Path

import bottleneck as bn
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clip as sigma_clip_ast

from ndcombine import combine_arrays, ndcombine, sigma_clip

data = None
mask = None
datamasked = None


def test_sigclip():
    data = np.array([1., 2, 3, 2, 3, 2, 1, 4, 2, 100], dtype=np.float32)
    print('data      :', data)
    print('astropy   :', sigma_clip_ast(data).mask.astype(int))
    print('ndcombine :', sigma_clip(data, lsigma=3, hsigma=3, max_iters=10))

    mask = np.zeros_like(data, dtype=np.uint16)
    mask[7] = 1
    print('\nwith mask :', mask)
    print('astropy   :',
          sigma_clip_ast(np.ma.array(data, mask=mask)).mask.astype(int))
    print('ndcombine :',
          sigma_clip(data, mask=mask, lsigma=3, hsigma=3, max_iters=10))


def test_simple():
    data = np.array([[1., 2, 3, 2, 3, 2, 1, 4, 2, 100]] * 3,
                    dtype=np.float32).T
    print('data:\n', data)
    out = combine_arrays(data, method='mean', clipping_method='sigclip')
    print('outmask:\n', out.meta['REJMASK'])
    print('outdata:\n', out.data)


def test_median():
    global data, mask, datamasked

    shape = (10, 1000, 1000)
    np.random.seed(42)
    data = np.random.normal(size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint16)
    data = data.reshape(shape[0], -1)
    mask = mask.reshape(shape[0], -1)
    datamasked = np.ma.array(data, mask=mask.astype(bool))

    outndcomb, _, _ = ndcombine(data,
                                mask,
                                combine_method='median',
                                reject_method='none')
    outnp = np.median(data, axis=0)
    outnpma = np.ma.median(datamasked, axis=0)
    outbn = bn.median(data, axis=0)

    np.testing.assert_array_equal(outndcomb, outnp)
    np.testing.assert_array_equal(outndcomb, outnpma)
    np.testing.assert_array_equal(outndcomb, outbn)

    nb = 10
    kwargs = dict(globals=globals(), number=nb, repeat=5)

    def run(label, command):
        res = timeit.repeat(command, **kwargs)
        res = np.array(res) / nb
        print(res)
        print(f'{label:20s}: {np.mean(res):.3f}s ± {np.std(res):.3f}s')

    run('np.median', 'np.median(data)')
    run('np.ma.median', 'np.ma.median(datamasked, axis=0)')
    run('bn.median', 'bn.median(data)')
    run('ndcombine 1 thread',
        "ndcombine(data, mask, combine_method='median', "
        "reject_method='none', num_threads=1)")
    run('ndcombine',
        "ndcombine(data, mask, combine_method='median', "
        "reject_method='none')")


def test_files(case):
    datadir = Path(os.path.expanduser('~/data/combiner'))
    flist = list(datadir.glob('image-*.fits'))
    ccds = [CCDData.read(f) for f in flist]

    if case == 'profile':
        import line_profiler
        profile = line_profiler.LineProfiler(combine_arrays)
        profile.runcall(combine_arrays,
                        ccds,
                        method='mean',
                        clipping_method='sigclip')
        profile.print_stats()
    else:
        n = 5

        t0 = time.time()
        for _ in range(n):
            print('.', end='', flush=True)
            combine_arrays(ccds,
                           method='mean',
                           clipping_method='sigclip',
                           num_threads=0)
        print('\nMean of 5 with max threads : {:.2f} sec.'.format(
            (time.time() - t0) / n))

        t0 = time.time()
        for _ in range(n):
            print('.', end='', flush=True)
            combine_arrays(ccds,
                           method='mean',
                           clipping_method='sigclip',
                           num_threads=1)
        print('\nMean of 5 with 1 thread : {:.2f} sec.'.format(
            (time.time() - t0) / n))


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else 'default'

    if case == 'sigclip':
        test_sigclip()
    elif case == 'simple':
        test_simple()
    elif case == 'median':
        test_median()
    else:
        test_files(case)
