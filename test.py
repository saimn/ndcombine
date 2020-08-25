import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clip as sigma_clip_ast

from ndcombine import combine_arrays, sigma_clip

case = sys.argv[1] if len(sys.argv) > 1 else 'default'

if case == 'sigclip':
    data = np.array([1., 2, 3, 2, 3, 2, 1, 4, 2, 100], dtype=np.float32)
    mask = np.zeros_like(data, dtype=np.uint16)
    var = np.zeros_like(data, dtype=np.float32)
    print(sigma_clip_ast(data).mask.astype(int))
    print(sigma_clip(data, var, mask, 3, 3, 0, 10, 1, 0, 0))

elif case == 'simple':
    data = np.array([
        [1., 2, 3, 2, 3, 2, 1, 4, 2, 100],
        [1., 2, 3, 2, 3, 2, 1, 4, 2, 100],
        [1., 2, 3, 2, 3, 2, 1, 4, 2, 100],
    ],
                    dtype=np.float32).T
    # mask = np.zeros_like(data, dtype=np.uint16)
    print('data:\n', data)
    out = combine_arrays(data, method='mean', clipping_method='sigclip')
    print('outmask:\n', out.meta['REJMASK'])
    print('outdata:\n', out.data)

else:
    datadir = Path(os.path.expanduser('~/data/combiner'))
    flist = list(datadir.glob('image-*.fits'))
    ccds = [CCDData.read(f) for f in flist]
    # data = np.array([ccd.data for ccd in ccds])
    # data = data.reshape(data.shape[0], -1)
    # mask = np.zeros_like(data, dtype=np.uint16)

    if case == 'profile':
        import line_profiler
        # profile = line_profiler.LineProfiler(ndcombine)
        # profile.runcall(ndcombine,
        #                 data,
        #                 mask,
        #                 combine_method='mean',
        #                 reject_method='sigclip')
        # profile.print_stats()
    else:
        t0 = time.time()
        out = combine_arrays(ccds, method='mean', clipping_method='sigclip')
        print(time.time() - t0)
