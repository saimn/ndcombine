import argparse
import os
import time
import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path

import ccdproc
import imcombinepy
import numpy as np
from astrodata import NDAstroData
from astropy.nddata import CCDData
from astropy.table import Table

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from gempy.library.nddops import NDStacker

from ndcombine import combine_arrays

benchmarks = {
    'ndcombine': {
        'mean': {'method': 'mean'},
        'mean+sigclip': {'method': 'mean', 'clipping_method': 'sigclip'},
        'median': {'method': 'median'},
    },
    'ccdproc': {
        'mean': {'method': 'average', 'sigma_clip': False},
        'mean+sigclip': {'method': 'average', 'sigma_clip': True},
        'median': {'method': 'median', 'sigma_clip': False},
    },
    'dragons': {
        'mean': {'combine': 'mean'},
        'mean+sigclip': {'combine': 'mean', 'reject': 'sigclip'},
        'median': {'combine': 'median'},
    },
    'imcombinepy': {
        'mean': {'combine': 'mean'},
        'mean+sigclip': {'combine': 'mean', 'reject': 'sigclip'},
        'median': {'combine': 'median'},
    },
}


def time_execution(f):
    """Decorator which returns the execution time of a function."""

    @wraps(f)
    def timed(*args, **kw):
        t0 = time.time()
        n = 5
        for _ in range(n):
            f(*args, **kw)
            print('.', end='', flush=True)

        measured = (time.time() - t0) / n
        print(f'\nMean of 5 : {measured:.2f} sec.')
        return measured

    return timed


class Compare:

    def __init__(self, limit=None, datadir='~/data/combiner'):
        datadir = Path(os.path.expanduser(datadir))

        if not datadir.exists():
            print('Creating test data')
            # datadir.mkdir(parents=True)
            from ndcombine.tests.helpers import make_fake_data
            make_fake_data(20, datadir, nsources=500)

        flist = list(datadir.glob('image-*.fits'))
        self.ccds = [CCDData.read(f, memmap=False) for f in flist[:limit]]
        self.ndds = [
            NDAstroData(ccd.data, uncertainty=ccd.uncertainty, unit=ccd.unit)
            for ccd in self.ccds
        ]
        self.arrays = np.array([ccd.data for ccd in self.ccds])

    def profile(self):
        import line_profiler
        profile = line_profiler.LineProfiler(combine_arrays)
        profile.runcall(combine_arrays,
                        self.ccds,
                        method='mean',
                        clipping_method='sigclip')
        profile.print_stats()

    def ndcombine(self, **kwargs):
        combine_arrays(self.ccds, **kwargs)

    def ccdproc(self, **kwargs):
        ccdproc.combine(self.ccds, **kwargs)

    def dragons(self, **kwargs):
        stackit = NDStacker(**kwargs)
        stackit(self.ndds)

    def imcombinepy(self, **kwargs):
        imcombinepy.ndcombine(self.arrays, **kwargs)

    def measure_times(self, parallel=False):
        measures = defaultdict(dict)
        for code, bench in benchmarks.items():
            for name, params in bench.items():
                if name == 'ndcombine':
                    name = 'ndcombine parallel'
                    params['num_threads'] = 0 if parallel else 1
                func = time_execution(getattr(self, code))
                print(f'Running {code} - {name}', end=' ')
                measures[code][name] = func(**params)
                print()

        tbl = Table([{'package': name, **values}
                     for name, values in measures.items()])
        for col in tbl.itercols():
            if col.dtype.kind == 'f':
                col.format = '.2f'
        tbl.pprint(max_lines=-1, max_width=-1)

    def measure_memory(self):
        from memory_profiler import memory_usage
        measures = defaultdict(dict)
        for code, bench in benchmarks.items():
            for name, params in bench.items():
                print(f'Running {code} - {name}', end=' ')
                res = memory_usage(
                    (getattr(self, code), [], params),
                    timestamps=True,
                    interval=0.01,
                )
                measures[code][name] = np.max(np.array(res) - res[0])
                print()

        tbl = Table([{'package': name, **values}
                     for name, values in measures.items()])
        for col in tbl.itercols():
            if col.dtype.kind == 'f':
                col.format = '.2f'
        tbl.pprint(max_lines=-1, max_width=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('mode', nargs='?',
                        help='profile, memory, or cputime (default)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use OpenMP (for ndcombine)')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--datadir', type=str, default='~/data/combiner',
                        help='Directory where test data is stored')
    args = parser.parse_args()

    comp = Compare(limit=args.limit, datadir=args.datadir)
    if args.mode == 'profile':
        comp.profile()
    elif args.mode == 'memory':
        comp.measure_memory()
    else:
        comp.measure_times(parallel=args.parallel)
