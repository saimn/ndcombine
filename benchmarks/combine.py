"""
Run combination benchmarks with various codes.

By default use 20 arrays, 2048x2048 and float32, 16Mb per array, 320Mb for the
20 arrays (x2 if using also the variance plane).

"""

import argparse
import os
import time
from functools import wraps
from pathlib import Path

import numpy as np
from astropy.nddata import CCDData
from astropy.table import Table

benchmarks = {
    'ndcombine': {
        'mean': {
            'method': 'mean'
        },
        'mean+sigclip': {
            'method': 'mean',
            'clipping_method': 'sigclip'
        },
        'median': {
            'method': 'median'
        },
    },
    'ccdproc': {
        'mean': {
            'method': 'average',
            'sigma_clip': False
        },
        'mean+sigclip': {
            'method': 'average',
            'sigma_clip': True
        },
        'median': {
            'method': 'median',
            'sigma_clip': False
        },
    },
    'dragons': {
        'mean': {
            'combine': 'mean'
        },
        'mean+sigclip': {
            'combine': 'mean',
            'reject': 'sigclip'
        },
        'median': {
            'combine': 'median'
        },
    },
    'imcombinepy': {
        'mean': {
            'combine': 'mean'
        },
        'mean+sigclip': {
            'combine': 'mean',
            'reject': 'sigclip'
        },
        'median': {
            'combine': 'median'
        },
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
    def __init__(self,
                 limit=None,
                 datadir='~/data/combiner',
                 dtype=None,
                 with_uncertainty=True):
        datadir = Path(os.path.expanduser(datadir))

        if not datadir.exists():
            print('Creating test data')
            datadir.mkdir(parents=True)
            from ndcombine.tests.helpers import make_fake_data
            make_fake_data(20, datadir, nsources=500)

        flist = list(datadir.glob('image-*.fits'))
        self.ccds = []
        for f in flist[:limit]:
            ccd = CCDData.read(f, memmap=False)
            if dtype:
                ccd.data = ccd.data.astype(dtype)
                ccd.uncertainty.array = ccd.uncertainty.array.astype(dtype)
            if not with_uncertainty:
                ccd.uncertainty = None
            self.ccds.append(ccd)

    def profile(self):
        import line_profiler
        from ndcombine import combine_arrays
        profile = line_profiler.LineProfiler(combine_arrays)
        profile.runcall(combine_arrays,
                        self.ccds,
                        method='mean',
                        clipping_method='sigclip')
        profile.print_stats()

    def setup_ndcombine(self, **kwargs):
        from ndcombine import combine_arrays
        self.combiner = combine_arrays

    def ndcombine(self, **kwargs):
        self.combiner(self.ccds, **kwargs)

    def setup_ccdproc(self, **kwargs):
        import ccdproc
        self.combiner = ccdproc.combine

    def ccdproc(self, **kwargs):
        self.combiner(self.ccds, **kwargs)

    def setup_dragons(self):
        from astrodata import NDAstroData
        from gempy.library.nddops import NDStacker
        self.ndds = [
            NDAstroData(ccd.data, uncertainty=ccd.uncertainty, unit=ccd.unit)
            for ccd in self.ccds
        ]
        self.combiner = NDStacker

    def dragons(self, **kwargs):
        stackit = self.combiner(**kwargs)
        stackit(self.ndds)

    def setup_imcombinepy(self):
        import imcombinepy
        self.arrays = np.array([ccd.data for ccd in self.ccds])
        self.combiner = imcombinepy.ndcombine

    def imcombinepy(self, **kwargs):
        self.combiner(self.arrays, **kwargs)

    def measure_times(self, parallel=False):
        stats = []
        for code, bench in benchmarks.items():
            for name, params in bench.items():
                if code == 'ndcombine':
                    if parallel:
                        code = 'ndcombine parallel'
                        params['num_threads'] = 0
                    else:
                        params['num_threads'] = 1

                setup_func = getattr(self, f'setup_{code}', None)
                if setup_func:
                    setup_func()

                print(f'Running {code} - {name}', end=' : ')
                run_func = getattr(self, code)
                run_func()
                func = time_execution(run_func)
                tottime = func(**params)
                stats.append({
                    'package': code,
                    'benchmark': name,
                    'cpu_time': tottime,
                })
                print()

        tbl = Table(stats)
        tbl['cpu_time'].format = '%.2f'
        tbl.pprint(max_lines=-1, max_width=-1)
        return tbl

    def measure_memory(self):
        from memory_profiler import memory_usage
        stats = []
        for code, bench in benchmarks.items():
            for name, params in bench.items():
                setup_func = getattr(self, f'setup_{code}', None)
                if setup_func:
                    setup_func()

                print(f'Running {code} - {name}', end=' : ')
                res = memory_usage(
                    (getattr(self, code), [], params),
                    timestamps=True,
                    interval=0.01,
                )
                stats.append({
                    'package': code,
                    'benchmark': name,
                    'memory_usage': np.array(res),
                    'memory_peak': np.max(np.array(res) - res[0]),
                })
                print(f'{stats[-1]["memory_peak"]:.1f} Mb')

        tbl = Table(stats)
        tbl['memory_peak'].format = '%d'
        tbl.pprint_exclude_names.add('memory_usage')
        tbl.pprint(max_lines=-1, max_width=-1)
        return tbl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('mode',
                        help='line_profile, memory, or cputime (default)')
    parser.add_argument('--parallel',
                        action='store_true',
                        help='Use OpenMP (for ndcombine)')
    parser.add_argument('--variance',
                        action='store_true',
                        help='Use the variance plane')
    parser.add_argument('--limit', type=int, help='Number of files to combine')
    parser.add_argument('--dtype', help='dtype of input data')
    parser.add_argument('--datadir',
                        type=str,
                        default='~/data/combiner',
                        help='Directory where test data is stored')
    args = parser.parse_args()

    comp = Compare(limit=args.limit,
                   datadir=args.datadir,
                   dtype=args.dtype,
                   with_uncertainty=args.variance)
    if args.mode == 'line_profile':
        comp.profile()
    elif args.mode == 'memory':
        comp.measure_memory()
    elif args.mode == 'cputime':
        comp.measure_times(parallel=args.parallel)
