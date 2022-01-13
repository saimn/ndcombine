"""
Run combination benchmarks with various codes.

By default use 20 arrays, 2048x2048 and float32, 16Mb per array, 320Mb for the
20 arrays (x2 if using also the variance plane).

"""

import argparse
import statistics
from functools import wraps
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import CCDData
from astropy.table import Table

BENCHMARKS = {
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


def time_execution(f, nstart=1, nrun=5):
    """Decorator which returns the execution time of a function."""

    @wraps(f)
    def timed(*args, **kw):
        measured = []
        for _ in range(nrun+nstart):
            t0 = time()
            f(*args, **kw)
            measured.append(time() - t0)
            print('.', end='', flush=True)

        mean = statistics.fmean(measured[nstart:])
        std = statistics.stdev(measured[nstart:], mean)
        print(f' Mean of {nrun} : {mean:.2f}±{std:.2f} sec.')
        return mean

    return timed


def autolabel(ax, rects, fmt='.2f'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height:{fmt}}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom')


def barplot(tbl, col='cputime', ax=None, label_fmt='.2f', legend=True):
    benchmarks = sorted(set(tbl['benchmark']))
    codes = sorted(set(tbl['package']))
    x = np.arange(len(benchmarks))  # the label locations
    nbars = len(codes)
    width = 1 / (nbars + 1)  # the width of the bars
    offsets = np.linspace(0, 1, nbars + 1, endpoint=False)
    colors = plt.get_cmap('tab10').colors

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for i, bench in enumerate(benchmarks):
        for j, (off, code) in enumerate(zip(offsets, codes)):
            subt = tbl[(tbl['package'] == code) & (tbl['benchmark'] == bench)]
            rects = ax.bar(x[i] + off,
                           subt[col][0],
                           width,
                           label=code if i == 0 else None,
                           color=colors[j])
            autolabel(ax, rects, fmt=label_fmt)

    ax.set_ylabel(col)
    ax.set_title(f'{col} comparison')
    ax.set_xticks(x + np.median(offsets))
    ax.set_xticklabels(benchmarks)
    if legend:
        ax.legend()


class Compare:

    def __init__(self,
                 limit=None,
                 datadir='~/data/combiner',
                 dtype=None,
                 with_uncertainty=True):

        self.dtype = dtype
        self.with_uncertainty = with_uncertainty
        datadir = Path(datadir).expanduser()

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
        return self.combiner(self.ccds, **kwargs)

    def setup_ccdproc(self, **kwargs):
        import ccdproc
        self.combiner = ccdproc.combine

    def ccdproc(self, **kwargs):
        return self.combiner(self.ccds, dtype=self.dtype, **kwargs)

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
        return stackit(self.ndds)

    def setup_imcombinepy(self):
        import imcombinepy
        self.arrays = np.array([ccd.data for ccd in self.ccds])
        self.combiner = imcombinepy.ndcombine

    def imcombinepy(self, **kwargs):
        if self.with_uncertainty:
            return self.combiner(self.arrays, full=True, **kwargs)[0]
        else:
            return self.combiner(self.arrays, **kwargs)

    def measure_times(self, parallel=False, nrun=5, verbose=True):
        stats = []
        for code, bench in BENCHMARKS.items():
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

                if verbose:
                    print(f'Running {code} - {name}', end='')
                run_func = getattr(self, code)
                run_func()
                func = time_execution(run_func, nrun=nrun)
                tottime = func(**params)
                stats.append({
                    'package': code,
                    'benchmark': name,
                    'cpu_time': tottime,
                })

        tbl = Table(stats)
        tbl['cpu_time'].format = '%.2f'
        return tbl

    def measure_memory(self, verbose=True):
        from memory_profiler import memory_usage
        stats = []
        for code, bench in BENCHMARKS.items():
            for name, params in bench.items():
                setup_func = getattr(self, f'setup_{code}', None)
                if setup_func:
                    setup_func()

                if verbose:
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
                if verbose:
                    print(f'{stats[-1]["memory_peak"]:.1f} Mb')

        tbl = Table(stats)
        tbl['memory_peak'].format = '%d'
        tbl.pprint_exclude_names.add('memory_usage')
        return tbl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks')
    add_arg = parser.add_argument
    add_arg('mode', help='line_profile, memory, or cputime')
    add_arg('--datadir', default='~/data/combiner', help='Path for test data')
    add_arg('--dtype', help='dtype of input data')
    add_arg('--limit', type=int, help='Number of files to combine')
    add_arg('--nrun', type=int, help='Number of execution (for cputime)')
    add_arg('--parallel', action='store_true', help='Use OpenMP (ndcombine)')
    add_arg('--variance', action='store_true', help='Use the variance plane')
    args = parser.parse_args()

    comp = Compare(limit=args.limit,
                   datadir=args.datadir,
                   dtype=args.dtype,
                   with_uncertainty=args.variance)
    if args.mode == 'line_profile':
        comp.profile()
    elif args.mode == 'memory':
        tbl = comp.measure_memory()
        tbl.pprint(max_lines=-1, max_width=-1)
    elif args.mode == 'cputime':
        tbl = comp.measure_times(parallel=args.parallel, nrun=args.nrun)
        tbl.pprint(max_lines=-1, max_width=-1)
