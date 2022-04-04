NDCombine: Fast ND arrays combination
=====================================

.. ifconfig:: 'dev' in release

    .. warning::

        This documentation is for the version currently under development.

.. include:: ../README.rst

Installation
============

(Not yet on PyPI!)

The last stable release of NDCombine can be installed simply with pip::

    pip install ndcombine

Usage
=====

First, let's create some fake images with a few sources and cosmic rays. The
FITS files are created in a temporary directory:

.. ipython::

    In [1]: import tempfile
       ...: import glob
       ...: import shutil
       ...: import matplotlib.pyplot as plt
       ...: from astropy.io import fits
       ...: from ndcombine import combine_arrays
       ...: from ndcombine.tests.helpers import make_fake_data

.. ipython::

    In [1]: tmpdir = tempfile.mkdtemp()

    In [2]: make_fake_data(10, tmpdir, nsources=5, ncosmics=10, shape=(50, 50))

Now we can read the data:

.. ipython::

    In [9]: data = [fits.getdata(f) for f in glob.glob(f'{tmpdir}/*.fits')]

    @savefig inputs.png width=12in
    In [3]:
       ...: fig, axes = plt.subplots(1, 4, figsize=(4*3, 3))
       ...: for ax, arr in zip (axes, data):
       ...:     ax.imshow(arr, origin='lower', vmax=1000)

.. ipython::

    In [1]: out = combine_arrays(data, method='mean', clipping_method='sigclip')

    @savefig combined.png width=4in
    In [2]: plt.figure()
       ...: plt.imshow(out.data, origin='lower', vmax=1000)

    In [2]: out.meta['REJMAP']

Cleanup :

.. ipython::

    In [6]: shutil.rmtree(tmpdir)

API
===

.. autofunction:: ndcombine.combine_arrays
