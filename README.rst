ndcombine: Fast ND arrays combination
-------------------------------------

.. image:: https://github.com/saimn/ndcombine/actions/workflows/python-tests.yml/badge.svg
    :target: https://github.com/saimn/ndcombine/actions
    :alt: CI Status
.. image:: https://codecov.io/gh/saimn/ndcombine/branch/main/graph/badge.svg
    :target: https://github.com/saimn/ndcombine
    :alt: Coverage Status

ndcombine is a Python package to combine efficiently n-dimensional arrays such
as images or datacubes. It is implemented in Cython which allows to parallelize
the computation at the C level with OpenMP.

Currently the implemented algorithms are:

- Combination: mean and median.
- Rejection: sigma clipping.

If variance arrays are provided, the variance is propagated with the usual
uncertainty propagation formulas.

Wishlist:

- Combination algorithms: sum
- Regression algorithms: minmax (value or number or percentile ?)
- Weights, scaling factor or function, offsets (?)
