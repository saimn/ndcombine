# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
#from cython cimport floating
from ndcombine.utils cimport compute_mean_std


def sigma_clip(float [:] data,
               float [:] variance,
               unsigned short [:] mask,
               double lsigma=3,
               double hsigma=3,
               int has_var=0,
               int max_iters=10,
               int use_median=1,
               int use_variance=0,
               int use_mad=0):

    cdef unsigned short [:] outmask = np.zeros(mask.shape[0], dtype=np.uint16,
                                               order='C')

    cy_sigma_clip(&data[0],
                  &variance[0],
                  &mask[0],
                  &outmask[0],
                  mask.shape[0],
                  lsigma,
                  hsigma,
                  has_var,
                  max_iters,
                  use_median,
                  use_variance,
                  use_mad)

    return np.asarray(outmask)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cy_sigma_clip(float data [],
                        float variance [],
                        unsigned short mask [],
                        unsigned short outmask [],
                        size_t npoints,
                        double lsigma,
                        double hsigma,
                        int has_var,
                        size_t max_iters,
                        int use_median,
                        int use_variance,
                        int use_mad) nogil:
    """
    Iterative sigma-clipping.

    Parameters
    ----------
    data : float array
        1D arrays of input, each made up of num_img points for each input pixel.
    variance : float array
        (add description)
    mask : unsigned short array
        (add description)
    lsigma : double
        Number of standard deviations for clipping below the mean.
    hsigma : double
        Number of standard deviations for clipping above the mean.
    has_var : int
        Worry about the input variance array?
    max_iters : int
        Maximum number of iterations to compute
    use_median : int
        Clip around the median rather than mean?
    use_variance : int
        Perform sigma-clipping using the pixel-to-pixel scatter, rather than
        use the variance array?

    Returns
    -------
    mask : numpy.ndarray(np.float)
        (add description)

    """

    cdef size_t i, ngood=0, new_ngood, niter=0
    cdef double avg, var, std
    cdef float low_limit, high_limit
    cdef double result[2]

    if max_iters == 0:
        max_iters = 100

    if use_mad: # TODO
        pass

    for i in range(npoints):
        if mask[i] > 0:
            outmask[i] = 1
        else:
            outmask[i] = 0
            ngood += 1

    #if has_var and use_variance:
    #    for i in range(npoints):
    #        std = sqrt(variance[i])
    #        if data[i] < avg-lsigma*std or data[i] > avg+hsigma*std:
    #            outmask[i] = 1
    #        else:
    #            new_ngood += 1

    while niter < max_iters:

        compute_mean_std(data, outmask, result, use_median, npoints)
        avg = result[0]
        std = result[1]

        new_ngood = 0
        low_limit = avg - lsigma * std
        high_limit = avg + hsigma * std
        for i in range(npoints):
            if data[i] < low_limit or data[i] > high_limit:
                outmask[i] = 1
            else:
                new_ngood += 1

        if new_ngood == ngood:
            break

        ngood = new_ngood
        niter += 1
