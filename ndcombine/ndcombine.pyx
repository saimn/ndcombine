# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport M_PI_2
from libc.stdlib cimport malloc, free

from ndcombine.utils cimport (compute_mean, compute_median, compute_mean_var,
                              compute_sum, cy_sigma_clip)

np.import_array()

ctypedef unsigned short mask_t

cdef enum rejection_methods:
    SIGCLIP,
    VARCLIP,
    MINMAX,
    NONE

cdef enum combine_methods:
    MEAN,
    MEDIAN,
    SUM


@cython.boundscheck(False)
@cython.wraparound(False)
def ndcombine(list list_of_data,
              list list_of_mask,
              list list_of_var=None,
              combine_method='mean',
              reject_method='none',
              double lsigma=3,
              double hsigma=3,
              size_t max_iters=100,
              int num_threads=0):

    cdef:
        ssize_t npoints = len(list_of_data)
        ssize_t npix = list_of_data[0].shape[0]
        ssize_t i, j

        int use_variance = 1 if list_of_var is not None else 0

        float *tmpdata
        float *tmpvar
        unsigned short *tmpmask

        float **data = <float **> malloc(npoints * sizeof(float *))
        unsigned short **mask = <unsigned short **> malloc(
            npoints * sizeof(unsigned short*))
        float **var = <float **> malloc(npoints * sizeof(float *))

        np.ndarray[np.float32_t, ndim=1, mode="c"] temp_float
        np.ndarray[np.uint16_t, ndim=1, mode="c"] temp_uint

    for i in range(npoints):
        temp_float = list_of_data[i]
        data[i]= &temp_float[0]
        if use_variance:
            temp_float = list_of_var[i]
            var[i] = &temp_float[0]
        temp_uint = list_of_mask[i]
        mask[i] = &temp_uint[0]

    cdef rejection_methods rejector
    if reject_method == 'sigclip':
        rejector = SIGCLIP
    elif reject_method == 'varclip':
        rejector = VARCLIP
    elif reject_method == 'none':
        rejector = NONE
    else:
        raise ValueError(f'unknow rejection method: {reject_method}')

    cdef combine_methods combiner
    if combine_method == 'mean':
        combiner = MEAN
    elif combine_method == 'median':
        combiner = MEDIAN
    elif combine_method == 'sum':
        combiner = SUM
    else:
        raise ValueError(f'unknow combination method {combine_method}')

    outarr = np.zeros(npix, dtype=np.float64, order='C')
    outmaskarr = np.zeros((npoints, npix), dtype=np.uint16, order='C')

    if use_variance:
        outvararr = np.zeros(npix, dtype=np.float64, order='C')
    else:
        outvararr = None

    cdef double [:] outdata = outarr
    cdef double [:] outvar = outvararr
    cdef unsigned short [:,:] outmask = outmaskarr

    with nogil, parallel(num_threads=num_threads):
        tmpdata = <float *> malloc(npoints * sizeof(float))
        tmpvar = <float *> malloc(npoints * sizeof(float))
        tmpmask = <unsigned short *> malloc(npoints * sizeof(unsigned short))

        for i in prange(npix):
            for j in range(npoints):
                tmpdata[j] = data[j][i]
                tmpmask[j] = mask[j][i]
            if use_variance:
                for j in range(npoints):
                    tmpvar[j] = var[j][i]

            #print('- iter ', i)
            #print('  data:', np.asarray(<float[:npoints]>tmpdata))
            #print('  mask:', np.asarray(<unsigned short[:npoints]>tmpmask))

            if rejector == SIGCLIP or rejector == VARCLIP:
                cy_sigma_clip(tmpdata, tmpvar, tmpmask, npoints, lsigma, hsigma,
                              use_variance, max_iters, 1, rejector == VARCLIP, 0)

            #print('  rejm:', np.asarray(<unsigned short[:npoints]>tmpmask))

            if combiner == MEAN:
                outdata[i] = compute_mean(tmpdata, tmpmask, npoints)
                if use_variance:
                    outvar[i] = compute_mean_var(tmpvar, tmpmask, npoints)

            elif combiner == SUM:
                outdata[i] = compute_sum(tmpdata, tmpmask, npoints)
                if use_variance:
                    outvar[i] = compute_sum(tmpvar, tmpmask, npoints)

            elif combiner == MEDIAN:
                outdata[i] = compute_median(tmpdata, tmpmask, npoints)
                # According to Laplace, the uncertainty on the median is
                # sqrt(2/pi) times greater than that on the mean
                if use_variance:
                    outvar[i] = M_PI_2 * compute_mean_var(tmpvar, tmpmask, npoints)

            for j in range(npoints):
                outmask[j, i] = tmpmask[j]

        free(tmpdata)
        free(tmpmask)
        free(tmpvar)

    return outarr, outvararr, outmaskarr


def sigma_clip(float [:] data,
               float [:] variance=None,
               unsigned short [:] mask=None,
               double lsigma=3,
               double hsigma=3,
               int max_iters=10,
               int use_median=1,
               int use_variance=0,
               int use_mad=0):
    """
    Iterative sigma-clipping.

    Parameters
    ----------
    data : float array
        Input data.
    variance : float array
        Array of variances. If provided and use_variance=True, those values
        will be used instead of computing the std from the data values.
    mask : unsigned short array
        Input mask.
    lsigma : double
        Number of standard deviations for clipping below the mean.
    hsigma : double
        Number of standard deviations for clipping above the mean.
    max_iters : int
        Maximum number of iterations to compute
    use_median : int
        Clip around the median rather than mean?
    use_variance : int
        Perform sigma-clipping using the pixel-to-pixel scatter, rather than
        use the variance array?

    Returns
    -------
    mask : uint16 array
        Output mask array.

    """

    outmask = np.zeros(data.shape[0], dtype=np.uint16, order='C')
    cdef unsigned short [:] outmask_view = outmask

    cdef int has_var=0
    cdef float* cvar=NULL

    if variance is not None:
        has_var = 1
        cvar = &variance[0]

    if mask is not None:
        outmask[:] = mask

    cy_sigma_clip(&data[0],
                  cvar,
                  &outmask_view[0],
                  data.shape[0],
                  lsigma,
                  hsigma,
                  has_var,
                  max_iters,
                  use_median,
                  use_variance,
                  use_mad)

    return np.asarray(outmask)
