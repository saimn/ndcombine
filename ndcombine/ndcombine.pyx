# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

from ndcombine.sigma_clip cimport cy_sigma_clip
from ndcombine.utils cimport compute_mean, compute_median, compute_mean_var

np.import_array()

ctypedef unsigned short mask_t

cdef enum rejection_methods:
    SIGCLIP,
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
    elif reject_method == 'none':
        rejector = NONE
    else:
        raise ValueError

    cdef combine_methods combiner
    if combine_method == 'mean':
        combiner = MEAN
    elif combine_method == 'median':
        combiner = MEDIAN
    else:
        raise ValueError

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

            #print('- iter ', i)
            #print('  data:', np.asarray(<float[:npoints]>tmpdata))
            #print('  mask:', np.asarray(<unsigned short[:npoints]>tmpmask))

            if rejector == SIGCLIP:
                cy_sigma_clip(tmpdata, tmpvar, tmpmask, npoints, lsigma, hsigma,
                              0, max_iters, 1, 0, 0)

            #print('  rejm:', np.asarray(<unsigned short[:npoints]>tmpmask))

            if combiner == MEAN:
                outdata[i] = compute_mean(tmpdata, tmpmask, npoints)
                if use_variance:
                    for j in range(npoints):
                        tmpvar[j] = var[j][i]
                    outvar[i] = compute_mean_var(tmpvar, tmpmask, npoints)

            elif combiner == MEDIAN:
                outdata[i] = compute_median(tmpdata, tmpmask, npoints)

            for j in range(npoints):
                outmask[j, i] = tmpmask[j]

        free(tmpdata)
        free(tmpmask)
        free(tmpvar)

    return outarr, outvararr, outmaskarr
