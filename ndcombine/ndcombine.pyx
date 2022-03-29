# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport M_PI_2, NAN
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from ndcombine.utils cimport compute_median, compute_sum, cy_sigma_clip

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
        ssize_t i, j, nvalid

        int use_variance = 1 if list_of_var is not None else 0

        float *tmpdata
        float *tmpvar

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
        raise ValueError(f'unknow combination method: {combine_method}')

    outarr = np.zeros(npix, dtype=np.float64, order='C')
    outmaskarr = np.zeros(npix, dtype=np.uint16, order='C')

    if use_variance:
        outvararr = np.zeros(npix, dtype=np.float64, order='C')
    else:
        outvararr = None

    cdef double [:] outdata = outarr
    cdef double [:] outvar = outvararr
    cdef unsigned short [:] outmask = outmaskarr

    with nogil, parallel(num_threads=num_threads):
        tmpdata = <float *> malloc(npoints * sizeof(float))
        tmpvar = <float *> malloc(npoints * sizeof(float))

        for i in prange(npix):
            nvalid = 0
            for j in range(npoints):
                if mask[j][i] == 0:
                    tmpdata[nvalid] = data[j][i]
                    if use_variance:
                        tmpvar[nvalid] = var[j][i]
                    nvalid = nvalid + 1

            # printf('- pix %ld:\n', i)
            # printf('  %ld values: ', nvalid)
            # for j in range(nvalid):
            #     printf('%.2f ', tmpdata[j])
            # printf('\n')
            # # printf('  data:', np.asarray(<float[:nvalid]>tmpdata))

            if rejector == SIGCLIP or rejector == VARCLIP:
                nvalid = cy_sigma_clip(tmpdata, tmpvar, nvalid, lsigma, hsigma,
                                       use_variance, max_iters, 1, rejector == VARCLIP, 0)

            # printf('  %ld values: ', nvalid)
            # for j in range(nvalid):
            #     printf('%.2f ', tmpdata[j])
            # printf('\n')

            if nvalid == 0:
                outdata[i] = NAN

            if combiner == MEAN:
                outdata[i] = compute_sum(tmpdata, nvalid) / nvalid
                if use_variance:
                    outvar[i] = compute_sum(tmpvar, nvalid) / (nvalid * nvalid)

            elif combiner == SUM:
                outdata[i] = compute_sum(tmpdata, nvalid)
                if use_variance:
                    outvar[i] = compute_sum(tmpvar, nvalid)

            elif combiner == MEDIAN:
                outdata[i] = compute_median(tmpdata, nvalid)
                # According to Laplace, the uncertainty on the median is
                # sqrt(2/pi) times greater than that on the mean
                if use_variance:
                    outvar[i] = M_PI_2 * compute_sum(tmpvar, nvalid) / (nvalid * nvalid)

            outmask[i] = nvalid

        free(tmpdata)
        free(tmpvar)

    return outarr, outvararr, outmaskarr
