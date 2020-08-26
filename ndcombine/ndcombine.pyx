# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

from ndcombine.sigma_clip cimport cy_sigma_clip
from ndcombine.utils cimport compute_mean, compute_median

ctypedef unsigned short mask_t

cdef enum rejection_methods:
    SIGCLIP,
    MINMAX,
    NONE


@cython.boundscheck(False)
@cython.wraparound(False)
def ndcombine(float [:,:] data,
              #float [:,:] variance,
              unsigned short [:,:] mask,
              combine_method='mean',
              reject_method='none',
              int num_threads=0):

    cdef size_t npoints = data.shape[0]
    cdef size_t npix = data.shape[1]
    cdef size_t i, j

    cdef float *tmpdata
    cdef float *tmpvar
    cdef unsigned short *tmpmask

    cdef rejection_methods rejector
    if reject_method == 'sigclip':
        rejector = SIGCLIP
    elif reject_method == 'none':
        rejector = NONE
    else:
        raise ValueError

    outarr = np.zeros(npix, dtype=np.float32, order='C')
    outmaskarr = np.zeros((npoints, npix), dtype=np.uint16, order='C')

    cdef float [:] outdata = outarr
    cdef unsigned short [:,:] outmask = outmaskarr

    if combine_method == 'mean':
        combine_func = compute_mean
    elif combine_method == 'median':
        combine_func = compute_median
    else:
        raise ValueError

    with nogil, parallel(num_threads=num_threads):
        tmpdata = <float *> malloc(npoints * sizeof(float))
        tmpvar = <float *> malloc(npoints * sizeof(float))
        tmpmask = <unsigned short *> malloc(npoints * sizeof(unsigned short))

        for i in prange(npix):
            for j in range(npoints):
                tmpdata[j] = data[j, i]
                tmpmask[j] = mask[j, i]

            #print('- iter ', i)
            #print('  data:', np.asarray(<float[:npoints]>tmpdata))
            #print('  mask:', np.asarray(<unsigned short[:npoints]>tmpmask))

            if rejector == SIGCLIP:
                cy_sigma_clip(tmpdata, tmpvar, tmpmask, npoints, 3, 3, 0, 10, 1, 0, 0)

            #print('  rejm:', np.asarray(<unsigned short[:npoints]>tmpmask))

            outdata[i] = combine_func(tmpdata, tmpmask, npoints)
            for j in range(npoints):
                outmask[j, i] = tmpmask[j]

        free(tmpdata)
        free(tmpmask)
        free(tmpvar)

    return outarr, outmaskarr
