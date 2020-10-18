# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

from ndcombine.sigma_clip cimport cy_sigma_clip
from ndcombine.utils cimport compute_mean, compute_median, compute_mean_var

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
def ndcombine(float [:,:] data,
              unsigned short [:,:] mask,
              float [:,:] variance=None,
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

    cdef combine_methods combiner
    if combine_method == 'mean':
        combiner = MEAN
    elif combine_method == 'median':
        combiner = MEDIAN
    else:
        raise ValueError

    outarr = np.zeros(npix, dtype=np.float32, order='C')
    outmaskarr = np.zeros((npoints, npix), dtype=np.uint16, order='C')

    if variance is not None:
        outvararr = np.zeros(npix, dtype=np.float32, order='C')
    else:
        outvararr = None

    cdef float [:] outdata = outarr
    cdef float [:] outvar = outvararr
    cdef unsigned short [:,:] outmask = outmaskarr

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

            if combiner == MEAN:
                outdata[i] = compute_mean(tmpdata, tmpmask, npoints)
                if variance is not None:
                    for j in range(npoints):
                        tmpvar[j] = variance[j, i]
                    outvar[i] = compute_mean_var(tmpvar, tmpmask, npoints)

            elif combiner == MEDIAN:
                outdata[i] = compute_median(tmpdata, tmpmask, npoints)

            for j in range(npoints):
                outmask[j, i] = tmpmask[j]

        free(tmpdata)
        free(tmpmask)
        free(tmpvar)

    return outarr, outvararr, outmaskarr
