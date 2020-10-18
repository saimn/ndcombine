# cython: language_level=3
#import numpy as np
#cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.math cimport isnan, NAN
from libc.stdlib cimport malloc, free
#from cython cimport floating

#cdef extern from "math.h":
#    bint isnan(double x)

#cdef extern from "numpy/npy_math.h" nogil:
#    long double NAN "NPY_NAN"
#    bint isnan "npy_isnan"(long double)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float compute_median(const float data[],
                          const unsigned short mask[],
                          size_t data_size) nogil:
    """
    One-dimensional true median, with optional masking.
    """
    cdef size_t i, j, k, l, m, nused=0
    cdef int ncycles, cycle
    cdef float x, y, med=0.
    cdef float *tmp = <float *> malloc(data_size * sizeof(float))

    for i in range(data_size):
        if mask[i] == 0:
            tmp[nused] = data[i]
            nused += 1

    if nused == 0:
        for i in range(data_size):
            tmp[i] = data[i]
        nused = data_size

    ncycles = 2 - nused % 2
    for cycle in range(0, ncycles):
        k = (nused - 1) // 2 + cycle
        l = 0
        m = nused - 1
        while (l < m):
            x = tmp[k]
            i = l
            j = m
            while True:
                while (tmp[i] < x):
                    i += 1
                while (x < tmp[j]):
                    j -= 1
                if i <= j:
                    y = tmp[i]
                    tmp[i] = tmp[j]
                    tmp[j] = y
                    i += 1
                    j -= 1
                if i > j:
                    break
            if j < k:
                l = i
            if k < i:
                m = j
        if cycle == 0:
            med = tmp[k]
        else:
            med = 0.5 * (med + tmp[k])

    free(tmp)
    return med


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float compute_mean_var(const float data[],
                            const unsigned short mask[],
                            size_t data_size) nogil:
    cdef float m = 0
    cdef size_t count = 0
    for i in range(data_size):
        if mask[i] == 0:
            count += 1
            m += data[i]
    if count > 0:
        return m / (count * count)
    else:
        return NAN


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float compute_mean(const float data[],
                        const unsigned short mask[],
                        size_t data_size) nogil:
    cdef float m = 0
    cdef size_t count = 0
    for i in range(data_size):
        if mask[i] == 0:
            count += 1
            m += data[i]
    if count > 0:
        return m / count
    else:
        return NAN


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_mean_std(const float data[],
                           const unsigned short mask[],
                           double result[2],
                           int use_median,
                           size_t data_size) nogil:

    cdef double mean, sum = 0, sumsq = 0
    cdef size_t i, count = 0

    for i in range(data_size):
        if mask[i] == 0:
            sum += data[i]
            sumsq += data[i] * data[i]
            count += 1

    if count > 0:
        mean = sum / count
        if use_median:
            result[0] = <double>compute_median(data, mask, data_size)
        else:
            result[0] = mean

        result[1] = sqrt(sumsq / count - mean*mean)
    else:
        result[0] = NAN
        result[1] = NAN
