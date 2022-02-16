# cython: boundscheck=False, nonecheck=False, wraparound=False, language_level=3, cdivision=True

cimport cython
from libc.math cimport sqrt, isnan, NAN
from libc.stdlib cimport malloc, free
# from libc.stdio cimport printf
# from cython cimport floating

#cdef extern from "math.h":
#    bint isnan(double x)

#cdef extern from "numpy/npy_math.h" nogil:
#    long double NAN "NPY_NAN"
#    bint isnan "npy_isnan"(long double)


cdef double compute_median(float data[], size_t data_size) nogil:
    """
    One-dimensional true median, with optional masking.
    From https://github.com/GeminiDRSoftware/DRAGONS/blob/master/gempy/library/cython_utils.pyx
    """
    cdef:
        size_t i, j, k, l, m
        int ncycles, cycle
        double x, y, med=0.

    ncycles = 2 - data_size % 2
    for cycle in range(0, ncycles):
        k = (data_size - 1) // 2 + cycle
        l = 0
        m = data_size - 1
        while (l < m):
            x = data[k]
            i = l
            j = m
            while True:
                while (data[i] < x):
                    i += 1
                while (x < data[j]):
                    j -= 1
                if i <= j:
                    y = data[i]
                    data[i] = data[j]
                    data[j] = y
                    i += 1
                    j -= 1
                if i > j:
                    break
            if j < k:
                l = i
            if k < i:
                m = j
        if cycle == 0:
            med = data[k]
        else:
            med = 0.5 * (med + data[k])

    return med


cdef void compute_mean_std(float data[],
                           double result[2],
                           int use_median,
                           size_t data_size) nogil:

    cdef:
        double mean, sum = 0, sumsq = 0
        size_t i

    for i in range(data_size):
        sum += data[i]
        sumsq += data[i] * data[i]

    mean = sum / data_size
    if use_median:
        result[0] = <double>compute_median(data, data_size)
    else:
        result[0] = mean

    result[1] = sqrt(sumsq / data_size - mean*mean)


cdef inline double compute_sum(const float data[], size_t data_size) nogil:
    cdef double m = 0
    for i in range(data_size):
        m += <double>data[i]
    return m


cdef size_t cy_sigma_clip(float data [],
                          const float variance [],
                          size_t data_size,
                          double lsigma,
                          double hsigma,
                          int has_var,
                          size_t max_iters,
                          int use_median,
                          int use_variance,
                          int use_mad) nogil:

    cdef:
        size_t i, ngood=data_size, nused, niter=0
        double avg, var, std, low_limit, high_limit
        double result[2]

    if use_mad: # TODO
        pass

    while niter < max_iters:

        compute_mean_std(data, result, use_median, ngood)
        avg = result[0]
        nused = 0

        if has_var and use_variance:
            # use the provided variance
            for i in range(ngood):
                std = sqrt(variance[i])
                low_limit = avg - lsigma * std
                high_limit = avg + hsigma * std

                if data[i] >= low_limit and data[i] <= high_limit:
                    data[nused] = data[i]
                    nused += 1
        else:
            # use std computed from the data values
            std = result[1]
            low_limit = avg - lsigma * std
            high_limit = avg + hsigma * std

            for i in range(ngood):
                if data[i] >= low_limit and data[i] <= high_limit:
                    data[nused] = data[i]
                    nused += 1

        if nused == ngood:
            break

        ngood = nused
        niter += 1

    return ngood
