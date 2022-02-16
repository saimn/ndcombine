# cython: language_level=3
cdef double compute_median(float data[], size_t data_size) nogil

cdef void compute_mean_std(float data[],
                           double result[2],
                           int use_median,
                           size_t data_size) nogil

cdef double compute_sum(const float data[], size_t data_size) nogil

cdef size_t cy_sigma_clip(float data [],
                          const float variance [],
                          size_t npoints,
                          double lsigma,
                          double hsigma,
                          int has_var,
                          size_t max_iters,
                          int use_median,
                          int use_variance,
                          int use_mad) nogil
