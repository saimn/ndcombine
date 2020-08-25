# cython: language_level=3
cdef float compute_median(const float data[],
                          const unsigned short mask[],
                          size_t data_size) nogil
cdef float compute_mean(const float data[],
                        const unsigned short mask[],
                        size_t data_size) nogil
cdef void compute_mean_std(const float data[],
                           const unsigned short mask[],
                           double result[2],
                           int use_median,
                           size_t data_size) nogil
