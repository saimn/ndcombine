# cython: language_level=3
cdef double compute_median(const float data[],
                           const unsigned short mask[],
                           size_t data_size) nogil

cdef double compute_mean(const float data[],
                         const unsigned short mask[],
                         size_t data_size) nogil

cdef double compute_mean_var(const float data[],
                             const unsigned short mask[],
                             size_t data_size) nogil

cdef void compute_mean_std(const float data[],
                           const unsigned short mask[],
                           double result[2],
                           int use_median,
                           size_t data_size) nogil

cdef double compute_sum(const float data[],
                        const unsigned short mask[],
                        size_t data_size) nogil

cdef void cy_sigma_clip(const float data [],
                        const float variance [],
                        unsigned short mask [],
                        size_t npoints,
                        double lsigma,
                        double hsigma,
                        int has_var,
                        size_t max_iters,
                        int use_median,
                        int use_variance,
                        int use_mad) nogil
