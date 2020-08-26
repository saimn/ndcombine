# cython: language_level=3

cdef void cy_sigma_clip(float data [],
                        float variance [],
                        unsigned short mask [],
                        size_t npoints,
                        double lsigma,
                        double hsigma,
                        int has_var,
                        size_t max_iters,
                        int use_median,
                        int use_variance,
                        int use_mad) nogil
