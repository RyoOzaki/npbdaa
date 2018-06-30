# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -O3 -w -DNDEBUG -DHLM_TEMPS_ON_HEAP
# distutils: include_dirs = deps/
# cython: boundscheck = False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t

from cython cimport floating

cdef extern from "hlm_messages.h":
    cdef cppclass hlmc[Type]:
        hlmc()
        void internal_messages_forwards_log(
            int T, int L, int P, Type *aBl, Type *alDl, int[] word,
            Type *alphal) nogil
        void messages_backwards_log(
            int M, int T, Type *A, Type *aBl, Type *aDl, Type *aDsl,
            Type *betal, Type *betastarl, int right_censoring, int trunc) nogil
        void sample_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *aD,
            Type *betal, Type *betastarl, int32_t *stateseq, Type *randseq) nogil


def internal_messages_forwards_log(
        floating[:,::1] aBl not None,
        floating[:,::1] alDl not None,
        int[::1] word not None,
        np.ndarray[floating, ndim=2, mode="c"] alphal not None):

    cdef hlmc[floating] ref

    ref.internal_messages_forwards_log(
        alphal.shape[0], alphal.shape[1], aBl.shape[1],
        &aBl[0, 0], &alDl[0, 0], &word[0], &alphal[0, 0])

    return alphal
