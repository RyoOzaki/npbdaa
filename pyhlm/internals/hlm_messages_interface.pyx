# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -O3 -w -DNDEBUG -DHLM_TEMPS_ON_HEAP
# distutils: include_dirs = deps/
# cython: boundscheck = False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from libcpp.vector cimport vector

# NOTE: using the cython.floating fused type with typed memory views generates
# all possible type combinations, is not intended here.
# https://groups.google.com/forum/#!topic/cython-users/zdlliIRF1a4

from cython cimport floating

from cython.parallel import prange

cdef extern from "hlm_messages.h":
    cdef cppclass hlmc[Type]:
        hlmc()
        void internal_messages_forwards_log(
            int T, int L, int P, Type *aBl, Type* alDl, int word[],
            Type *alphal) nogil
        void messages_backwards_log(
            int M, int T, Type *A, Type *aBl, Type *aDl, Type *aDsl,
            Type *betal, Type *betastarl, int right_censoring, int trunc) nogil
        void sample_forwards_log(
            int M, int T, Type *A, Type *pi0, Type *aBl, Type *aD,
            Type *betal, Type *betastarl, int32_t *stateseq, Type *randseq) nogil

#int T, int L, int P, Type *aBl, Type* alDl, int word[], Type *alphal)
def internal_messages_forwards_log(
        floating[:,::1] aBl not None,
        floating[:,::1] alDl not None,
        int[::1] word,
        np.ndarray[floating,ndim=2,mode="c"] alphal not None):
    cdef hlmc[floating] ref

    ref.internal_messages_forwards_log(alphal.shape[0], alphal.shape[1], aBl.shape[1],
            &aBl[0,0], &alDl[0,0], &word[0], &alphal[0,0])

    return alphal[:, -1]

def messages_backwards_log(
        floating[:,::1] A not None,
        floating[:,::1] aBl not None,
        floating[:,::1] aDl not None,
        floating[:,::1] aDsl not None,
        np.ndarray[floating,ndim=2,mode="c"] betal not None,
        np.ndarray[floating,ndim=2,mode="c"] betastarl not None,
        int right_censoring, int trunc):
    cdef hlmc[floating] ref

    ref.messages_backwards_log(A.shape[0],aBl.shape[0],&A[0,0],
            &aBl[0,0],&aDl[0,0],&aDsl[0,0],&betal[0,0],&betastarl[0,0],
            right_censoring,trunc)

    return betal, betastarl

def sample_forwards_log(
        floating[:,::1] A not None,
        floating[:,::1] caBl not None,
        floating[:,::1] aDl not None,
        floating[::1] pi0 not None,
        floating[:,::1] betal not None,
        floating[:,::1] betastarl not None,
        int32_t[::1] stateseq not None,
        ):
    cdef hlmc[floating] ref

    # NOTE: not all of randseq will be consumed; one entry is consumed for each
    # duration and one for each transition, so they can only all be used if each
    # duration is deterministically 1
    cdef floating[:] randseq
    if floating is double:
        randseq = np.random.random(size=2*caBl.shape[0]).astype(np.double)
    else:
        randseq = np.random.random(size=2*caBl.shape[0]).astype(np.float)

    ref.sample_forwards_log(A.shape[0],caBl.shape[0],&A[0,0],&pi0[0],
            &caBl[0,0],&aDl[0,0],&betal[0,0],&betastarl[0,0],&stateseq[0],&randseq[0])

    return np.asarray(stateseq)
