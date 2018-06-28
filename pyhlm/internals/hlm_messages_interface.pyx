import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t INT_t

def internal_messages_forwards_log(
        np.ndarray[DOUBLE_t, ndim=2, mode="c"] aBl not None,
        np.ndarray[DOUBLE_t, ndim=2, mode="c"] alDl not None,
        np.ndarray[INT_t, ndim=1, mode="c"] word not None,
        np.ndarray[DOUBLE_t,ndim=2,mode="c"] alphal not None):

    cdef int j, t
    cdef int T = alphal.shape[0]
    cdef int L = alphal.shape[1]

    alphal[:] = -np.inf

    if T-L+1 <= 0:
        return alphal[:, -1]

    cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] cumsum_aBl = np.zeros(T-L+1, dtype=np.float64)

    alphal[:T-L+1, 0] = np.cumsum(aBl[:T-L+1, word[0]]) + alDl[:T-L+1, word[0]]

    for j in range(L-1):
        cumsum_aBl[:] = 0.0
        for t in range(T - L + 1):
            cumsum_aBl[:t+1] += aBl[t+j+1, word[j+1]]
            alphal[t+j+1, j+1] = np.logaddexp.reduce(cumsum_aBl[:t+1] + alDl[t::-1, word[j+1]] + alphal[j:t+j+1, j])
    return alphal[:, -1]

def cumulative_internal_message_forwards_log(
        np.ndarray[DOUBLE_t, ndim=2, mode="c"] aBl not None,
        np.ndarray[DOUBLE_t, ndim=2, mode="c"] alDl not None,
        list words not None,
        np.ndarray[DOUBLE_t,ndim=2,mode="c"] cumulative_alphal not None):

        cdef int n
        cdef int T = cumulative_alphal.shape[0]
        cdef int N = len(words)

        cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] alphal

        for n in range(N):
            alphal = np.empty((T, words[n].shape[0]), dtype=np.float64)
            cumulative_alphal[:, n] = internal_messages_forwards_log(aBl, alDl, words[n], alphal)

        return cumulative_alphal
