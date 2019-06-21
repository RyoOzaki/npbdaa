#ifndef HLM_H
#define HLM_H

#include <Eigen/Core>
#include <iostream> // cout, endl
#include <algorithm> // min
#include <math.h>

#include "util.h"
#include "nptypes.h"

namespace hlm
{
    using namespace std;
    using namespace Eigen;
    using namespace nptypes;

    template <typename Type>
    void internal_hsmm_messages_forwards_log(
      int T, int L, int P, Type *aBl, Type* alDl, int word[],
      Type *alphal)
    {
      // T: Length of observations.
      // P: Number of phonemes in model. (Number of upper limit of phonemes.)
      // L: Length of the word. (Number of letters in word.)
      NPArray<Type> eaBl(aBl, T, P);
      NPArray<Type> ealDl(alDl, T, P);

      NPArray<Type> ealphal(alphal, T, L);

#ifdef HLM_TEMPS_ON_HEAP
        Array<Type,1,Dynamic> sumsofar(T-L+1);
        Array<Type,1,Dynamic> result(T-L+1);
#else
        Type sumsofar_buf[T-L+1] __attribute__((aligned(16)));
        NPRowVectorArray<Type> sumsofar(sumsofar_buf,T-L+1);
        Type result_buf[T-L+1] __attribute__((aligned(16)));
        NPRowVectorArray<Type> result(result_buf,T-L+1);
#endif

      //initialize.
      ealphal.setConstant(-1.0*numeric_limits<Type>::infinity());

      /* Same as follows.
      sumsofar.setZero();
      for(int t=0; t<T-L+1; t++){
        sumsofar.tail(T-L+1-t) += eaBl(t, word[0]);
      }
      ealphal.block(0, 0, T-L+1, 1) = sumsofar.transpose() + ealDl.block(0, word[0], T-L+1, 1);
      */

      Type ctmp = 0.0;
      for(int t=0; t<T-L+1; t++){
        // sumsofar.tail(T-L+1-t) += eaBl(t, word[0]);//Same as follows.
        ctmp = ctmp + eaBl(t, word[0]);
        ealphal(t, 0) = ctmp + ealDl(t, word[0]);
      }

      for(int j=0; j<L-1; j++){
        sumsofar.setZero();
        for(int t=0; t<T-L+1; t++){
          /* Same as follows.
          // sumsofar.head(t+1) += eaBl(t+j+1, word[j+1]);
          // result.head(t+1) = sumsofar.head(t+1) + ealDl.block(0, word[j+1], t+1, 1).transpose().reverse() + ealphal.block(j, j, t+1, 1).transpose();
          */
          for(int tau=0; tau<=t; tau++){
            sumsofar(tau) = sumsofar(tau) + eaBl(t+j+1, word[j+1]);
            result(tau) = sumsofar(tau) + ealDl(t-tau, word[j+1]) + ealphal(j+tau, j);
          }
          ctmp = result.head(t+1).maxCoeff();
          ealphal(t+j+1, j+1) = log((result.head(t+1) - ctmp).exp().sum()) + ctmp;
        }
      }
    }

}

// NOTE: this class exists for cyhton binding convenience

template <typename FloatType, typename IntType = int32_t>
class hlmc
{
    public:

    static void internal_hsmm_messages_forwards_log(
      int T, int L, int P, FloatType *aBl, FloatType *alDl, int word[],
      FloatType *alphal)
    { hlm::internal_hsmm_messages_forwards_log(T, L, P, aBl, alDl, word, alphal); }

};

#endif
