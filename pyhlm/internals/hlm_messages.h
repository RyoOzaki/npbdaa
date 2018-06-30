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
    void internal_messages_forwards_log(
      int T, int L, int P, Type *aBl, Type* alDl, int word[],
      Type *alphal)
    {
      // T: Length of observations.
      // P: Number of phonemes in model. (Number of upper limit of phonemes.)
      // L: Length of the word. (Number of letters in word.)
      NPArray<Type> eaBl(aBl, T, P);
      NPArray<Type> ealDl(alDl, T, P);

      NPArray<Type> ealphal(alphal, T, L);

      ealphal.col(L-1).setConstant(-1.0*numeric_limits<Type>::infinity());

#ifdef HLM_TEMPS_ON_HEAP
        Array<Type,1,Dynamic> sumsofar(T-L+1);
        Array<Type,1,Dynamic> result(T-L+1);
#else
        Type sumsofar_buf[T-L+1] __attribute__((aligned(16)));
        NPRowVectorArray<Type> sumsofar(sumsofar_buf,T-L+1);
        Type result_buf[T-L+1] __attribute__((aligned(16)));
        NPRowVectorArray<Type> result(result_buf,T-L+1);
#endif

      //calculate cumsum vector.
      sumsofar.setZero();
      for(int t=0; t<T-L+1; t++){
        sumsofar.tail(T-L+1-t) += eaBl(t, word[0]);
      }
      ealphal.col(0).head(T-L+1) = sumsofar + ealDl.col(word[0]).head(T-L+1);

      Type cmax;
      for(int j=0; j<L-1; j++){
        sumsofar.setZero();
        ealphal.col(j+1).setConstant(-1.0*numeric_limits<Type>::infinity());
        for(int t=0; t<T-L+1; t++){
          sumsofar.head(t+1) += eaBl(t+j+1, word[j+1]);
          result.head(t+1) = sumsofar.head(t+1) + ealDl.col(word[j+1]).head(t+1).reverse() + ealphal.col(j).segment(j, t+j);
          cmax = result.head(t+1).maxCoeff();
          ealphal(t+j+1, j+1) = log((result.head(t+1) - cmax).exp().sum()) + cmax;
        }
      }
    }

}

// NOTE: this class exists for cyhton binding convenience

template <typename FloatType, typename IntType = int32_t>
class hlmc
{
    public:

    static void internal_messages_forwards_log(
      int T, int L, int P, FloatType *aBl, FloatType *alDl, int word[],
      FloatType *alphal)
    { hlm::internal_messages_forwards_log(T, L, P, aBl, alDl, word, alphal); }

};

#endif
