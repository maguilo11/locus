/*
 * Locus_NonlinearConjugateGradientStageMngBase.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENTSTAGEMNGBASE_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENTSTAGEMNGBASE_HPP_

#include <limits>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;
template<typename ScalarType, typename OrdinalType>
class NonlinearConjugateGradientDataMng;

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStageMngBase
{
public:
    virtual ~NonlinearConjugateGradientStageMngBase()
    {
    }

    virtual void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                         ScalarType aTolerance = std::numeric_limits<ScalarType>::max()) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

} // namespace locus

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENTSTAGEMNGBASE_HPP_ */
