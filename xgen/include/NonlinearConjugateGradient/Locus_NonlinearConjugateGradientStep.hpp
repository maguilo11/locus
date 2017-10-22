/*
 * Locus_NonlinearConjugateGradientStep.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENTSTEP_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENTSTEP_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class NonlinearConjugateGradientDataMng;
template<typename ScalarType, typename OrdinalType>
class NonlinearConjugateGradientStageMngBase;

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStep
{
public:
    virtual ~NonlinearConjugateGradientStep()
    {
    }

    virtual void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                               locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng) = 0;
};

} // namespace locus

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENTSTEP_HPP_ */
