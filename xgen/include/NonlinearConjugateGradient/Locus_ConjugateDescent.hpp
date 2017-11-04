/*
 * Locus_ConjugateDescent.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CONJUGATEDESCENT_HPP_
#define LOCUS_CONJUGATEDESCENT_HPP_

#include <cmath>
#include <limits>
#include <memory>

#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_NonlinearConjugateGradientStep.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class ConjugateDescent : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    ConjugateDescent(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mPreviousStep(1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~ConjugateDescent()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tCurrentGradDotCurrentGrad = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tNormCurrentGradient = std::sqrt(tCurrentGradDotCurrentGrad);
        ScalarType tPreviousGradDotPreviousGrad = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradDotPreviousGrad);

        ScalarType tDenominator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentSteepestDescent()) / tNormCurrentGradient;
        ScalarType tNumerator = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousSteepestDescent()) / tNormPreviousGradient;

        ScalarType tBeta = mPreviousStep * (tNumerator / tDenominator);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(tBeta,
                      aDataMng.getCurrentSteepestDescent(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());

        mPreviousStep = tBeta;
        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mPreviousStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    ConjugateDescent(const locus::ConjugateDescent<ScalarType, OrdinalType> & aRhs);
    locus::ConjugateDescent<ScalarType, OrdinalType> & operator=(const locus::ConjugateDescent<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_CONJUGATEDESCENT_HPP_ */
