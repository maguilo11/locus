/*
 * Locus_HagerZhang.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_HAGERZHANG_HPP_
#define LOCUS_HAGERZHANG_HPP_

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
class HagerZhang : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    HagerZhang(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mLowerBound(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~HagerZhang()
    {
    }

    void setLowerBound(const ScalarType & aInput)
    {
        mLowerBound = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType DeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;
        ScalarType tScaleFactor = static_cast<ScalarType>(2) * DeltaGradientDotDeltaGradient
                * tOneOverTrialStepDotDeltaGradient;

        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (tScaleFactor * tTrialStepDotCurrentGradient));
        ScalarType tTrialStepDotTrialStep = locus::dot(aDataMng.getTrialStep(), aDataMng.getTrialStep());
        ScalarType tNormTrialStep = std::sqrt(tTrialStepDotTrialStep);
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradientDotPreviousGradient);
        ScalarType tLowerBound = static_cast<ScalarType>(-1)
                / (tNormTrialStep * std::min(tNormPreviousGradient, mLowerBound));
        tBeta = std::max(tBeta, tLowerBound);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mLowerBound;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    HagerZhang(const locus::HagerZhang<ScalarType, OrdinalType> & aRhs);
    locus::HagerZhang<ScalarType, OrdinalType> & operator=(const locus::HagerZhang<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_HAGERZHANG_HPP_ */
