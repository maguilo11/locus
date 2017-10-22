/*
 * Locus_PerryShanno.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_PERRYSHANNO_HPP_
#define LOCUS_PERRYSHANNO_HPP_

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
class PerryShanno : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    PerryShanno(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mLowerBound(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~PerryShanno()
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

        ScalarType tBeta = this->computeBeta(aDataMng);
        ScalarType tAlpha = this->computeAlpha(aDataMng);
        ScalarType tTheta = this->computeTheta(aDataMng);

        locus::scale(tBeta, mScaledDescentDirection.operator*());
        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::update(tAlpha,
                      aDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::update(-tAlpha,
                      aDataMng.getPreviousGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::scale(tTheta, mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType computeBeta(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType tDeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;

        ScalarType tScaleFactor = static_cast<ScalarType>(2) * tDeltaGradientDotDeltaGradient
                * tOneOverTrialStepDotDeltaGradient;
        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (tScaleFactor * tTrialStepDotCurrentGradient));
        ScalarType tNormTrialStep = locus::dot(aDataMng.getTrialStep(), aDataMng.getTrialStep());
        tNormTrialStep = std::sqrt(tNormTrialStep);
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradientDotPreviousGradient);
        ScalarType tLowerBound = static_cast<ScalarType>(-1)
                / (tNormTrialStep * std::min(tNormPreviousGradient, mLowerBound));

        tBeta = std::max(tBeta, tLowerBound);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());
        return (tBeta);
    }
    ScalarType computeAlpha(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tAlpha = tTrialStepDotCurrentGradient / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        tAlpha = std::max(tAlpha, std::numeric_limits<ScalarType>::min());
        return (tAlpha);
    }
    ScalarType computeTheta(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tCurrentGradientDotCurrentControl = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentControl());
        ScalarType tPreviousGradientDotCurrentControl = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentControl());
        ScalarType tCurrentGradientDotPreviousControl = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousControl());
        ScalarType tPreviousGradientDotPreviousControl = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousControl());

        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tDeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;
        ScalarType tDeltaGradientDotDeltaControl = tCurrentGradientDotCurrentControl
                - tPreviousGradientDotCurrentControl - tCurrentGradientDotPreviousControl
                + tPreviousGradientDotPreviousControl;

        ScalarType tTheta = tDeltaGradientDotDeltaControl / tDeltaGradientDotDeltaGradient;
        tTheta = std::max(tTheta, std::numeric_limits<ScalarType>::min());
        return (tTheta);
    }

private:
    ScalarType mLowerBound;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    PerryShanno(const locus::PerryShanno<ScalarType, OrdinalType> & aRhs);
    locus::PerryShanno<ScalarType, OrdinalType> & operator=(const locus::PerryShanno<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_PERRYSHANNO_HPP_ */
