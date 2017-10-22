/*
 * Locus_DaiLiao.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DAILIAO_HPP_
#define LOCUS_DAILIAO_HPP_

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
class DaiLiao : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiLiao(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaleFactor(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiLiao()
    {
    }

    void setScaleFactor(const ScalarType & aInput)
    {
        mScaleFactor = aInput;
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
        ScalarType tCurrentControlDotCurrentGradient = locus::dot(aDataMng.getCurrentControl(), aDataMng.getCurrentGradient());
        ScalarType tPreviousControlDotCurrentGradient = locus::dot(aDataMng.getPreviousControl(), aDataMng.getCurrentGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType tDeltaControlDotCurrentGradient = tCurrentControlDotCurrentGradient
                - tPreviousControlDotCurrentGradient;

        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (mScaleFactor * tDeltaControlDotCurrentGradient));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mScaleFactor;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiLiao(const locus::DaiLiao<ScalarType, OrdinalType> & aRhs);
    locus::DaiLiao<ScalarType, OrdinalType> & operator=(const locus::DaiLiao<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_DAILIAO_HPP_ */
