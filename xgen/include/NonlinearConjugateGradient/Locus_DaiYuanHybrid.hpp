/*
 * Locus_DaiYuanHybrid.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DAIYUANHYBRID_HPP_
#define LOCUS_DAIYUANHYBRID_HPP_

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
class DaiYuanHybrid : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiYuanHybrid(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mWolfeConstant(static_cast<ScalarType>(1) / static_cast<ScalarType>(3)),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiYuanHybrid()
    {
    }

    void setWolfeConstant(const ScalarType & aInput)
    {
        mWolfeConstant = aInput;
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

        ScalarType tHestenesStiefelBeta = (tCurrentGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);

        ScalarType tDaiYuanBeta = tCurrentGradientDotCurrentGradient
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tScaleFactor = (static_cast<ScalarType>(1) - mWolfeConstant)
                / (static_cast<ScalarType>(1) + mWolfeConstant);
        tScaleFactor = static_cast<ScalarType>(-1) * tScaleFactor;
        ScalarType tScaledDaiYuanBeta = tScaleFactor * tDaiYuanBeta;

        ScalarType tBeta = std::max(tScaledDaiYuanBeta, std::min(tDaiYuanBeta, tHestenesStiefelBeta));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mWolfeConstant;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiYuanHybrid(const locus::DaiYuanHybrid<ScalarType, OrdinalType> & aRhs);
    locus::DaiYuanHybrid<ScalarType, OrdinalType> & operator=(const locus::DaiYuanHybrid<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_DAIYUANHYBRID_HPP_ */
