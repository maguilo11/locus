/*
 * Locus_Daniels.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DANIELS_HPP_
#define LOCUS_DANIELS_HPP_

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
class Daniels : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    Daniels(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mHessianTimesVector(aDataFactory.control().create()),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~Daniels()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        aStageMng.applyVectorToHessian(aDataMng.getCurrentControl(),
                                       aDataMng.getTrialStep(),
                                       mHessianTimesVector.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), mHessianTimesVector.operator*());
        ScalarType tDenominator = locus::dot(aDataMng.getTrialStep(), mHessianTimesVector.operator*());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mHessianTimesVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    Daniels(const locus::Daniels<ScalarType, OrdinalType> & aRhs);
    locus::Daniels<ScalarType, OrdinalType> & operator=(const locus::Daniels<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_DANIELS_HPP_ */
