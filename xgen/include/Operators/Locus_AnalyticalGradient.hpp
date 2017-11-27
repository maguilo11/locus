/*
 * Locus_AnalyticalGradient.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_ANALYTICALGRADIENT_HPP_
#define LOCUS_ANALYTICALGRADIENT_HPP_

#include <memory>

#include "Locus_StateData.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_GradientOperator.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class AnalyticalGradient : public locus::GradientOperator<ScalarType, OrdinalType>
{
public:
    explicit AnalyticalGradient(const locus::Criterion<ScalarType, OrdinalType> & aCriterion) :
            mCriterion(aCriterion.create())
    {
    }
    explicit AnalyticalGradient(const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & aCriterion) :
            mCriterion(aCriterion)
    {
    }
    virtual ~AnalyticalGradient()
    {
    }

    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void compute(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mCriterion->gradient(aControl, aOutput);
    }
    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::AnalyticalGradient<ScalarType, OrdinalType>>(mCriterion);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mCriterion;

private:
    AnalyticalGradient(const locus::AnalyticalGradient<ScalarType, OrdinalType> & aRhs);
    locus::AnalyticalGradient<ScalarType, OrdinalType> & operator=(const locus::AnalyticalGradient<ScalarType, OrdinalType> & aRhs);
};

}

#endif /* LOCUS_ANALYTICALGRADIENT_HPP_ */
