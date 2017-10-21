/*
 * Locus_AnalyticalHessian.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_ANALYTICALHESSIAN_HPP_
#define LOCUS_ANALYTICALHESSIAN_HPP_

#include <memory>

#include "Locus_Criterion.hpp"
#include "Locus_StateData.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_LinearOperator.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class AnalyticalHessian : public locus::LinearOperator<ScalarType, OrdinalType>
{
public:
    explicit AnalyticalHessian(const locus::Criterion<ScalarType, OrdinalType> & aCriterion) :
            mCriterion(aCriterion.create())
    {
    }
    explicit AnalyticalHessian(const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & aCriterion) :
            mCriterion(aCriterion)
    {
    }
    virtual ~AnalyticalHessian()
    {
    }

    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void apply(const locus::MultiVector<ScalarType, OrdinalType> & aState,
               const locus::MultiVector<ScalarType, OrdinalType> & aControl,
               const locus::MultiVector<ScalarType, OrdinalType> & aVector,
               locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mCriterion->hessian(aState, aControl, aVector, aOutput);
    }
    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::AnalyticalHessian<ScalarType, OrdinalType>>(mCriterion);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mCriterion;

private:
    AnalyticalHessian(const locus::AnalyticalHessian<ScalarType, OrdinalType> & aRhs);
    locus::AnalyticalHessian<ScalarType, OrdinalType> & operator=(const locus::AnalyticalHessian<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_ANALYTICALHESSIAN_HPP_ */
