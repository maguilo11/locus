/*
 * Locus_OptimalityCriteriaStageMng.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIASTAGEMNG_HPP_
#define LOCUS_OPTIMALITYCRITERIASTAGEMNG_HPP_

#include <memory>

#include "Locus_Criterion.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_OptimalityCriteriaDataMng.hpp"
#include "Locus_OptimalityCriteriaStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaStageMng : public locus::OptimalityCriteriaStageMngBase<ScalarType, OrdinalType>
{
public:
    OptimalityCriteriaStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory,
                               const locus::Criterion<ScalarType, OrdinalType> & aObjective,
                               const locus::CriterionList<ScalarType, OrdinalType> & aInequality) :
            mCurrentGradient(aDataFactory.control().create()),
            mObjective(aObjective.create()),
            mConstraint(aInequality.create())
    {
    }
    virtual ~OptimalityCriteriaStageMng()
    {
    }

    void update(locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = aDataMng.getCurrentControl();
        ScalarType tObjectiveValue = mObjective->value(tControl);
        aDataMng.setCurrentObjectiveValue(tObjectiveValue);

        locus::fill(static_cast<ScalarType>(0), mCurrentGradient.operator*());
        mObjective->gradient(tControl, mCurrentGradient.operator*());
        aDataMng.setObjectiveGradient(mCurrentGradient.operator*());

        this->computeConstraintQuantities(aDataMng);
    }
    void computeConstraintQuantities(locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        locus::fill(static_cast<ScalarType>(0), mCurrentGradient.operator*());

        const OrdinalType tNumConstraints = mConstraint->size();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = aDataMng.getCurrentControl();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            ScalarType tMyConstraintValue = mConstraint->operator[](tConstraintIndex).value(tControl);
            aDataMng.setCurrentConstraintValue(tConstraintIndex, tMyConstraintValue);
            mConstraint->operator[](tConstraintIndex).gradient(tControl, mCurrentGradient.operator*());
        }

        aDataMng.setInequalityGradient(mCurrentGradient.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentGradient;

    std::shared_ptr<locus::Criterion<ScalarType,OrdinalType>> mObjective;
    std::shared_ptr<locus::CriterionList<ScalarType,OrdinalType>> mConstraint;

private:
    OptimalityCriteriaStageMng(const locus::OptimalityCriteriaStageMng<ScalarType, OrdinalType>&);
    locus::OptimalityCriteriaStageMng<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteriaStageMng<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIASTAGEMNG_HPP_ */
