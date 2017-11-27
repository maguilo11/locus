/*
 * Locus_OptimalityCriteriaTestObjectiveOne.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIATESTOBJECTIVEONE_HPP_
#define LOCUS_OPTIMALITYCRITERIATESTOBJECTIVEONE_HPP_

#include <memory>

#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaTestObjectiveOne : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    OptimalityCriteriaTestObjectiveOne() :
            mConstant(0.0624),
            mReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ScalarType,OrdinalType>>())
    {

    }
    explicit OptimalityCriteriaTestObjectiveOne(const locus::ReductionOperations<ScalarType, OrdinalType> & aInterface) :
            mConstant(0.0624),
            mReductionOperations(aInterface.create())
    {
    }
    virtual ~OptimalityCriteriaTestObjectiveOne()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const OrdinalType tVectorIndex = 0;
        ScalarType tSum = mReductionOperations->sum(aControl[tVectorIndex]);
        ScalarType tOutput = mConstant * tSum;
        return (tOutput);
    }

    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aGradient)
    {
        const OrdinalType tVectorIndex = 0;
        aGradient[tVectorIndex].fill(mConstant);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaTestObjectiveOne<ScalarType, OrdinalType>>(*mReductionOperations);
        return (tOutput);
    }

private:
    ScalarType mConstant;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mReductionOperations;

private:
    OptimalityCriteriaTestObjectiveOne(const locus::OptimalityCriteriaTestObjectiveOne<ScalarType, OrdinalType>&);
    locus::OptimalityCriteriaTestObjectiveOne<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteriaTestObjectiveOne<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIATESTOBJECTIVEONE_HPP_ */
