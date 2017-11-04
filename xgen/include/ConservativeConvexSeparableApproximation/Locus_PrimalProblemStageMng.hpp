/*
 * Locus_PrimalProblemStageMng.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_PRIMALPROBLEMSTAGEMNG_HPP_
#define LOCUS_PRIMALPROBLEMSTAGEMNG_HPP_

#include <memory>
#include <vector>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_StateData.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_MultiVectorList.hpp"
#include "Locus_AnalyticalGradient.hpp"
#include "Locus_GradientOperatorList.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"
#include "Locus_ConservativeConvexSeparableAppxStageMng.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class PrimalProblemStageMng : public locus::ConservativeConvexSeparableAppxStageMng<ScalarType, OrdinalType>
{
public:
    PrimalProblemStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumConstraintEvaluations(),
            mNumConstraintGradientEvaluations(),
            mState(aDataFactory.state().create()),
            mObjective(),
            mConstraints(),
            mObjectiveGradient(),
            mConstraintGradients(),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory))
    {
    }
    PrimalProblemStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory,
                          const locus::Criterion<ScalarType, OrdinalType> & aObjective,
                          const locus::CriterionList<ScalarType, OrdinalType> & aConstraints) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumConstraintEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mNumConstraintGradientEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mState(aDataFactory.state().create()),
            mObjective(aObjective.create()),
            mConstraints(aConstraints.create()),
            mObjectiveGradient(),
            mConstraintGradients(),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory))
    {
        mObjectiveGradient = std::make_shared<locus::AnalyticalGradient<ScalarType, OrdinalType>>(mObjective);
        mConstraintGradients = std::make_shared<locus::GradientOperatorList<ScalarType, OrdinalType>>(mConstraints);
    }
    virtual ~PrimalProblemStageMng()
    {
    }

    OrdinalType getNumObjectiveFunctionEvaluations() const
    {
        return (mNumObjectiveFunctionEvaluations);
    }
    OrdinalType getNumObjectiveGradientEvaluations() const
    {
        return (mNumObjectiveGradientEvaluations);
    }
    OrdinalType getNumConstraintEvaluations(const OrdinalType & aIndex) const
    {
        assert(mNumConstraintEvaluations.empty() == false);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mNumConstraintEvaluations.size());
        return (mNumConstraintEvaluations[aIndex]);
    }
    OrdinalType getNumConstraintGradientEvaluations(const OrdinalType & aIndex) const
    {
        assert(mNumConstraintGradientEvaluations.empty() == false);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mNumConstraintGradientEvaluations.size());
        return (mNumConstraintGradientEvaluations[aIndex]);
    }

    void setObjectiveGradient(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mObjectiveGradient = aInput.create();
    }
    void setConstraintGradients(const locus::GradientOperatorList<ScalarType, OrdinalType> & aInput)
    {
        mConstraintGradients = aInput.create();
    }

    void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mStateData->setCurrentTrialStep(aDataMng.getTrialStep());
        mStateData->setCurrentControl(aDataMng.getCurrentControl());
        mStateData->setCurrentObjectiveGradient(aDataMng.getCurrentObjectiveGradient());
        mStateData->setCurrentObjectiveFunctionValue(aDataMng.getCurrentObjectiveFunctionValue());

        mObjectiveGradient->update(mStateData.operator*());

        const OrdinalType tNumConstraints = mConstraintGradients->size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintGradient =
                    aDataMng.getCurrentConstraintGradients(tConstraintIndex);
            mStateData->setCurrentConstraintGradient(tMyConstraintGradient);
            mConstraintGradients->operator[](tConstraintIndex).update(mStateData.operator*());
        }
    }
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(mObjective.get() != nullptr);

        ScalarType tObjectiveFunctionValue = mObjective->value(mState.operator*(), aControl);
        mNumObjectiveFunctionEvaluations++;

        return (tObjectiveFunctionValue);
    }
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mObjectiveGradient.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mObjectiveGradient->compute(mState.operator*(), aControl, aOutput);
        mNumObjectiveGradientEvaluations++;
    }
    void evaluateConstraints(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                             locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mConstraints.get() != nullptr);

        locus::fill(static_cast<ScalarType>(0), aOutput);
        const OrdinalType tNumVectors = aOutput.getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyOutput = aOutput[tVectorIndex];
            const OrdinalType tNumConstraints = tMyOutput.size();
            assert(tNumConstraints == mConstraints->size());

            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                assert(mConstraints->ptr(tConstraintIndex).get() != nullptr);

                tMyOutput[tConstraintIndex] = mConstraints->operator[](tConstraintIndex).value(*mState, aControl);
                mNumConstraintEvaluations[tConstraintIndex] =
                        mNumConstraintEvaluations[tConstraintIndex] + static_cast<OrdinalType>(1);
            }
        }
    }
    void computeConstraintGradients(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                    locus::MultiVectorList<ScalarType, OrdinalType> & aOutput)
    {
        assert(mConstraintGradients.get() != nullptr);
        assert(mConstraintGradients->size() == aOutput.size());

        const OrdinalType tNumConstraints = aOutput.size();
        for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            assert(mConstraints->ptr(tIndex).get() != nullptr);

            locus::MultiVector<ScalarType, OrdinalType> & tMyOutput = aOutput[tIndex];
            locus::fill(static_cast<ScalarType>(0), tMyOutput);
            mConstraints->operator[](tIndex).gradient(*mState, aControl, tMyOutput);
            mNumConstraintGradientEvaluations[tIndex] =
                    mNumConstraintGradientEvaluations[tIndex] + static_cast<OrdinalType>(1);
        }
    }

private:
    OrdinalType mNumObjectiveFunctionEvaluations;
    OrdinalType mNumObjectiveGradientEvaluations;

    std::vector<OrdinalType> mNumConstraintEvaluations;
    std::vector<OrdinalType> mNumConstraintGradientEvaluations;

    std::shared_ptr<MultiVector<ScalarType, OrdinalType>> mState;
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mObjective;
    std::shared_ptr<locus::CriterionList<ScalarType, OrdinalType>> mConstraints;
    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> mObjectiveGradient;
    std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> mConstraintGradients;

    std::shared_ptr<locus::StateData<ScalarType, OrdinalType>> mStateData;

private:
    PrimalProblemStageMng(const locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aRhs);
    locus::PrimalProblemStageMng<ScalarType, OrdinalType> & operator=(const locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_PRIMALPROBLEMSTAGEMNG_HPP_ */
