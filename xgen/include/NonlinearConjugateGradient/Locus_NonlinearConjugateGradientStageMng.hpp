/*
 * Locus_NonlinearConjugateGradientStageMng.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENTSTAGEMNG_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENTSTAGEMNG_HPP_

#include <memory>
#include <cassert>

#include "Locus_StateData.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_AnalyticalHessian.hpp"
#include "Locus_AnalyticalGradient.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStageMng : public locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType>
{
public:
    NonlinearConjugateGradientStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory,
                                       const locus::Criterion<ScalarType, OrdinalType> & aObjective) :
            mNumHessianEvaluations(0),
            mNumFunctionEvaluations(0),
            mNumGradientEvaluations(0),
            mState(aDataFactory.state().create()),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory)),
            mObjective(aObjective.create()),
            mHessian(std::make_shared<locus::AnalyticalHessian<ScalarType, OrdinalType>>(aObjective)),
            mGradient(std::make_shared<locus::AnalyticalGradient<ScalarType, OrdinalType>>(aObjective))

    {
    }
    ~NonlinearConjugateGradientStageMng()
    {
    }

    OrdinalType getNumObjectiveFunctionEvaluations() const
    {
        return (mNumFunctionEvaluations);
    }
    void setNumObjectiveFunctionEvaluations(const ScalarType & aInput)
    {
        mNumFunctionEvaluations = aInput;
    }
    OrdinalType getNumObjectiveGradientEvaluations() const
    {
        return (mNumGradientEvaluations);
    }
    void setNumObjectiveGradientEvaluations(const ScalarType & aInput)
    {
        mNumGradientEvaluations = aInput;
    }
    OrdinalType getNumObjectiveHessianEvaluations() const
    {
        return (mNumHessianEvaluations);
    }
    void setNumObjectiveHessianEvaluations(const ScalarType & aInput)
    {
        mNumHessianEvaluations = aInput;
    }

    void setGradient(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mGradient = aInput.create();
    }
    void setHessian(const locus::LinearOperator<ScalarType, OrdinalType> & aInput)
    {
        mHessian = aInput.create();
    }

    void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mStateData->setCurrentTrialStep(aDataMng.getTrialStep());
        mStateData->setCurrentControl(aDataMng.getCurrentControl());
        mStateData->setCurrentObjectiveGradient(aDataMng.getCurrentGradient());
        mStateData->setCurrentObjectiveFunctionValue(aDataMng.getCurrentObjectiveFunctionValue());

        mHessian->update(mStateData.operator*());
        mGradient->update(mStateData.operator*());
    }
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 ScalarType aTolerance = std::numeric_limits<ScalarType>::max())
    {
        assert(mObjective.get() != nullptr);
        ScalarType tObjectiveFunctionValue = mObjective->value(mState.operator*(), aControl);
        mNumFunctionEvaluations++;
        return (tObjectiveFunctionValue);
    }
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mGradient.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mGradient->compute(mState.operator*(), aControl, aOutput);
        mNumGradientEvaluations++;
    }
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mHessian.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mHessian->apply(mState.operator*(), aControl, aVector, aOutput);
        mNumHessianEvaluations++;
    }

private:
    OrdinalType mNumHessianEvaluations;
    OrdinalType mNumFunctionEvaluations;
    OrdinalType mNumGradientEvaluations;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mState;
    std::shared_ptr<locus::StateData<ScalarType, OrdinalType>> mStateData;
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mObjective;

    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> mHessian;
    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> mGradient;

private:
    NonlinearConjugateGradientStageMng(const locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENTSTAGEMNG_HPP_ */
