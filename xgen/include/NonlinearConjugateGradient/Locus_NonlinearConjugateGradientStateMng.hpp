/*
 * Locus_NonlinearConjugateGradientStateMng.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENTSTATEMNG_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENTSTATEMNG_HPP_

#include <memory>

#include "Locus_MultiVector.hpp"
#include "Locus_StateManager.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStateMng : public locus::StateManager<ScalarType, OrdinalType>
{
public:
    NonlinearConjugateGradientStateMng(const std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> & aDataMng,
                                       const std::shared_ptr<locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType>> & aStageMng) :
            mDataMng(aDataMng),
            mStageMng(aStageMng)
    {
    }
    virtual ~NonlinearConjugateGradientStateMng()
    {
    }

    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        ScalarType tOutput = mStageMng->evaluateObjective(aControl);
        return (tOutput);
    }
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        mStageMng->computeGradient(aControl, aOutput);
    }
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        mStageMng->applyVectorToHessian(aControl, aVector, aOutput);
    }

    ScalarType getCurrentObjectiveValue() const
    {
        return (mDataMng->getCurrentObjectiveFunctionValue());
    }
    void setCurrentObjectiveValue(const ScalarType & aInput)
    {
        mDataMng->setCurrentObjectiveFunctionValue(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const
    {
        return (mDataMng->getTrialStep());
    }
    void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setTrialStep(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        return (mDataMng->getCurrentControl());
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setCurrentControl(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const
    {
        return (mDataMng->getCurrentGradient());
    }
    void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setCurrentGradient(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        return (mDataMng->getControlLowerBounds());
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setControlLowerBounds(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        return (mDataMng->getControlUpperBounds());
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setControlUpperBounds(aInput);
    }

    locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & getDataMng()
    {
        return (mDataMng.operator*());
    }
    locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & getStageMng()
    {
        return (mStageMng.operator*());
    }

private:
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> mDataMng;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType>> mStageMng;

private:
    NonlinearConjugateGradientStateMng(const locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENTSTATEMNG_HPP_ */
