/*
 * Locus_StateData.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_STATEDATA_HPP_
#define LOCUS_STATEDATA_HPP_

#include <limits>
#include <memory>
#include <cassert>

#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class StateData
{
public:
    explicit StateData(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mCurrentControl(aDataFactory.control().create()),
            mCurrentTrialStep(aDataFactory.control().create()),
            mCurrentObjectiveGradient(aDataFactory.control().create()),
            mCurrentConstraintGradient(aDataFactory.control().create())
    {
    }
    ~StateData()
    {
    }

    ScalarType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ScalarType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);
        return (mCurrentControl.operator*());
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentControl.operator*());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentTrialStep() const
    {
        assert(mCurrentTrialStep.get() != nullptr);
        return (mCurrentTrialStep.operator*());
    }
    void setCurrentTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentTrialStep.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentTrialStep->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentTrialStep.operator*());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentObjectiveGradient() const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        return (mCurrentObjectiveGradient);
    }
    void setCurrentObjectiveGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentObjectiveGradient->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentObjectiveGradient.operator*());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentConstraintGradient() const
    {
        assert(mCurrentConstraintGradient.get() != nullptr);
        return (mCurrentConstraintGradient);
    }
    void setCurrentConstraintGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradient.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentConstraintGradient->getNumVectors());

        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentConstraintGradient.operator*());
    }

private:
    ScalarType mCurrentObjectiveFunctionValue;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintGradient;

private:
    StateData(const locus::StateData<ScalarType, OrdinalType>&);
    locus::StateData<ScalarType, OrdinalType> & operator=(const locus::StateData<ScalarType, OrdinalType>&);
};

} // namespace locus

#endif /* LOCUS_STATEDATA_HPP_ */
