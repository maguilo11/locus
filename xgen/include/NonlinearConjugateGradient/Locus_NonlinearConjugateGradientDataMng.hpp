/*
 * Locus_NonlinearConjugateGradientDataMng.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENTDATAMNG_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENTDATAMNG_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_ReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientDataMng
{
public:
    NonlinearConjugateGradientDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
        mIsInitialGuessSet(false),
        mNormGradient(std::numeric_limits<ScalarType>::max()),
        mStagnationMeasure(std::numeric_limits<ScalarType>::max()),
        mStationarityMeasure(std::numeric_limits<ScalarType>::max()),
        mObjectiveStagnationMeasure(std::numeric_limits<ScalarType>::max()),
        mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
        mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
        mControlWork(),
        mTrialStep(aDataFactory.control().create()),
        mCurrentControl(aDataFactory.control().create()),
        mPreviousControl(aDataFactory.control().create()),
        mCurrentGradient(aDataFactory.control().create()),
        mPreviousGradient(aDataFactory.control().create()),
        mControlLowerBounds(aDataFactory.control().create()),
        mControlUpperBounds(aDataFactory.control().create()),
        mControlReductions(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize();
    }
    ~NonlinearConjugateGradientDataMng()
    {
    }

    bool isInitialGuessSet() const
    {
        return (mIsInitialGuessSet);
    }
    OrdinalType getNumControlVectors() const
    {
        return (mCurrentControl->getNumVectors());
    }

    // NOTE: OBJECTIVE FUNCTION VALUE
    ScalarType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ScalarType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    ScalarType getPreviousObjectiveFunctionValue() const
    {
        return (mPreviousObjectiveFunctionValue);
    }
    void setPreviousObjectiveFunctionValue(const ScalarType & aInput)
    {
        mPreviousObjectiveFunctionValue = aInput;
    }

    // NOTE: SET INITIAL GUESS
    void setInitialGuess(const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }

    // NOTE: TRIAL STEP
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const
    {
        assert(mTrialStep.get() != nullptr);
        return (mTrialStep.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getTrialStep(const OrdinalType & aVectorIndex) const
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());
        return (mTrialStep->operator [](aVectorIndex));
    }
    void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mTrialStep->getNumVectors());
        locus::update(1., aInput, 0., *mTrialStep);
    }
    void setTrialStep(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());
        mTrialStep->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);
        return (mCurrentControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentControl(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());
        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mCurrentControl);
    }
    void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());
        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);
        return (mPreviousControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousControl(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());
        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousControl->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousControl);
    }
    void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());
        mPreviousControl->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }

    // NOTE: CURRENT GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const
    {
        assert(mCurrentGradient.get() != nullptr);
        return (mCurrentGradient.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());
        return (mCurrentGradient->operator[](aVectorIndex));
    }
    void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentGradient->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentGradient.operator*());
    }
    void setCurrentGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());
        mCurrentGradient->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }

    // NOTE: PREVIOUS GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousGradient() const
    {
        assert(mPreviousGradient.get() != nullptr);
        return (mPreviousGradient.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());
        return (mPreviousGradient->operator[](aVectorIndex));
    }
    void setPreviousGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousGradient->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mPreviousGradient.operator*());
    }
    void setPreviousGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());
        mPreviousGradient->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);
        return (mControlLowerBounds.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());
        return (mControlLowerBounds->operator[](aVectorIndex));
    }
    void setControlLowerBounds(const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<OrdinalType>(0));
        OrdinalType tNumVectors = mControlLowerBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());
        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());
        mControlLowerBounds->operator [](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mControlLowerBounds.operator*());
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);
        return (mControlUpperBounds.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());
        return (mControlUpperBounds->operator[](aVectorIndex));
    }
    void setControlUpperBounds(const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<OrdinalType>(0));
        OrdinalType tNumVectors = mControlUpperBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator[](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());
        mControlUpperBounds->operator[](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());
        mControlUpperBounds->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlUpperBounds->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mControlUpperBounds.operator*());
    }

    // NOTE: CONTROL STAGNATION MEASURE CALCULATION
    void computeStagnationMeasure()
    {
        const OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWork->update(static_cast<ScalarType>(1), tMyCurrentControl, static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWork->update(static_cast<ScalarType>(-1), tMyPreviousControl, static_cast<ScalarType>(1));
            mControlWork->modulus();
            storage[tIndex] = mControlReductions->max(*mControlWork);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeObjectiveStagnationMeasure()
    {
        mObjectiveStagnationMeasure = mPreviousObjectiveFunctionValue - mCurrentObjectiveFunctionValue;
        mObjectiveStagnationMeasure = std::abs(mObjectiveStagnationMeasure);
    }
    ScalarType getObjectiveStagnationMeasure() const
    {
        return (mObjectiveStagnationMeasure);
    }

    // NOTE: NORM OF GRADIENT MEASURE CALCULATION
    void computeNormGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        const OrdinalType tNumVectors = mCurrentGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = mCurrentGradient->operator[](tIndex);
            tCummulativeDotProduct += tMyGradient.dot(tMyGradient);
        }
        mNormGradient = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormGradient() const
    {
        return (mNormGradient);
    }

    // NOTE: COMPUTE STATIONARITY MEASURE
    void computeStationarityMeasure()
    {
        ScalarType tCummulativeDotProduct = 0.;
        const OrdinalType tNumVectors = mTrialStep->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyTrialStep = mTrialStep->operator[](tIndex);
            tCummulativeDotProduct += tMyTrialStep.dot(tMyTrialStep);
        }
        mStationarityMeasure = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    // NOTE: STORE PREVIOUS STATE (SIDENOTE: THE PREVIOUS TRIAL STEP IS ALWAYS STORED IN mTrialStep)
    void storePreviousState()
    {
        mPreviousObjectiveFunctionValue = mCurrentObjectiveFunctionValue;
        locus::update(static_cast<ScalarType>(1), *mCurrentControl, static_cast<ScalarType>(0), *mPreviousControl);
        locus::update(static_cast<ScalarType>(1), *mCurrentGradient, static_cast<ScalarType>(0), *mPreviousGradient);
    }

private:
    void initialize()
    {
        const OrdinalType tVectorIndex = 0;
        mControlWork = mCurrentControl->operator[](tVectorIndex).create();

        ScalarType tValue = -std::numeric_limits<ScalarType>::max();
        locus::fill(tValue, mControlLowerBounds.operator*());
        tValue = std::numeric_limits<ScalarType>::max();
        locus::fill(tValue, mControlUpperBounds.operator*());
    }

private:
    bool mIsInitialGuessSet;

    ScalarType mNormGradient;
    ScalarType mStagnationMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mObjectiveStagnationMeasure;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWork;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductions;

private:
    NonlinearConjugateGradientDataMng(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENTDATAMNG_HPP_ */
