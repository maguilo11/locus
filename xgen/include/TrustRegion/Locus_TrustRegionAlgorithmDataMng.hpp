/*
 * Locus_TrustRegionAlgorithmDataMng.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_TRUSTREGIONALGORITHMDATAMNG_HPP_
#define LOCUS_TRUSTREGIONALGORITHMDATAMNG_HPP_

#include <limits>
#include <memory>
#include <cassert>

#include "Locus_Bounds.hpp"
#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_ReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class TrustRegionAlgorithmDataMng
{
public:
    explicit TrustRegionAlgorithmDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mNumDualVectors(aDataFactory.dual().getNumVectors()),
            mNumControlVectors(aDataFactory.control().getNumVectors()),
            mStagnationMeasure(0),
            mStationarityMeasure(0),
            mNormProjectedGradient(0),
            mGradientInexactnessTolerance(0),
            mObjectiveInexactnessTolerance(0),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mIsInitialGuessSet(false),
            mGradientInexactnessToleranceExceeded(false),
            mObjectiveInexactnessToleranceExceeded(false),
            mDual(aDataFactory.dual().create()),
            mTrialStep(aDataFactory.control().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mCurrentControl(aDataFactory.control().create()),
            mPreviousControl(aDataFactory.control().create()),
            mCurrentGradient(aDataFactory.control().create()),
            mPreviousGradient(aDataFactory.control().create()),
            mControlLowerBounds(aDataFactory.control().create()),
            mControlUpperBounds(aDataFactory.control().create()),
            mControlWorkVector(),
            mControlWorkMultiVector(aDataFactory.control().create()),
            mDualReductionOperations(aDataFactory.getDualReductionOperations().create()),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize(aDataFactory);
    }
    virtual ~TrustRegionAlgorithmDataMng()
    {
    }

    // NOTE: NUMBER OF CONTROL VECTORS
    OrdinalType getNumControlVectors() const
    {
        return (mNumControlVectors);
    }
    // NOTE: NUMBER OF DUAL VECTORS
    OrdinalType getNumDualVectors() const
    {
        return (mNumDualVectors);
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
    void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }

    // NOTE: DUAL VECTOR
    const locus::MultiVector<ScalarType, OrdinalType> & getDual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getDual(const OrdinalType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        return (mDual->operator [](aVectorIndex));
    }
    void setDual(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mDual->getNumVectors());
        locus::update(1., aInput, 0., *mDual);
    }
    void setDual(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        mDual->operator [](aVectorIndex).update(1., aInput, 0.);
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

    // NOTE: ACTIVE SET FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getActiveSet() const
    {
        assert(mActiveSet.get() != nullptr);

        return (mActiveSet.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getActiveSet(const OrdinalType & aVectorIndex) const
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        return (mActiveSet->operator [](aVectorIndex));
    }
    void setActiveSet(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mActiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mActiveSet);
    }
    void setActiveSet(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        mActiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: INACTIVE SET FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getInactiveSet() const
    {
        assert(mInactiveSet.get() != nullptr);

        return (mInactiveSet.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getInactiveSet(const OrdinalType & aVectorIndex) const
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        return (mInactiveSet->operator [](aVectorIndex));
    }
    void setInactiveSet(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mInactiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mInactiveSet);
    }
    void setInactiveSet(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        mInactiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
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

        mPreviousControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const
    {
        assert(mCurrentGradient.get() != nullptr);

        return (mCurrentGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());

        return (mCurrentGradient->operator [](aVectorIndex));
    }
    void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentGradient->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentGradient);
    }
    void setCurrentGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());

        mCurrentGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousGradient() const
    {
        assert(mPreviousGradient.get() != nullptr);

        return (mPreviousGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());

        return (mPreviousGradient->operator [](aVectorIndex));
    }
    void setPreviousGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousGradient->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousGradient);
    }
    void setPreviousGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());

        mPreviousGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);

        return (mControlLowerBounds.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        return (mControlLowerBounds->operator [](aVectorIndex));
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

        mControlLowerBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlLowerBounds);
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);

        return (mControlUpperBounds.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        return (mControlUpperBounds->operator [](aVectorIndex));
    }
    void setControlUpperBounds(const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mControlUpperBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlUpperBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlUpperBounds);
    }

    // NOTE: OBJECTIVE AND GRADIENT INEXACTNESS VIOLATION FLAGS
    bool isInitialGuessSet() const
    {
        return (mIsInitialGuessSet);
    }
    void setGradientInexactnessFlag(const bool & aInput)
    {
        mGradientInexactnessToleranceExceeded = aInput;
    }
    bool isGradientInexactnessToleranceExceeded() const
    {
        return (mGradientInexactnessToleranceExceeded);
    }
    void setObjectiveInexactnessFlag(const bool & aInput)
    {
        mObjectiveInexactnessToleranceExceeded = aInput;
    }
    bool isObjectiveInexactnessToleranceExceeded() const
    {
        return (mObjectiveInexactnessToleranceExceeded);
    }

    // NOTE: STAGNATION MEASURE CRITERION
    void computeStagnationMeasure()
    {
        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkVector->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkVector->update(-1., tMyPreviousControl, 1.);
            mControlWorkVector->modulus();
            storage[tIndex] = mControlReductionOperations->max(*mControlWorkVector);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: NORM OF CURRENT PROJECTED GRADIENT
    ScalarType computeProjectedVectorNorm(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = aInput.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyInputVector = aInput[tIndex];

            mControlWorkVector->update(1., tMyInputVector, 0.);
            mControlWorkVector->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVector->dot(*mControlWorkVector);
        }
        ScalarType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormProjectedGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = mCurrentGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = (*mCurrentGradient)[tIndex];

            mControlWorkVector->update(1., tMyGradient, 0.);
            mControlWorkVector->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVector->dot(*mControlWorkVector);
        }
        mNormProjectedGradient = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormProjectedGradient() const
    {
        return (mNormProjectedGradient);
    }

    // NOTE: STATIONARITY MEASURE CALCULATION
    void computeStationarityMeasure()
    {
        assert(mInactiveSet.get() != nullptr);
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentGradient.get() != nullptr);
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlUpperBounds.get() != nullptr);

        locus::update(1., *mCurrentControl, 0., *mControlWorkMultiVector);
        locus::update(-1., *mCurrentGradient, 1., *mControlWorkMultiVector);
        locus::bounds::project(*mControlLowerBounds, *mControlUpperBounds, *mControlWorkMultiVector);
        locus::update(1., *mCurrentControl, -1., *mControlWorkMultiVector);
        locus::entryWiseProduct(*mInactiveSet, *mControlWorkMultiVector);
        mStationarityMeasure = locus::norm(*mControlWorkMultiVector);
    }
    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    // NOTE: RESET AND STORE STAGE DATA
    void resetCurrentStageDataToPreviousStageData()
    {
        OrdinalType tNumVectors = mCurrentGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = (*mCurrentControl)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = (*mPreviousControl)[tIndex];
            tMyCurrentControl.update(1., tMyPreviousControl, 0.);

            locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = (*mCurrentGradient)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousGradient = (*mPreviousGradient)[tIndex];
            tMyCurrentGradient.update(1., tMyPreviousGradient, 0.);
        }
        mCurrentObjectiveFunctionValue = mPreviousObjectiveFunctionValue;
    }
    void storeCurrentStageData()
    {
        const ScalarType tCurrentObjectiveValue = this->getCurrentObjectiveFunctionValue();
        this->setPreviousObjectiveFunctionValue(tCurrentObjectiveValue);

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = this->getCurrentControl(tVectorIndex);
            this->setPreviousControl(tVectorIndex, tMyCurrentControl);

            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = this->getCurrentGradient(tVectorIndex);
            this->setPreviousGradient(tVectorIndex, tMyCurrentGradient);
        }
    }

private:
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        assert(aDataFactory.control().getNumVectors() > 0);

        const OrdinalType tVectorIndex = 0;
        mControlWorkVector = aDataFactory.control(tVectorIndex).create();
        locus::fill(static_cast<ScalarType>(0), *mActiveSet);
        locus::fill(static_cast<ScalarType>(1), *mInactiveSet);

        ScalarType tScalarValue = std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlUpperBounds);
        tScalarValue = -std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlLowerBounds);
    }

private:
    OrdinalType mNumDualVectors;
    OrdinalType mNumControlVectors;

    ScalarType mStagnationMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mNormProjectedGradient;
    ScalarType mGradientInexactnessTolerance;
    ScalarType mObjectiveInexactnessTolerance;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;

    bool mIsInitialGuessSet;
    bool mGradientInexactnessToleranceExceeded;
    bool mObjectiveInexactnessToleranceExceeded;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWorkMultiVector;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    TrustRegionAlgorithmDataMng(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aRhs);
    locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & operator=(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aRhs);
};

}

#endif /* LOCUS_TRUSTREGIONALGORITHMDATAMNG_HPP_ */
