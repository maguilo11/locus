/*
 * Locus_ConservativeConvexSeparableAppxDataMng.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXDATAMNG_HPP_
#define LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXDATAMNG_HPP_

#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <numeric>
#include <cassert>
#include <algorithm>

#include "Locus_Bounds.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_MultiVectorList.hpp"
#include "Locus_ReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxDataMng
{
public:
    explicit ConservativeConvexSeparableAppxDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mIsInitialGuessSet(false),
            mStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mFeasibilityMeasure(std::numeric_limits<ScalarType>::max()),
            mStationarityMeasure(std::numeric_limits<ScalarType>::max()),
            mNormInactiveGradient(std::numeric_limits<ScalarType>::max()),
            mObjectiveStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mDualProblemBoundsScaleFactor(0.5),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mDualObjectiveGlobalizationFactor(1),
            mKarushKuhnTuckerConditionsInexactness(std::numeric_limits<ScalarType>::max()),
            mDualWorkVectorOne(),
            mDualWorkVectorTwo(),
            mControlWorkVectorOne(),
            mControlWorkVectorTwo(),
            mDual(aDataFactory.dual().create()),
            mTrialStep(aDataFactory.control().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mCurrentSigma(aDataFactory.control().create()),
            mCurrentControl(aDataFactory.control().create()),
            mPreviousControl(aDataFactory.control().create()),
            mControlLowerBounds(aDataFactory.control().create()),
            mControlUpperBounds(aDataFactory.control().create()),
            mControlWorkMultiVector(aDataFactory.control().create()),
            mCurrentConstraintValues(aDataFactory.dual().create()),
            mCurrentObjectiveGradient(aDataFactory.control().create()),
            mConstraintGlobalizationFactors(aDataFactory.dual().create()),
            mDualReductions(aDataFactory.getDualReductionOperations().create()),
            mControlReductions(aDataFactory.getControlReductionOperations().create()),
            mCurrentConstraintGradients(std::make_shared<locus::MultiVectorList<ScalarType, OrdinalType>>())
    {
        this->initialize();
    }
    ~ConservativeConvexSeparableAppxDataMng()
    {
    }

    bool isInitialGuessSet() const
    {
        return (mIsInitialGuessSet);
    }

    // NOTE: NUMBER OF CONTROL VECTORS
    OrdinalType getNumControlVectors() const
    {
        return (mCurrentControl->getNumVectors());
    }
    // NOTE: NUMBER OF DUAL VECTORS
    OrdinalType getNumDualVectors() const
    {
        return (mDual->getNumVectors());
    }
    // NOTE :GET NUMBER OF CONSTRAINTS
    OrdinalType getNumConstraints() const
    {
        OrdinalType tNumConstraints = mCurrentConstraintGradients->size();
        return (tNumConstraints);
    }

    // NOTE: DUAL PROBLEM PARAMETERS
    ScalarType getDualProblemBoundsScaleFactor() const
    {
        return (mDualProblemBoundsScaleFactor);
    }
    void setDualProblemBoundsScaleFactor(const ScalarType & aInput)
    {
        mDualProblemBoundsScaleFactor = aInput;
    }
    ScalarType getDualObjectiveGlobalizationFactor() const
    {
        return (mDualObjectiveGlobalizationFactor);
    }
    void setDualObjectiveGlobalizationFactor(const ScalarType & aInput)
    {
        mDualObjectiveGlobalizationFactor = aInput;
    }

    // NOTE: CONSTRAINT GLOBALIZATION FACTORS FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintGlobalizationFactors() const
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        return (mConstraintGlobalizationFactors.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintGlobalizationFactors(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintGlobalizationFactors->getNumVectors());
        return (mConstraintGlobalizationFactors->operator [](aVectorIndex));
    }
    void setConstraintGlobalizationFactors(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintGlobalizationFactors->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintGlobalizationFactors);
    }
    void setConstraintGlobalizationFactors(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintGlobalizationFactors->getNumVectors());
        mConstraintGlobalizationFactors->operator [](aVectorIndex).update(1., aInput, 0.);
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

    // NOTE: TRIAL STEP FUNCTIONS
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

    // NOTE: CURRENT OBJECTIVE GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentObjectiveGradient() const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);

        return (mCurrentObjectiveGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentObjectiveGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentObjectiveGradient->getNumVectors());

        return (mCurrentObjectiveGradient->operator [](aVectorIndex));
    }
    void setCurrentObjectiveGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentObjectiveGradient->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentObjectiveGradient);
    }
    void setCurrentObjectiveGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentObjectiveGradient->getNumVectors());

        mCurrentObjectiveGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT SIGMA VECTOR
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentSigma() const
    {
        assert(mCurrentSigma.get() != nullptr);

        return (mCurrentSigma.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentSigma(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentSigma.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentSigma->getNumVectors());

        return (mCurrentSigma->operator [](aVectorIndex));
    }
    void setCurrentSigma(const ScalarType & aInput)
    {
        assert(mCurrentSigma.get() != nullptr);
        locus::fill(aInput, mCurrentSigma.operator*());
    }
    void setCurrentSigma(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentSigma->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentSigma);
    }
    void setCurrentSigma(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentSigma.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentSigma->getNumVectors());

        mCurrentSigma->operator [](aVectorIndex).update(1., aInput, 0.);
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

    // NOTE: CURRENT CONSTRAINT VALUES
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentConstraintValues() const
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        return (mCurrentConstraintValues.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentConstraintValues(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentConstraintValues->getNumVectors());
        return (mCurrentConstraintValues->operator [](aVectorIndex));
    }
    void setCurrentConstraintValues(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentConstraintValues->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mCurrentConstraintValues);
    }
    void setCurrentConstraintValues(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentConstraintValues->getNumVectors());
        mCurrentConstraintValues->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONSTRAINT GRADIENTS
    const locus::MultiVectorList<ScalarType, OrdinalType> & getCurrentConstraintGradients() const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        return (mCurrentConstraintGradients.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentConstraintGradients(const OrdinalType & aConstraintIndex) const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);

        return (mCurrentConstraintGradients->operator[](aConstraintIndex));
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentConstraintGradients(const OrdinalType & aConstraintIndex,
                                                                                 const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);
        assert(aVectorIndex < mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());

        return (mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex));
    }
    void setCurrentConstraintGradients(const locus::MultiVectorList<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.size() > static_cast<OrdinalType>(0));

        const OrdinalType tNumConstraints = aInput.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            locus::update(static_cast<ScalarType>(1),
                          aInput[tConstraintIndex],
                          static_cast<ScalarType>(0),
                          mCurrentConstraintGradients->operator[](tConstraintIndex));
        }
    }
    void setCurrentConstraintGradients(const OrdinalType & aConstraintIndex,
                                       const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());

        locus::update(static_cast<ScalarType>(1),
                      aInput,
                      static_cast<ScalarType>(0),
                      mCurrentConstraintGradients->operator[](aConstraintIndex));
    }
    void setCurrentConstraintGradients(const OrdinalType & aConstraintIndex,
                                       const OrdinalType & aVectorIndex,
                                       const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(aVectorIndex < mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());
        assert(aInput.size() == mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex).size());

        const ScalarType tAlpha = 1;
        const ScalarType tBeta = 0;
        mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex).update(tAlpha, aInput, tBeta);
    }

    // NOTE: STAGNATION MEASURE CRITERION
    void computeStagnationMeasure()
    {
        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkVectorOne->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkVectorOne->update(-1., tMyPreviousControl, 1.);
            mControlWorkVectorOne->modulus();
            storage[tIndex] = mControlReductions->max(*mControlWorkVectorOne);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: NORM OF CURRENT PROJECTED GRADIENT
    ScalarType computeInactiveVectorNorm(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = aInput.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyInputVector = aInput[tIndex];

            mControlWorkVectorOne->update(1., tMyInputVector, 0.);
            mControlWorkVectorOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVectorOne->dot(*mControlWorkVectorOne);
        }
        ScalarType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormInactiveGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = mCurrentObjectiveGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = (*mCurrentObjectiveGradient)[tIndex];

            mControlWorkVectorOne->update(1., tMyGradient, 0.);
            mControlWorkVectorOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVectorOne->dot(*mControlWorkVectorOne);
        }
        mNormInactiveGradient = (static_cast<ScalarType>(1) / mGlobalNumControls) * std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormInactiveGradient() const
    {
        return (mNormInactiveGradient);
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

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeFeasibilityMeasure()
    {
        const OrdinalType tNumVectors = mCurrentConstraintValues->getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors, static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyConstraintValues = mCurrentConstraintValues->operator[](tVectorIndex);
            mDualWorkVectorOne->update(static_cast<ScalarType>(1), tMyConstraintValues, static_cast<ScalarType>(0));
            mDualWorkVectorOne->modulus();
            tStorage[tVectorIndex] = mDualReductions->max(mDualWorkVectorOne.operator*());
        }
        const ScalarType tInitialValue = 0;
        mFeasibilityMeasure = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
    }
    ScalarType getFeasibilityMeasure() const
    {
        return (mFeasibilityMeasure);
    }

    // NOTE: STATIONARITY MEASURE CALCULATION
    void computeStationarityMeasure()
    {
        assert(mInactiveSet.get() != nullptr);
        assert(mCurrentControl.get() != nullptr);
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlUpperBounds.get() != nullptr);
        assert(mCurrentObjectiveGradient.get() != nullptr);

        locus::update(static_cast<ScalarType>(1), *mCurrentControl, static_cast<ScalarType>(0), *mControlWorkMultiVector);
        locus::update(static_cast<ScalarType>(-1), *mCurrentObjectiveGradient, static_cast<ScalarType>(1), *mControlWorkMultiVector);
        locus::bounds::project(*mControlLowerBounds, *mControlUpperBounds, *mControlWorkMultiVector);
        locus::update(static_cast<ScalarType>(1), *mCurrentControl, static_cast<ScalarType>(-1), *mControlWorkMultiVector);
        locus::update(static_cast<ScalarType>(1), *mControlWorkMultiVector, static_cast<ScalarType>(0), *mTrialStep);

        locus::entryWiseProduct(*mInactiveSet, *mControlWorkMultiVector);
        mStationarityMeasure = locus::norm(*mControlWorkMultiVector);
    }
    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    /*! Check inexactness in the Karush-Kuhn-Tucker (KKT) conditions (i.e. KKT residual) and compute
     * the norm of the KKT residual, where r(x,\lambda) = \{C1, C2, C3, C4\} denotes the residual vector
     * and C# denotes the corresponding Condition. The KKT conditions are given by:
     *
     * Condition 1: \left(1 + x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{+} = 0,\quad{j}=1,\dots,n_x
     * Condition 2: \left(1 - x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{-} = 0,\quad{j}=1,\dots,n_x
     * Condition 3: f_i(x)^{+} = 0,\quad{i}=1,\dots,N_c
     * Condition 4: \lambda_{i}f_i(x)^{-} = 0,\quad{i}=1,\dots,N_c.
     *
     * The nomenclature is given as follows: x denotes the control vector, \lambda denotes the dual
     * vector, N_c is the number of constraints, n_x is the number of controls, f_0 is the objective
     * function and f_i is the i-th constraint. Finally, a^{+} = max{0, a} and a^{-} = max{0, −a}.
     **/
    void computeKarushKuhnTuckerConditionsInexactness(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                                      const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aDual.getNumVectors() == mDual->getNumVectors());
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aDual[tVectorIndex].size() == mDual->operator[](tVectorIndex).size());
        assert(aControl.getNumVectors() == mCurrentControl->getNumVectors());
        assert(aControl[tVectorIndex].size() == mCurrentControl->operator[](tVectorIndex).size());

        locus::fill(static_cast<ScalarType>(0), mControlWorkMultiVector.operator*());
        const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tVectorIndex];
        const OrdinalType tNumConstraints = tDual.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tConstraintGradients =
                    mCurrentConstraintGradients->operator[](tConstraintIndex);
            locus::update(tDual[tConstraintIndex],
                          tConstraintGradients,
                          static_cast<ScalarType>(1),
                          mControlWorkMultiVector.operator*());
        }

        ScalarType tConditioneOne = std::numeric_limits<ScalarType>::max();
        ScalarType tConditioneTwo = std::numeric_limits<ScalarType>::max();
        this->computeConditionsOneAndTwo(aControl, aDual, tConditioneOne, tConditioneTwo);

        ScalarType tConditioneThree = std::numeric_limits<ScalarType>::max();
        ScalarType tConditioneFour = std::numeric_limits<ScalarType>::max();
        this->computeConditionsThreeAndFour(aControl, aDual, tConditioneThree, tConditioneFour);

        ScalarType tSum = tConditioneOne + tConditioneTwo + tConditioneThree + tConditioneFour;
        mKarushKuhnTuckerConditionsInexactness = (static_cast<ScalarType>(1) / mGlobalNumControls) * std::sqrt(tSum);
    }
    ScalarType getKarushKuhnTuckerConditionsInexactness() const
    {
        return (mKarushKuhnTuckerConditionsInexactness);
    }


private:
    void initialize()
    {
        const OrdinalType tControlVectorIndex = 0;
        mControlWorkVectorOne = mCurrentControl->operator[](tControlVectorIndex).create();
        mControlWorkVectorTwo = mCurrentControl->operator[](tControlVectorIndex).create();
        locus::fill(static_cast<ScalarType>(0), *mActiveSet);
        locus::fill(static_cast<ScalarType>(1), *mInactiveSet);

        assert(mDual->getNumVectors() == static_cast<OrdinalType>(1));
        const OrdinalType tDualVectorIndex = 0;
        mDualWorkVectorOne = mDual->operator[](tDualVectorIndex).create();
        mDualWorkVectorTwo = mDual->operator[](tDualVectorIndex).create();

        const OrdinalType tNumConstraints = mDual->operator[](tDualVectorIndex).size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            mCurrentConstraintGradients->add(mCurrentControl.operator*());
        }

        ScalarType tScalarValue = std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlUpperBounds);
        tScalarValue = -std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlLowerBounds);

        tScalarValue = 1;
        locus::fill(tScalarValue, mConstraintGlobalizationFactors.operator*());

        const size_t tNumElements = 1;
        locus::StandardVector<double> tVector(tNumElements);
        const size_t tVectorIndex = 0;
        tVector[0] = mCurrentControl->operator[](tVectorIndex).size();
        mGlobalNumControls = mControlReductions->sum(tVector);
    }
    /*!
     * Compute the following Karush-Kuhn-Tucker (KKT) conditions:
     *
     * Condition 3: f_i(x)^{+} = 0,\quad{i}=1,\dots,N_c
     * Condition 4: \lambda_{i}f_i(x)^{-} = 0,\quad{i}=1,\dots,N_c.
     *
     * where the nomenclature is given as follows: \lambda denotes the dual vector, N_c is the
     * number of constraints, n_x is the number of controls and f_i is the i-th constraint.
     * Finally, a^{+} = max{0, a} and a^{-} = max{0, −a}.
     **/
    void computeConditionsOneAndTwo(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                    const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                                    ScalarType & aConditionOne,
                                    ScalarType & aConditionTwo)
    {
        const OrdinalType tNumControlVectors = aControl.getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumControlVectors, static_cast<ScalarType>(0));
        std::vector<ScalarType> tStorageTwo(tNumControlVectors, static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tWorkOne = mControlWorkVectorOne.operator*();
            locus::Vector<ScalarType, OrdinalType> & tWorkTwo = mControlWorkVectorTwo.operator*();
            const locus::Vector<ScalarType, OrdinalType> & tControl = aControl[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tObjectiveGradient =
                    mCurrentObjectiveGradient->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tConstraintGradientTimesDual =
                    mControlWorkMultiVector->operator[](tVectorIndex);

            const OrdinalType tNumControls = tControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                tWorkOne[tControlIndex] = tObjectiveGradient[tControlIndex] + tConstraintGradientTimesDual[tControlIndex];
                tWorkOne[tControlIndex] = std::max(static_cast<ScalarType>(0), tWorkOne[tControlIndex]);
                tWorkOne[tControlIndex] = (static_cast<ScalarType>(1) + tControl[tControlIndex]) * tWorkOne[tControlIndex];
                tWorkOne[tControlIndex] = tWorkOne[tControlIndex] * tWorkOne[tControlIndex];

                tWorkTwo[tControlIndex] = tObjectiveGradient[tControlIndex] + tConstraintGradientTimesDual[tControlIndex];
                tWorkTwo[tControlIndex] = std::max(static_cast<ScalarType>(0), -tWorkTwo[tControlIndex]);
                tWorkTwo[tControlIndex] = (static_cast<ScalarType>(1) - tControl[tControlIndex]) * tWorkTwo[tControlIndex];
                tWorkTwo[tControlIndex] = tWorkTwo[tControlIndex] * tWorkTwo[tControlIndex];
            }

            tStorageOne[tVectorIndex] = mControlReductions->sum(tWorkOne);
            tStorageTwo[tVectorIndex] = mControlReductions->sum(tWorkTwo);
        }

        const ScalarType tInitialValue = 0;
        aConditionOne = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        aConditionTwo = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
    }
    /*!
     * Compute the following Karush-Kuhn-Tucker (KKT) conditions:
     *
     * Condition 1: \left(1 + x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{+} = 0,\quad{j}=1,\dots,n_x
     * Condition 2: \left(1 - x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{-} = 0,\quad{j}=1,\dots,n_x,
     *
     * where the nomenclature is given as follows: x denotes the control vector, \lambda denotes
     * the dual vector, N_c is the number of constraints, n_x is the number of controls, f_0 is
     * the objective function and f_i is the i-th constraint. Finally, a^{+} = max{0, a} and a^{-}
     * = max{0, −a}.
     **/
    void computeConditionsThreeAndFour(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                       const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                                       ScalarType & aConditionThree,
                                       ScalarType & aConditionFour)
    {
        const OrdinalType tNumDualVectors = aDual.getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumDualVectors, static_cast<ScalarType>(0));
        std::vector<ScalarType> tStorageTwo(tNumDualVectors, static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tWorkOne = mDualWorkVectorOne.operator*();
            locus::Vector<ScalarType, OrdinalType> & tWorkTwo = mDualWorkVectorTwo.operator*();
            const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tConstraintValues =
                    mCurrentConstraintValues->operator[](tVectorIndex);

            const OrdinalType tNumConstraints = tDual.size();
            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                tWorkOne[tConstraintIndex] = std::max(static_cast<ScalarType>(0), tConstraintValues[tConstraintIndex]);
                tWorkOne[tConstraintIndex] = tWorkOne[tConstraintIndex] * tWorkOne[tConstraintIndex];

                tWorkTwo[tConstraintIndex] = tDual[tConstraintIndex]
                        * std::max(static_cast<ScalarType>(0), -tConstraintValues[tConstraintIndex]);
                tWorkTwo[tConstraintIndex] = tWorkTwo[tConstraintIndex] * tWorkTwo[tConstraintIndex];
            }

            tStorageOne[tVectorIndex] = mDualReductions->sum(tWorkOne);
            tStorageTwo[tVectorIndex] = mDualReductions->sum(tWorkTwo);
        }

        const ScalarType tInitialValue = 0;
        aConditionThree = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        aConditionFour = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
    }

private:
    bool mIsInitialGuessSet;

    ScalarType mGlobalNumControls;
    ScalarType mStagnationMeasure;
    ScalarType mFeasibilityMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mNormInactiveGradient;
    ScalarType mObjectiveStagnationMeasure;
    ScalarType mDualProblemBoundsScaleFactor;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;
    ScalarType mDualObjectiveGlobalizationFactor;
    ScalarType mKarushKuhnTuckerConditionsInexactness;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkVectorOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkVectorTwo;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVectorOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVectorTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentSigma;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWorkMultiVector;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintValues;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintGlobalizationFactors;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductions;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductions;

    std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> mCurrentConstraintGradients;

private:
    ConservativeConvexSeparableAppxDataMng(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXDATAMNG_HPP_ */
