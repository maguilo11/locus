/*
 * Locus_VectorTest.cpp
 *
 *  Created on: Jun 14, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <mpi.h>

#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <sstream>
#include <numeric>
#include <iterator>
#include <iostream>
#include <algorithm>

#include "Locus_UnitTestUtils.hpp"

#include "Locus_Types.hpp"
#include "Locus_Bounds.hpp"
#include "Locus_StateData.hpp"
#include "Locus_Rosenbrock.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_IdentityHessian.hpp"
#include "Locus_MultiVectorList.hpp"
#include "Locus_AnalyticalHessian.hpp"
#include "Locus_AnalyticalGradient.hpp"
#include "Locus_LinearOperatorList.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_GradientOperatorList.hpp"
#include "Locus_IdentityPreconditioner.hpp"
#include "Locus_NonlinearConjugateGradient.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

/**********************************************************************************************************/
/*************** AUGMENTED LAGRANGIAN IMPLEMENTATION OF KELLEY-SACHS TRUST REGION ALGORITHM ***************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class StandardAlgorithmDataMng
{
public:
    virtual ~StandardAlgorithmDataMng()
    {
    }

    virtual OrdinalType getNumControlVectors() const = 0;

    // NOTE: OBJECTIVE FUNCTION VALUE
    virtual ScalarType getCurrentObjectiveFunctionValue() const = 0;
    virtual void setCurrentObjectiveFunctionValue(const ScalarType & aInput) = 0;
    virtual ScalarType getPreviousObjectiveFunctionValue() const = 0;
    virtual void setPreviousObjectiveFunctionValue(const ScalarType & aInput) = 0;

    // NOTE: SET INITIAL GUESS
    virtual void setInitialGuess(const ScalarType & aValue) = 0;
    virtual void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue) = 0;
    virtual void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInitialGuess) = 0;
    virtual void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInitialGuess) = 0;

    // NOTE: TRIAL STEP
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getTrialStep(const OrdinalType & aVectorIndex) const = 0;
    virtual void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aTrialStep) = 0;
    virtual void setTrialStep(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aTrialStep) = 0;

    // NOTE: CURRENT CONTROL
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getCurrentControl(const OrdinalType & aVectorIndex) const = 0;
    virtual void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aControl) = 0;

    // NOTE: PREVIOUS CONTROL
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getPreviousControl() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getPreviousControl(const OrdinalType & aVectorIndex) const = 0;
    virtual void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aControl) = 0;

    // NOTE: CURRENT GRADIENT
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getCurrentGradient(const OrdinalType & aVectorIndex) const = 0;
    virtual void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aGradient) = 0;
    virtual void setCurrentGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aGradient) = 0;

    // NOTE: PREVIOUS GRADIENT
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getPreviousGradient() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getPreviousGradient(const OrdinalType & aVectorIndex) const = 0;
    virtual void setPreviousGradient(const locus::MultiVector<ScalarType, OrdinalType> & aGradient) = 0;
    virtual void setPreviousGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aGradient) = 0;

    // NOTE: SET CONTROL LOWER BOUNDS
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const = 0;
    virtual void setControlLowerBounds(const ScalarType & aValue) = 0;
    virtual void setControlLowerBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue) = 0;
    virtual void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aLowerBound) = 0;
    virtual void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aLowerBound) = 0;

    // NOTE: SET CONTROL UPPER BOUNDS
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const = 0;
    virtual void setControlUpperBounds(const ScalarType & aValue) = 0;
    virtual void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue) = 0;
    virtual void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aUpperBound) = 0;
    virtual void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aUpperBound) = 0;
};

/**********************************************************************************************************/
/************************* CONSERVATIVE CONVEX SEPARABLE APPROXIMATION ALGORITHM **************************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxDataMng
{
public:
    explicit ConservativeConvexSeparableAppxDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mIsInitialGuessSet(false),
            mStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mFeasibilityMeasure(std::numeric_limits<ScalarType>::max()),
            mStationarityMeasure(std::numeric_limits<ScalarType>::max()),
            mNormProjectedGradient(std::numeric_limits<ScalarType>::max()),
            mObjectiveStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mDualProblemBoundsScaleFactor(0.5),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mDualObjectiveGlobalizationFactor(1),
            mKarushKuhnTuckerConditionsInexactness(std::numeric_limits<ScalarType>::max()),
            mDualWorkOne(),
            mDualWorkTwo(),
            mControlWorkOne(),
            mControlWorkTwo(),
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
    void setDualObjectiveGlobalizationFactor(const ScalarType & aInput) const
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
    void getCurrentConstraintGradients(locus::MultiVectorList<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aInput.size() == mCurrentConstraintGradients->size());

        const OrdinalType tNumConstraints = aInput.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            assert(aInput[tConstraintIndex].get() != nullptr);
            assert(mCurrentConstraintGradients->ptr(tConstraintIndex).get() != nullptr);
            locus::update(static_cast<ScalarType>(1),
                          mCurrentConstraintGradients->operator[](tConstraintIndex),
                          static_cast<ScalarType>(0),
                          aInput[tConstraintIndex]);
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
            mControlWorkOne->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkOne->update(-1., tMyPreviousControl, 1.);
            mControlWorkOne->modulus();
            storage[tIndex] = mControlReductions->max(*mControlWorkOne);
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

            mControlWorkOne->update(1., tMyInputVector, 0.);
            mControlWorkOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkOne->dot(*mControlWorkOne);
        }
        ScalarType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormProjectedGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = mCurrentObjectiveGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = (*mCurrentObjectiveGradient)[tIndex];

            mControlWorkOne->update(1., tMyGradient, 0.);
            mControlWorkOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkOne->dot(*mControlWorkOne);
        }
        mNormProjectedGradient = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormProjectedGradient() const
    {
        return (mNormProjectedGradient);
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
            const locus::Vector<ScalarType, OrdinalType> & tDual = mDual->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tWork = mDualWorkOne->operator[](tVectorIndex);
            locus::update(static_cast<ScalarType>(1), tDual, static_cast<ScalarType>(0), tWork);
            tWork.modulus();
            tStorage[tVectorIndex] = mDualReductions->max(tWork);
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
            const locus::MultiVector<ScalarType, OrdinalType> tConstraintGradients =
                    mCurrentConstraintGradients[tConstraintIndex].operator*();
            locus::MultiVector<ScalarType, OrdinalType> tConstraintGradientsTimesDual =
                    mControlWorkMultiVector.operator*();
            locus::update(tDual[tConstraintIndex],
                          tConstraintGradients,
                          static_cast<ScalarType>(1),
                          tConstraintGradientsTimesDual);
        }

        ScalarType tConditioneOne = std::numeric_limits<ScalarType>::max();
        ScalarType tConditioneTwo = std::numeric_limits<ScalarType>::max();
        this->computeConditionsOneAndTwo(aControl, aDual, tConditioneOne, tConditioneTwo);

        const ScalarType tConditioneThree = std::numeric_limits<ScalarType>::max();
        const ScalarType tConditioneFour = std::numeric_limits<ScalarType>::max();
        this->computeConditionsThreeAndFour(aControl, aDual, tConditioneThree, tConditioneFour);

        ScalarType tNumControls = aControl[tVectorIndex].size();
        ScalarType tSum = tConditioneOne + tConditioneTwo + tConditioneThree + tConditioneFour;
        mKarushKuhnTuckerConditionsInexactness = (static_cast<ScalarType>(1) / tNumControls) * std::sqrt(tSum);
    }
    ScalarType getKarushKuhnTuckerConditionsInexactness() const
    {
        return (mKarushKuhnTuckerConditionsInexactness);
    }


private:
    void initialize()
    {
        const OrdinalType tControlVectorIndex = 0;
        mControlWorkOne = mCurrentControl->operator[](tControlVectorIndex).create();
        mControlWorkTwo = mCurrentControl->operator[](tControlVectorIndex).create();
        locus::fill(static_cast<ScalarType>(0), *mActiveSet);
        locus::fill(static_cast<ScalarType>(1), *mInactiveSet);

        assert(mDual->getNumVectors() == static_cast<OrdinalType>(1));
        const OrdinalType tDualVectorIndex = 0;
        mDualWorkOne = mDual->operator[](tDualVectorIndex).create();
        mDualWorkTwo = mDual->operator[](tDualVectorIndex).create();

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
            locus::Vector<ScalarType, OrdinalType> & tWorkOne = mControlWorkOne.operator*();
            locus::Vector<ScalarType, OrdinalType> & tWorkTwo = mControlWorkTwo.operator*();
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
            locus::Vector<ScalarType, OrdinalType> & tWorkOne = mDualWorkOne.operator*();
            locus::Vector<ScalarType, OrdinalType> & tWorkTwo = mDualWorkTwo.operator*();
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

    ScalarType mStagnationMeasure;
    ScalarType mFeasibilityMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mNormProjectedGradient;
    ScalarType mObjectiveStagnationMeasure;
    ScalarType mDualProblemBoundsScaleFactor;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;
    ScalarType mDualObjectiveGlobalizationFactor;
    ScalarType mKarushKuhnTuckerConditionsInexactness;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkTwo;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkTwo;

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

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxStageMng
{
public:
    virtual ~ConservativeConvexSeparableAppxStageMng()
    {
    }

    virtual void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

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
            mConstraints->operator[](tIndex).computeConstraintGradients(*mState, aControl, tMyOutput);
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

template<typename ScalarType, typename OrdinalType = size_t>
class DualProblemStageMng : public locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType>
{
public:
    explicit DualProblemStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mEpsilon(1e-6),
            mObjectiveCoefficientA(1),
            mObjectiveCoefficientR(1),
            mTrialAuxiliaryVariableZ(0),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mDualWorkVector(),
            mControlWorkVectorOne(),
            mControlWorkVectorTwo(),
            mTermA(aDataFactory.control().create()),
            mTermB(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mLowerAsymptotes(aDataFactory.control().create()),
            mUpperAsymptotes(aDataFactory.control().create()),
            mDualTimesCoefficientsP(aDataFactory.control().create()),
            mDualTimesCoefficientsQ(aDataFactory.control().create()),
            mObjectiveCoefficientsP(aDataFactory.control().create()),
            mObjectiveCoefficientsQ(aDataFactory.control().create()),
            mTrialControlLowerBounds(aDataFactory.control().create()),
            mTrialControlUpperBounds(aDataFactory.control().create()),
            mConstraintCoefficientsA(aDataFactory.dual().create()),
            mConstraintCoefficientsC(aDataFactory.dual().create()),
            mConstraintCoefficientsD(aDataFactory.dual().create()),
            mConstraintCoefficientsR(aDataFactory.dual().create()),
            mTrialAuxiliaryVariableY(aDataFactory.dual().create()),
            mCurrentConstraintValues(aDataFactory.dual().create()),
            mDualReductionOperations(aDataFactory.getDualReductionOperations().create()),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create()),
            mConstraintCoefficientsP(),
            mConstraintCoefficientsQ()
    {
        this->initialize(aDataFactory);
    }
    virtual ~DualProblemStageMng()
    {
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS A
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsA() const
    {
        assert(mConstraintCoefficientsA.get() != nullptr);

        return (mConstraintCoefficientsA.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintCoefficientsA(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        return (mConstraintCoefficientsA->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsA(const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(mConstraintCoefficientsA->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mConstraintCoefficientsA->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsA->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsA(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        mConstraintCoefficientsA->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsA(const OrdinalType & aVectorIndex,
                                    const OrdinalType & aElementIndex,
                                    const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());
        assert(aElementIndex >= static_cast<OrdinalType>(0));
        assert(aElementIndex < mConstraintCoefficientsA->operator [](aVectorIndex).size());

        mConstraintCoefficientsA->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsA(const OrdinalType & aVectorIndex,
                                    const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        mConstraintCoefficientsA->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsA(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsA->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsA);
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS C
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsC() const
    {
        assert(mConstraintCoefficientsC.get() != nullptr);

        return (mConstraintCoefficientsC.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintCoefficientsC(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        return (mConstraintCoefficientsC->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsC(const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(mConstraintCoefficientsC->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mConstraintCoefficientsC->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsC->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsC(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        mConstraintCoefficientsC->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsC(const OrdinalType & aVectorIndex,
                                    const OrdinalType & aElementIndex,
                                    const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());
        assert(aElementIndex >= static_cast<OrdinalType>(0));
        assert(aElementIndex < mConstraintCoefficientsC->operator [](aVectorIndex).size());

        mConstraintCoefficientsC->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsC(const OrdinalType & aVectorIndex,
                                    const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        mConstraintCoefficientsC->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsC(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsC->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsC);
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS D
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsD() const
    {
        assert(mConstraintCoefficientsD.get() != nullptr);

        return (mConstraintCoefficientsD.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintCoefficientsD(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        return (mConstraintCoefficientsD->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsD(const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(mConstraintCoefficientsD->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mConstraintCoefficientsD->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsD->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsD(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        mConstraintCoefficientsD->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsD(const OrdinalType & aVectorIndex,
                                    const OrdinalType & aElementIndex,
                                    const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());
        assert(aElementIndex >= static_cast<OrdinalType>(0));
        assert(aElementIndex < mConstraintCoefficientsD->operator [](aVectorIndex).size());

        mConstraintCoefficientsD->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsD(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        mConstraintCoefficientsD->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsD(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsD->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsD);
    }
    // UPDATE DUAL PROBLEM DATA
    void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        // Update Current Objective Function Value
        mCurrentObjectiveFunctionValue = aDataMng.getCurrentObjectiveFunctionValue();

        // Update Current Constraint Values
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getCurrentConstraintValues(),
                      static_cast<ScalarType>(0),
                      mCurrentConstraintValues.operator*());

        // Update Moving Asymptotes
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma();
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mLowerAsymptotes);
        locus::update(static_cast<ScalarType>(-1), tCurrentSigma, static_cast<ScalarType>(1), *mLowerAsymptotes);
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mUpperAsymptotes);
        locus::update(static_cast<ScalarType>(1), tCurrentSigma, static_cast<ScalarType>(1), *mUpperAsymptotes);

        // Update Trial Control Bounds
        const ScalarType tScaleFactor = aDataMng.getDualProblemBoundsScaleFactor();
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mTrialControlLowerBounds);
        locus::update(-tScaleFactor, tCurrentSigma, static_cast<ScalarType>(1), *mTrialControlLowerBounds);
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mTrialControlUpperBounds);
        locus::update(tScaleFactor, tCurrentSigma, static_cast<ScalarType>(1), *mTrialControlUpperBounds);
    }

    // UPDATE PRIMAL PROBLEM DATA
    void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        return;
    }
    // EVALUATE DUAL OBJECTIVE FUNCTION
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                                 ScalarType aTolerance = std::numeric_limits<ScalarType>::max())
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        this->computeDualTimesConstraintCoefficientTerms(aDual);
        this->computeTrialControl(aDual);
        this->computeTrialAuxiliaryVariables(aDual);

        const ScalarType tObjectiveTerm = mObjectiveCoefficientR + (mTrialAuxiliaryVariableZ * mObjectiveCoefficientA)
                + (mEpsilon * mTrialAuxiliaryVariableZ * mTrialAuxiliaryVariableZ);

        const ScalarType tConstraintSummationTerm = this->computeConstraintContribution(aDual);

        ScalarType tMovingAsymptotesTerm = this->computeMovingAsymptotesContribution();

        // Add all contributions to dual objective function
        ScalarType tOutput = static_cast<ScalarType>(-1)
                * (tObjectiveTerm + tConstraintSummationTerm + tMovingAsymptotesTerm);

        return (tOutput);
    }
    // COMPUTE DUAL GRADIENT
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                         locus::MultiVector<ScalarType, OrdinalType> & aDualGradient)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tDualVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tDualGradient = aDualGradient[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tAuxiliaryVariableY = (*mTrialAuxiliaryVariableY)[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tCoefficientsR = (*mConstraintCoefficientsR)[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tCoefficientsA = (*mConstraintCoefficientsA)[tDualVectorIndex];

        OrdinalType tNumConstraints = tDual.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            tDualGradient[tConstraintIndex] = tCoefficientsR[tConstraintIndex] - tAuxiliaryVariableY[tConstraintIndex]
                    - (tCoefficientsA[tConstraintIndex] * mTrialAuxiliaryVariableZ);

            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();

            const OrdinalType tNumControlVectors = mTrialControl->getNumVectors();
            std::vector<ScalarType> tMyStorageOne(tNumControlVectors);
            std::vector<ScalarType> tMyStorageTwo(tNumControlVectors);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
                mControlWorkVectorTwo->fill(static_cast<ScalarType>(0));
                const locus::Vector<ScalarType, OrdinalType> & tMyTrialControl = (*mTrialControl)[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyLowerAsymptotes = (*mLowerAsymptotes)[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyUpperAsymptotes = (*mUpperAsymptotes)[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsP = tMyConstraintCoefficientsP[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsQ = tMyConstraintCoefficientsQ[tVectorIndex];

                const OrdinalType tNumControls = tMyTrialControl.size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    (*mControlWorkVectorOne)[tControlIndex] = tMyCoefficientsP[tControlIndex]
                            / (tMyUpperAsymptotes[tControlIndex] - tMyTrialControl[tControlIndex]);

                    (*mControlWorkVectorTwo)[tControlIndex] = tMyCoefficientsQ[tControlIndex]
                            / (tMyTrialControl[tControlIndex] - tMyLowerAsymptotes[tControlIndex]);
                }

                tMyStorageOne[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
                tMyStorageTwo[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorTwo.operator*());
            }

            const ScalarType tInitialValue = 0;
            const ScalarType tGlobalSumP = std::accumulate(tMyStorageOne.begin(), tMyStorageOne.end(), tInitialValue);
            const ScalarType tGlobalSumQ = std::accumulate(tMyStorageTwo.begin(), tMyStorageTwo.end(), tInitialValue);
            // Add contribution to dual gradient
            tDualGradient[tConstraintIndex] = static_cast<ScalarType>(-1)
                    * (tDualGradient[tConstraintIndex] + tGlobalSumP + tGlobalSumQ);
        }
    }
    // APPLY VECTOR TO DUAL HESSIAN
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        return;
    }

    // GET OPTIMAL TRIAL CONTROL FROM DUAL PROBLEM
    void getTrialControl(locus::MultiVector<ScalarType, OrdinalType> & aInput) const
    {
        locus::update(static_cast<ScalarType>(1), *mTrialControl, static_cast<ScalarType>(0), aInput);
    }
    void updateObjectiveCoefficients(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mObjectiveCoefficientR = mCurrentObjectiveFunctionValue;
        const ScalarType tGlobalizationFactor = aDataMng.getDualObjectiveGlobalizationFactor();
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma();

        const OrdinalType tNumVectors = tCurrentSigma.getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentSigma = tCurrentSigma[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentObjectiveGradient =
                    aDataMng.getCurrentObjectiveGradient(tVectorIndex);

            OrdinalType tNumControls = tMyCurrentSigma.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                        * tMyCurrentSigma[tControlIndex];
                (*mObjectiveCoefficientsP)(tVectorIndex, tControlIndex) = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ScalarType>(0), tMyCurrentObjectiveGradient[tControlIndex]);
                +((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ScalarType>(4));

                (*mObjectiveCoefficientsQ)(tVectorIndex, tControlIndex) = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ScalarType>(0), -tMyCurrentObjectiveGradient[tControlIndex])
                        + ((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ScalarType>(4));
                (*mControlWorkVectorOne)[tControlIndex] = ((*mObjectiveCoefficientsP)(tVectorIndex, tControlIndex)
                        + (*mObjectiveCoefficientsQ)(tVectorIndex, tControlIndex)) / tMyCurrentSigma[tControlIndex];
            }
            tStorage[tVectorIndex] = mControlReductionOperations->sum(*mControlWorkVectorOne);
        }

        const ScalarType tInitialValue = 0;
        const ScalarType tValue = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
        mObjectiveCoefficientR = mObjectiveCoefficientR - tValue;
    }
    void updateConstraintCoefficients(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tDualVectorIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tCurrentConstraintValues =
                mCurrentConstraintValues->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType, OrdinalType> & tGlobalizationFactor =
                aDataMng.getConstraintGlobalizationFactors(tDualVectorIndex);
        locus::Vector<ScalarType, OrdinalType> & tConstraintCoefficientsR =
                mConstraintCoefficientsR.operator*()[tDualVectorIndex];

        const OrdinalType tNumConstraints = tConstraintCoefficientsR.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            tConstraintCoefficientsR[tConstraintIndex] = tCurrentConstraintValues[tConstraintIndex];
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentConstraintGradients =
                    aDataMng.getCurrentConstraintGradients(tConstraintIndex);
            locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoeffP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoeffQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();
            assert(tCurrentConstraintGradients.getNumVectors() == aDataMng.getNumControlVectors());

            const OrdinalType tNumControlVectors = tCurrentConstraintGradients.getNumVectors();
            std::vector<ScalarType> tStorage(tNumControlVectors);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
                locus::Vector<ScalarType, OrdinalType> & tMyCoeffP = tMyConstraintCoeffP[tVectorIndex];
                locus::Vector<ScalarType, OrdinalType> & tMyCoeffQ = tMyConstraintCoeffQ[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = tCurrentConstraintGradients[tVectorIndex];

                const OrdinalType tNumControls = tMyCurrentGradient.size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ScalarType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                            * tMyCurrentSigma[tControlIndex];
                    tMyCoeffP[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                            * std::max(static_cast<ScalarType>(0), tMyCurrentGradient[tControlIndex])
                            + ((tGlobalizationFactor[tConstraintIndex] * tMyCurrentSigma[tControlIndex])
                                    / static_cast<ScalarType>(4));

                    tMyCoeffQ[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                            * std::max(static_cast<ScalarType>(0), -tMyCurrentGradient[tControlIndex])
                            + ((tGlobalizationFactor[tConstraintIndex] * tMyCurrentSigma[tControlIndex])
                                    / static_cast<ScalarType>(4));

                    (*mControlWorkVectorOne)[tControlIndex] = (tMyCoeffP[tControlIndex] + tMyCoeffQ[tControlIndex])
                            / tMyCurrentSigma[tControlIndex];
                }
                tStorage[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
            }

            const ScalarType tInitialValue = 0;
            const ScalarType tValue = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
            tConstraintCoefficientsR[tConstraintIndex] = tConstraintCoefficientsR[tConstraintIndex] - tValue;
        }
    }
    void initializeAuxiliaryVariables(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tDualVectorIndex = 0;
        locus::Vector<ScalarType> & tAuxiliaryVariablesY =
                mTrialAuxiliaryVariableY->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType> & tCoefficientsA =
                mConstraintCoefficientsA->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType, OrdinalType> & tCurrentConstraintValues =
                aDataMng.getCurrentConstraintValues(tDualVectorIndex);

        const OrdinalType tNumConstraints = mDualWorkVector->size();
        const ScalarType tMaxCoefficientA = mDualReductionOperations->max(tCoefficientsA);
        if(tMaxCoefficientA > static_cast<ScalarType>(0))
        {
            mDualWorkVector->fill(static_cast<ScalarType>(0));
            for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
            {
                if(tCoefficientsA[tIndex] > static_cast<ScalarType>(0))
                {
                    ScalarType tValue = std::max(static_cast<ScalarType>(0), tCurrentConstraintValues[tIndex]);
                    (*mDualWorkVector)[tIndex] = tValue / tCoefficientsA[tIndex];
                    tAuxiliaryVariablesY[tIndex] = 0;
                }
                else
                {
                    tAuxiliaryVariablesY[tIndex] =
                            std::max(static_cast<ScalarType>(0), tCurrentConstraintValues[tIndex]);
                }
            }
            mTrialAuxiliaryVariableZ = mDualReductionOperations->max(mDualWorkVector.operator*());
        }
        else
        {
            for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
            {
                tAuxiliaryVariablesY[tIndex] = tCurrentConstraintValues[tIndex];
            }
            mTrialAuxiliaryVariableZ = 0;
        }
    }
    void checkConstraintCoefficients()
    {
        const OrdinalType tDualVectorIndex = 0;
        ScalarType tMinCoeffA = mDualReductionOperations->min(mConstraintCoefficientsA->operator[](tDualVectorIndex));
        assert(tMinCoeffA >= static_cast<ScalarType>(0));
        ScalarType tMinCoeffC = mDualReductionOperations->min(mConstraintCoefficientsC->operator[](tDualVectorIndex));
        assert(tMinCoeffC >= static_cast<ScalarType>(0));
        ScalarType tMinCoeffD = mDualReductionOperations->min(mConstraintCoefficientsD->operator[](tDualVectorIndex));
        assert(tMinCoeffD >= static_cast<ScalarType>(0));
    }

private:
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        const OrdinalType tDualVectorIndex = 0;
        mDualWorkVector = aDataFactory.dual(tDualVectorIndex).create();

        const OrdinalType tControlVectorIndex = 0;
        mControlWorkVectorOne = aDataFactory.control(tControlVectorIndex).create();
        mControlWorkVectorTwo = aDataFactory.control(tControlVectorIndex).create();

        const OrdinalType tNumConstraints = aDataFactory.dual(tDualVectorIndex).size();
        mConstraintCoefficientsP.resize(tNumConstraints);
        mConstraintCoefficientsQ.resize(tNumConstraints);
        for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            mConstraintCoefficientsP[tIndex] = aDataFactory.control().create();
            mConstraintCoefficientsQ[tIndex] = aDataFactory.control().create();
        }

        locus::fill(static_cast<ScalarType>(0), mConstraintCoefficientsA.operator*());
        locus::fill(static_cast<ScalarType>(1), mConstraintCoefficientsD.operator*());
        locus::fill(static_cast<ScalarType>(1e3), mConstraintCoefficientsC.operator*());
    }
    ScalarType computeMovingAsymptotesContribution()
    {
        const OrdinalType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ScalarType> tMySumP(tNumVectors);
        std::vector<ScalarType> tMySumQ(tNumVectors);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
            mControlWorkVectorTwo->fill(static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyTrialControl = mTrialControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyLowerAsymptotes =
                    mLowerAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyUpperAsymptotes =
                    mUpperAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyObjectiveCoefficientsP =
                    mObjectiveCoefficientsP->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyObjectiveCoefficientsQ =
                    mObjectiveCoefficientsQ->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsP =
                    mDualTimesCoefficientsP->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsQ =
                    mDualTimesCoefficientsQ->operator[](tVectorIndex);

            const OrdinalType tNumControls = tMyTrialControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tNumerator = tMyObjectiveCoefficientsP[tControlIndex]
                        + tMyDualTimesCoefficientsP[tControlIndex];
                ScalarType tDenominator = tMyUpperAsymptotes[tControlIndex] - tMyTrialControl[tControlIndex];
                (*mControlWorkVectorOne)[tControlIndex] = tNumerator / tDenominator;

                tNumerator = tMyObjectiveCoefficientsQ[tControlIndex] + tMyDualTimesCoefficientsQ[tControlIndex];
                tDenominator = tMyTrialControl[tControlIndex] - tMyLowerAsymptotes[tControlIndex];
                (*mControlWorkVectorTwo)[tControlIndex] = tNumerator / tDenominator;
            }

            tMySumP[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
            tMySumQ[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorTwo.operator*());
        }

        const ScalarType tInitialValue = 0;
        const ScalarType tGlobalSumP = std::accumulate(tMySumP.begin(), tMySumP.end(), tInitialValue);
        const ScalarType tGlobalSumQ = std::accumulate(tMySumQ.begin(), tMySumQ.end(), tInitialValue);
        const ScalarType tMovingAsymptotesTerm = tGlobalSumP + tGlobalSumQ;

        return (tMovingAsymptotesTerm);
    }
    // Compute trial controls based on the following explicit expression:
    // \[ x(\lambda)=\frac{u_j^k\mathtt{b}^{1/2}+l_j^k\mathtt{a}^{1/2}}{(\mathtt{a}^{1/2}+\mathtt{b}^{1/2})} \],
    // where
    //      \[ \mathtt{a}=(p_{0j}+\lambda^{\intercal}p_j) \] and [ \mathtt{b}=(q_{0j}+\lambda^{\intercal}q_j) ]
    //      and j=1\dots,n_{x}
    // Here, x denotes the trial control vector
    void computeTrialControl(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        locus::update(static_cast<ScalarType>(1), *mObjectiveCoefficientsP, static_cast<ScalarType>(0), *mTermA);
        locus::update(static_cast<ScalarType>(1), *mDualTimesCoefficientsP, static_cast<ScalarType>(1), *mTermA);
        locus::update(static_cast<ScalarType>(1), *mObjectiveCoefficientsQ, static_cast<ScalarType>(0), *mTermB);
        locus::update(static_cast<ScalarType>(1), *mDualTimesCoefficientsQ, static_cast<ScalarType>(1), *mTermB);

        const OrdinalType tNumControlVectors = mTrialControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyTrialControl = mTrialControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyLowerAsymptotes =
                    mLowerAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyUpperAsymptotes =
                    mUpperAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyLowerBounds =
                    mTrialControlLowerBounds->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyUpperBounds =
                    mTrialControlUpperBounds->operator[](tVectorIndex);

            OrdinalType tNumControls = tMyTrialControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tSqrtTermA = std::sqrt(mTermA->operator()(tVectorIndex, tControlIndex));
                ScalarType tSqrtTermB = std::sqrt(mTermB->operator()(tVectorIndex, tControlIndex));
                ScalarType tNumerator = (tMyLowerAsymptotes[tControlIndex] * tSqrtTermA)
                        + (tMyUpperAsymptotes[tControlIndex] * tSqrtTermB);
                ScalarType tDenominator = (tSqrtTermA + tSqrtTermB);
                tMyTrialControl[tControlIndex] = tNumerator / tDenominator;
                // Project trial control to feasible set
                tMyTrialControl[tControlIndex] =
                        std::max(tMyTrialControl[tControlIndex], tMyLowerBounds[tControlIndex]);
                tMyTrialControl[tControlIndex] =
                        std::min(tMyTrialControl[tControlIndex], tMyUpperBounds[tControlIndex]);
            }
        }
    }
    /*!
     * Update auxiliary variables based on the following expression:
     *  \[ y_i(\lambda)=\frac{\lambda_i-c_i}{2d_i} \]
     *  and
     *  \[ z(\lambda)=\frac{\lambda^{\intercal}a-a_0}{2\varepsilon} \]
     */
    void computeTrialAuxiliaryVariables(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        OrdinalType tNumVectors = aDual.getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyDual = aDual[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsC = (*mConstraintCoefficientsC)[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsD = (*mConstraintCoefficientsD)[tVectorIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyAuxiliaryVariablesY = (*mTrialAuxiliaryVariableY)[tVectorIndex];

            const OrdinalType tNumDual = tMyAuxiliaryVariablesY.size();
            for(OrdinalType tIndex = 0; tIndex < tNumDual; tIndex++)
            {
                ScalarType tDualMinusConstraintCoefficientC = tMyDual[tIndex] - tMyCoefficientsC[tIndex];
                tMyAuxiliaryVariablesY[tIndex] = tDualMinusConstraintCoefficientC / tMyCoefficientsD[tIndex];
                // Project auxiliary variables Y to feasible set (Y >= 0)
                tMyAuxiliaryVariablesY[tIndex] = std::max(tMyAuxiliaryVariablesY[tIndex], static_cast<ScalarType>(0));
            }
        }
        ScalarType tDualDotConstraintCoefficientA = locus::dot(aDual, *mConstraintCoefficientsA);
        mTrialAuxiliaryVariableZ = (tDualDotConstraintCoefficientA - mObjectiveCoefficientA)
                / (static_cast<ScalarType>(2) * mEpsilon);
        // Project auxiliary variables Z to feasible set (Z >= 0)
        mTrialAuxiliaryVariableZ = std::max(mTrialAuxiliaryVariableZ, static_cast<ScalarType>(0));
    }
    /*! Compute: \sum_{i=1}^{m}\left( c_iy_i + \frac{1}{2}d_iy_i^2 \right) -
     * \lambda^{T}y - (\lambda^{T}a)z + \lambda^{T}r, where m is the number of constraints
     **/
    ScalarType computeConstraintContribution(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tNumVectors = aDual.getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors);
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mDualWorkVector->fill(static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyDual = aDual[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsC = (*mConstraintCoefficientsC)[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsD = (*mConstraintCoefficientsD)[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyAuxiliaryVariablesY =
                    (*mTrialAuxiliaryVariableY)[tVectorIndex];

            const OrdinalType tNumDuals = tMyDual.size();
            for(OrdinalType tIndex = 0; tIndex < tNumDuals; tIndex++)
            {
                ScalarType tValueOne = tMyCoefficientsC[tIndex] * tMyAuxiliaryVariablesY[tIndex];
                ScalarType tValueTwo = tMyCoefficientsD[tIndex] * tMyAuxiliaryVariablesY[tIndex]
                        * tMyAuxiliaryVariablesY[tIndex];
                (*mDualWorkVector)[tIndex] = tValueOne + tValueTwo;
            }

            tStorage[tVectorIndex] = mDualReductionOperations->sum(mDualWorkVector.operator*());
        }

        const ScalarType tInitialValue = 0;
        ScalarType tConstraintSummationTerm = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);

        // Add additional contributions to inequality summation term
        ScalarType tDualDotConstraintCoeffR = locus::dot(aDual, *mConstraintCoefficientsR);
        ScalarType tDualDotConstraintCoeffA = locus::dot(aDual, *mConstraintCoefficientsA);
        ScalarType tDualDotTrialAuxiliaryVariableY = locus::dot(aDual, *mTrialAuxiliaryVariableY);
        ScalarType tOutput = tConstraintSummationTerm - tDualDotTrialAuxiliaryVariableY
                - (tDualDotConstraintCoeffA * mTrialAuxiliaryVariableZ) + tDualDotConstraintCoeffR;

        return (tOutput);
    }
    /*
     * Compute \lambda_j\times{p}_j and \lambda_j\times{q}_j, where
     * j=1,\dots,N_{c}. Here, N_{c} denotes the number of constraints.
     **/
    void computeDualTimesConstraintCoefficientTerms(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        const OrdinalType tDualVectorIndex = 0;
        locus::fill(static_cast<ScalarType>(0), mDualTimesCoefficientsP.operator*());
        locus::fill(static_cast<ScalarType>(0), mDualTimesCoefficientsQ.operator*());
        const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tDualVectorIndex];

        const ScalarType tBeta = 1;
        const OrdinalType tNumConstraints = tDual.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();

            const OrdinalType tNumControlVectors = tMyConstraintCoefficientsP.getNumVectors();
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsP =
                        tMyConstraintCoefficientsP[tVectorIndex];
                locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsP =
                        (*mDualTimesCoefficientsP)[tVectorIndex];
                tMyDualTimesCoefficientsP.update(tDual[tConstraintIndex], tMyCoefficientsP, tBeta);

                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsQ =
                        tMyConstraintCoefficientsQ[tVectorIndex];
                locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsQ =
                        (*mDualTimesCoefficientsQ)[tVectorIndex];
                tMyDualTimesCoefficientsQ.update(tDual[tConstraintIndex], tMyCoefficientsQ, tBeta);
            }
        }
    }

private:
    ScalarType mEpsilon;
    ScalarType mObjectiveCoefficientA;
    ScalarType mObjectiveCoefficientR;
    ScalarType mTrialAuxiliaryVariableZ;
    ScalarType mCurrentObjectiveFunctionValue;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkVector;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVectorOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVectorTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTermA;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTermB;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mLowerAsymptotes;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mUpperAsymptotes;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualTimesCoefficientsP;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualTimesCoefficientsQ;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mObjectiveCoefficientsP;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mObjectiveCoefficientsQ;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControlUpperBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsA;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsC;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsD;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsR;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialAuxiliaryVariableY;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintValues;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

    std::vector<std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>>> mConstraintCoefficientsP;
    std::vector<std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>>> mConstraintCoefficientsQ;

private:
    DualProblemStageMng(const locus::DualProblemStageMng<ScalarType, OrdinalType> & aRhs);
    locus::DualProblemStageMng<ScalarType, OrdinalType> & operator=(const locus::DualProblemStageMng<ScalarType, OrdinalType> & aRhs);
};

struct ccsa
{
    enum method_t
    {
        MMA = 1,
        GCMMA = 2
    };

    enum stop_t
    {
        STATIONARITY_TOLERANCE = 1,
        KKT_CONDITIONS_TOLERANCE = 2,
        CONTROL_STAGNATION = 3,
        OBJECTIVE_STAGNATION = 4,
        MAX_NUMBER_ITERATIONS = 5,
        OPTIMALITY_AND_FEASIBILITY_MET = 6,
        NOT_CONVERGED = 7,
    };
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxDualSolver
{
public:
    ConservativeConvexSeparableAppxDualSolver()
    {
    }
    virtual ~ConservativeConvexSeparableAppxDualSolver()
    {
    }

private:
    ConservativeConvexSeparableAppxDualSolver(const locus::ConservativeConvexSeparableAppxDualSolver<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableAppxDualSolver<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableAppxDualSolver<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableApproximation
{
public:
    virtual ~ConservativeConvexSeparableApproximation()
    {
    }

    virtual void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
                       locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class DualProblemSolver
{
public:
    virtual ~DualProblemSolver()
    {
    }

    virtual void solve(locus::MultiVector<ScalarType, OrdinalType> & aDual,
                       locus::MultiVector<ScalarType, OrdinalType> & aTrialControl) = 0;
    virtual void update(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void updateObjectiveCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void updateConstraintCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientDualSolver : public locus::DualProblemSolver<ScalarType, OrdinalType>
{
public:
    explicit NonlinearConjugateGradientDualSolver(const locus::DataFactory<ScalarType, OrdinalType> & aPrimalDataFactory) :
            mDualWork(aPrimalDataFactory.dual().create()),
            mDualInitialGuess(aPrimalDataFactory.dual().create()),
            mDualDataFactory(std::make_shared<locus::DataFactory<ScalarType, OrdinalType>>()),
            mDualStageMng(),
            mDualAlgorithm(),
            mDualDataMng()
    {
        this->initialize(aPrimalDataFactory);
    }
    virtual ~NonlinearConjugateGradientDualSolver()
    {
    }

    void solve(locus::MultiVector<ScalarType, OrdinalType> & aDual,
               locus::MultiVector<ScalarType, OrdinalType> & aTrialControl)
    {
        this->reset();
        mDualDataMng->setInitialGuess(mDualInitialGuess.operator*());
        mDualAlgorithm->solve();

        mDualStageMng->getTrialControl(aTrialControl);
        const locus::MultiVector<ScalarType, OrdinalType> & tDualSolution = mDualDataMng->getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tDualSolution, static_cast<ScalarType>(0), aDual);
        // Store dual solution and use it as the initial guess for the next iteration.
        locus::update(static_cast<ScalarType>(1), tDualSolution, static_cast<ScalarType>(0), mDualInitialGuess.operator*());
    }

    void update(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->update(aDataMng);
    }
    void updateObjectiveCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->updateObjectiveCoefficients(aDataMng);
    }
    void updateConstraintCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->updateConstraintCoefficients(aDataMng);
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->initializeAuxiliaryVariables(aDataMng);
    }

private:
    void reset()
    {
        const ScalarType tValue = 0;
        mDualDataMng->setCurrentObjectiveFunctionValue(tValue);
        mDualDataMng->setPreviousObjectiveFunctionValue(tValue);

        locus::fill(tValue, mDualWork.operator*());
        mDualDataMng->setTrialStep(mDualWork.operator*());
        mDualDataMng->setCurrentControl(mDualWork.operator*());
        mDualDataMng->setPreviousControl(mDualWork.operator*());
        mDualDataMng->setCurrentGradient(mDualWork.operator*());
        mDualDataMng->setPreviousGradient(mDualWork.operator*());
    }
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aPrimalDataFactory)
    {
        mDualDataFactory->allocateControl(mDualWork.operator*());
        mDualDataMng = std::make_shared<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>>(mDualDataFactory.operator*());

        ScalarType tValue = 0;
        mDualDataMng->setControlLowerBounds(tValue);
        tValue = std::numeric_limits<ScalarType>::max();
        mDualDataMng->setControlLowerBounds(tValue);

        mDualStageMng = std::make_shared<locus::DualProblemStageMng<ScalarType, OrdinalType>>(aPrimalDataFactory);
        mDualAlgorithm = std::make_shared<locus::NonlinearConjugateGradient<ScalarType, OrdinalType>>(mDualDataFactory, mDualDataMng, mDualStageMng);
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualInitialGuess;

    std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> mDualDataFactory;
    std::shared_ptr<locus::DualProblemStageMng<ScalarType, OrdinalType>> mDualStageMng;
    std::shared_ptr<locus::NonlinearConjugateGradient<ScalarType, OrdinalType>> mDualAlgorithm;
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> mDualDataMng;

private:
    NonlinearConjugateGradientDualSolver(const locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class MethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>
{
public:
    explicit MethodMovingAsymptotes(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mConstraintValues(aDataFactory.dual().create()),
            mDualSolver(std::make_shared<locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType>>(aDataFactory))
    {
    }
    virtual ~MethodMovingAsymptotes()
    {
    }

    void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        // NOTE: REMBER THAT THE GLOBALIZATION FACTORS FOR BOTH OBJECTIVE AND CONSTRAINTS ARE SET TO ZERO IF USING
        //       THE MMA APPROACH. THE MMA METHOD IS DETECTED AT THE OUTTER LOOP LEVEL (NOT THE SUBPROBLEM LEVEL)
        //       AND THE VALUES SET INSIDE THE INITIALZE FUNCTION IN THE ALGORITHM CLASS. DEFAULT VALUES ARE ONLY
        //       SET FOR THE GCMMA CASE.
        mDualSolver->update(aDataMng);
        mDualSolver->updateObjectiveCoefficients(aDataMng);
        mDualSolver->updateConstraintCoefficients(aDataMng);
        mDualSolver->solve(mTrialDual.operator*(), mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());
        locus::bounds::computeActiveAndInactiveSets(mTrialControl.operator*(),
                                                    tLowerBounds,
                                                    tUpperBounds,
                                                    mActiveSet.operator*(),
                                                    mInactiveSet.operator*());
        aDataMng.setActiveSet(mActiveSet.operator*());
        aDataMng.setInactiveSet(mInactiveSet.operator*());

        ScalarType tObjectiveFunctionValue = aPrimalProblemStageMng.evaluateObjective(mTrialControl.operator*());
        aDataMng.setCurrentObjectiveFunctionValue(tObjectiveFunctionValue);
        aPrimalProblemStageMng.evaluateConstraints(mTrialControl.operator*(), mConstraintValues.operator*());
        aDataMng.setCurrentConstraintValues(mConstraintValues.operator*());
        aDataMng.setCurrentControl(mTrialControl.operator*());
        aDataMng.setDual(mTrialDual.operator*());
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->initializeAuxiliaryVariables(aDataMng);
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintValues;
    std::shared_ptr<locus::DualProblemSolver<ScalarType, OrdinalType>> mDualSolver;

private:
    MethodMovingAsymptotes(const locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
    locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & operator=(const locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class GloballyConvergentMethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>
{
public:
    explicit GloballyConvergentMethodMovingAsymptotes(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(10),
            mNumIterationsDone(0),
            mStagnationTolerance(1e-6),
            mMinObjectiveGlobalizationFactor(1e-5),
            mCurrentTrialObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousTrialObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mKarushKuhnTuckerConditionsTolerance(1e-6),
            mStoppingCriterion(locus::ccsa::stop_t::NOT_CONVERGED),
            mControlWorkOne(),
            mControlWorkTwo(),
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mDeltaControl(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mTrialConstraintValues(aDataFactory.dual().create()),
            mMinConstraintGlobalizationFactors(aDataFactory.dual().create()),
            mDualSolver(std::make_shared<locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType>>(aDataFactory)),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize();
    }
    virtual ~GloballyConvergentMethodMovingAsymptotes()
    {
    }

    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    void setNumIterationsDone(const OrdinalType & aInput)
    {
        mNumIterationsDone = aInput;
    }

    ScalarType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    void setStagnationTolerance(const ScalarType & aInput) const
    {
        mStagnationTolerance = aInput;
    }
    ScalarType getKarushKuhnTuckerConditionsTolerance() const
    {
        return (mKarushKuhnTuckerConditionsTolerance);
    }
    void setKarushKuhnTuckerConditionsTolerance(const ScalarType & aInput)
    {
        mKarushKuhnTuckerConditionsTolerance = aInput;
    }

    locus::ccsa::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    void setStoppingCriterion(const locus::ccsa::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->update(aDataMng);

        OrdinalType tIterations = 0;
        this->setNumIterationsDone(tIterations);
        while(1)
        {
            mDualSolver->updateObjectiveCoefficients(aDataMng);
            mDualSolver->updateConstraintCoefficients(aDataMng);
            mDualSolver->solve(mTrialDual.operator*(), mTrialControl.operator*());

            const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
            const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
            locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());
            locus::bounds::computeActiveAndInactiveSets(mTrialControl.operator*(),
                                                        tLowerBounds,
                                                        tUpperBounds,
                                                        mActiveSet.operator*(),
                                                        mInactiveSet.operator*());
            aDataMng.setActiveSet(mActiveSet.operator*());
            aDataMng.setInactiveSet(mInactiveSet.operator*());

            mCurrentTrialObjectiveFunctionValue = aPrimalProblemStageMng.evaluateObjective(mTrialControl.operator*());
            aPrimalProblemStageMng.evaluateConstraints(mTrialControl.operator*(), mTrialConstraintValues.operator*());

            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
            locus::update(static_cast<ScalarType>(1), *mTrialControl, static_cast<ScalarType>(0), *mDeltaControl);
            locus::update(static_cast<ScalarType>(-1), tCurrentControl, static_cast<ScalarType>(1), *mDeltaControl);
            this->updateObjectiveGlobalizationFactor(aDataMng);
            this->updateConstraintGlobalizationFactors(aDataMng);

            tIterations++;
            this->setNumIterationsDone(tIterations);
            if(this->checkStoppingCriteria(aDataMng) == true)
            {
                break;
            }
        }

        aDataMng.setDual(mTrialDual.operator*());
        aDataMng.setCurrentControl(mTrialControl.operator*());
        aDataMng.setCurrentObjectiveFunctionValue(mCurrentTrialObjectiveFunctionValue);
        aDataMng.setCurrentConstraintValues(mTrialConstraintValues.operator*());
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->initializeAuxiliaryVariables(aDataMng);
    }

private:
    void initialize()
    {
        const OrdinalType tVectorIndex = 0;
        mControlWorkOne = mTrialControl->operator[](tVectorIndex).create();
        mControlWorkTwo = mTrialControl->operator[](tVectorIndex).create();
        locus::fill(static_cast<ScalarType>(1e-5), mMinConstraintGlobalizationFactors.operator*());
    }
    void updateObjectiveGlobalizationFactor(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        locus::fill(static_cast<ScalarType>(0), mControlWorkOne.operator*());
        locus::fill(static_cast<ScalarType>(0), mControlWorkTwo.operator*());

        const OrdinalType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumVectors, 0);
        std::vector<ScalarType> tStorageTwo(tNumVectors, 0);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tWorkVectorOne = mControlWorkOne->operator[](tVectorIndex);
            locus::Vector<ScalarType, OrdinalType> & tWorkVectorTwo = mControlWorkOne->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tCurrentGradient = aDataMng.getCurrentObjectiveGradient(tVectorIndex);
            assert(tWorkVectorOne.size() == tWorkVectorTwo.size());

            const OrdinalType tNumControls = tWorkVectorOne.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tNumerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                ScalarType tDenominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                        - (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]);
                tWorkVectorOne[tControlIndex] = tNumerator / tDenominator;

                tNumerator = ((tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                        * tCurrentGradient[tControlIndex] * tDeltaControl[tControlIndex])
                        + (tCurrentSigma[tControlIndex] * std::abs(tCurrentGradient[tControlIndex])
                                * (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]));
                tWorkVectorTwo[tControlIndex] = tNumerator / tDenominator;
            }

            tStorageOne[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorOne);
            tStorageTwo[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorTwo);
        }

        const ScalarType tInitialValue = 0;
        ScalarType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        tFunctionEvaluationW = static_cast<ScalarType>(0.5) * tFunctionEvaluationW;
        ScalarType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
        const ScalarType tCurrentObjectiveValue = aDataMng.getCurrentObjectiveFunctionValue();
        tFunctionEvaluationV = tCurrentObjectiveValue + tFunctionEvaluationV;

        ScalarType tGlobalizationFactor = aDataMng.getDualObjectiveGlobalizationFactor();
        const ScalarType tCcsaFunctionValue = tFunctionEvaluationV + (tGlobalizationFactor * tFunctionEvaluationW);

        const ScalarType tActualOverPredictedReduction = (mCurrentTrialObjectiveFunctionValue - tCcsaFunctionValue)
                / tFunctionEvaluationW;
        if(tActualOverPredictedReduction > static_cast<ScalarType>(0))
        {
            ScalarType tValueOne = static_cast<ScalarType>(10) * tGlobalizationFactor;
            ScalarType tValueTwo = static_cast<ScalarType>(1.1)
                    * (tGlobalizationFactor + tActualOverPredictedReduction);
            tGlobalizationFactor = std::min(tValueOne, tValueTwo);
        }

        aDataMng.setDualObjectiveGlobalizationFactor(tGlobalizationFactor);
    }
    void updateConstraintGlobalizationFactors(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<OrdinalType>(1));
        const OrdinalType tNumDualVectors = 1;
        const locus::Vector<ScalarType, OrdinalType> & tConstraintValues =
                aDataMng.getCurrentConstraintValues(tNumDualVectors);
        locus::Vector<ScalarType, OrdinalType> & tGlobalizationFactors =
                aDataMng.getConstraintGlobalizationFactors(tNumDualVectors);

        const OrdinalType tNumConstraints = aDataMng.getNumConstraints();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            locus::fill(static_cast<ScalarType>(0), mControlWorkOne.operator*());
            locus::fill(static_cast<ScalarType>(0), mControlWorkTwo.operator*());

            const OrdinalType tNumVectors = mTrialControl->getNumVectors();
            std::vector<ScalarType> tStorageOne(tNumVectors, 0);
            std::vector<ScalarType> tStorageTwo(tNumVectors, 0);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                locus::Vector<ScalarType, OrdinalType> & tWorkVectorOne = mControlWorkOne->operator[](tVectorIndex);
                locus::Vector<ScalarType, OrdinalType> & tWorkVectorTwo = mControlWorkOne->operator[](tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tConstraintGradient =
                        aDataMng.getCurrentConstraintGradients(tConstraintIndex, tVectorIndex);
                assert(tDeltaControl.size() == tWorkVectorOne.size());
                assert(tWorkVectorOne.size() == tWorkVectorTwo.size());
                assert(tConstraintGradient.size() == tDeltaControl.size());

                const OrdinalType tNumControls = tWorkVectorOne.size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ScalarType numerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                    ScalarType denominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                            - (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]);
                    tWorkVectorOne[tControlIndex] = numerator / denominator;

                    numerator = ((tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                            * tConstraintGradient[tControlIndex] * tDeltaControl[tControlIndex])
                            + (tCurrentSigma[tControlIndex] * std::abs(tConstraintGradient[tControlIndex])
                                    * (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]));
                    tWorkVectorTwo[tControlIndex] = numerator / denominator;
                }

                tStorageOne[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorOne);
                tStorageTwo[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorTwo);
            }

            const ScalarType tInitialValue = 0;
            ScalarType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
            tFunctionEvaluationW = static_cast<ScalarType>(0.5) * tFunctionEvaluationW;

            ScalarType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
            tFunctionEvaluationV = tFunctionEvaluationV + tConstraintValues[tConstraintIndex];

            ScalarType tCcsaFunctionValue = tFunctionEvaluationV + (tGlobalizationFactors[tConstraintIndex] * tFunctionEvaluationW);
            ScalarType tActualOverPredictedReduction = (tConstraintValues[tConstraintIndex] - tCcsaFunctionValue) / tFunctionEvaluationW;

            if(tActualOverPredictedReduction > static_cast<ScalarType>(0))
            {
                ScalarType tValueOne = static_cast<ScalarType>(10) * tGlobalizationFactors[tConstraintIndex];
                ScalarType tValueTwo = static_cast<ScalarType>(1.1)
                        * (tGlobalizationFactors[tConstraintIndex] + tActualOverPredictedReduction);
                tGlobalizationFactors[tConstraintIndex] = std::min(tValueOne, tValueTwo);
            }
        }
    }
    bool checkStoppingCriteria(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        aDataMng.computeKarushKuhnTuckerConditionsInexactness(mTrialControl.operator*(), mTrialDual.operator*());
        const ScalarType t_KKT_ConditionsInexactness = aDataMng.getKarushKuhnTuckerConditionsInexactness();

        const ScalarType tObjectiveStagnation =
                std::abs(mCurrentTrialObjectiveFunctionValue - mPreviousTrialObjectiveFunctionValue);

        const OrdinalType tNumIterationsDone = this->getNumIterationsDone();

        bool tStop = false;
        if(tNumIterationsDone >= this->getMaxNumIterations())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::MAX_NUMBER_ITERATIONS);
        }
        else if(t_KKT_ConditionsInexactness < this->getKarushKuhnTuckerConditionsTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::KKT_CONDITIONS_TOLERANCE);
        }
        else if(tObjectiveStagnation < this->getStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::OBJECTIVE_STAGNATION);
        }

        return (tStop);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mStagnationTolerance;
    ScalarType mMinObjectiveGlobalizationFactor;
    ScalarType mCurrentTrialObjectiveFunctionValue;
    ScalarType mPreviousTrialObjectiveFunctionValue;
    ScalarType mKarushKuhnTuckerConditionsTolerance;

    locus::ccsa::stop_t mStoppingCriterion;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDeltaControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialConstraintValues;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mMinConstraintGlobalizationFactors;

    std::shared_ptr<locus::DualProblemSolver<ScalarType, OrdinalType>> mDualSolver;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    GloballyConvergentMethodMovingAsymptotes(const locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
    locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & operator=(const locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableApproximationsAlgorithm
{
public:
    ConservativeConvexSeparableApproximationsAlgorithm(const std::shared_ptr<locus::PrimalProblemStageMng<ScalarType, OrdinalType>> & aPrimalStageMng,
                                                       const std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType>> & aDataMng,
                                                       const std::shared_ptr<locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>> & aSubProblem) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mOptimalityTolerance(1e-4),
            mStagnationTolerance(1e-3),
            mFeasibilityTolerance(1e-4),
            mStationarityTolerance(1e-4),
            mObjectiveStagnationTolerance(1e-8),
            mMovingAsymptoteExpansionFactor(1.2),
            mMovingAsymptoteContractionFactor(0.4),
            mKarushKuhnTuckerConditionsTolerance(1e-4),
            mMovingAsymptoteUpperBoundScaleFactor(10),
            mMovingAsymptoteLowerBoundScaleFactor(0.01),
            mStoppingCriterion(locus::ccsa::stop_t::NOT_CONVERGED),
            mDualWork(),
            mControlWork(),
            mPreviousSigma(),
            mAntepenultimateControl(),
            mWorkMultiVectorList(),
            mPrimalStageMng(aPrimalStageMng),
            mDataMng(aDataMng),
            mSubProblem(aSubProblem)
    {
        this->initialize(aDataMng.operator*());
    }
    ~ConservativeConvexSeparableApproximationsAlgorithm()
    {
    }

    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }

    ScalarType getOptimalityTolerance() const
    {
        return (mOptimalityTolerance);
    }
    void setOptimalityTolerance(const ScalarType & aInput)
    {
        mOptimalityTolerance = aInput;
    }
    ScalarType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    void setStagnationTolerance(const ScalarType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    ScalarType getFeasibilityTolerance() const
    {
        return (mFeasibilityTolerance);
    }
    void setFeasibilityTolerance(const ScalarType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }
    ScalarType getStationarityTolerance() const
    {
        return (mStationarityTolerance);
    }
    void setStationarityTolerance(const ScalarType & aInput)
    {
        mStationarityTolerance = aInput;
    }
    ScalarType getObjectiveStagnationTolerance() const
    {
        return (mObjectiveStagnationTolerance);
    }
    void setObjectiveStagnationTolerance(const ScalarType & aInput)
    {
        mObjectiveStagnationTolerance = aInput;
    }
    ScalarType getMovingAsymptoteExpansionFactor() const
    {
        return (mMovingAsymptoteExpansionFactor);
    }
    void setMovingAsymptoteExpansionFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteExpansionFactor = aInput;
    }
    ScalarType getMovingAsymptoteContractionFactor() const
    {
        return (mMovingAsymptoteContractionFactor);
    }
    void setMovingAsymptoteContractionFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteContractionFactor = aInput;
    }
    ScalarType getKarushKuhnTuckerConditionsTolerance() const
    {
        return (mKarushKuhnTuckerConditionsTolerance);
    }
    void setKarushKuhnTuckerConditionsTolerance(const ScalarType & aInput)
    {
        mKarushKuhnTuckerConditionsTolerance = aInput;
    }
    ScalarType getMovingAsymptoteUpperBoundScaleFactor() const
    {
        return (mMovingAsymptoteUpperBoundScaleFactor);
    }
    void setMovingAsymptoteUpperBoundScaleFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteUpperBoundScaleFactor = aInput;
    }
    ScalarType getMovingAsymptoteLowerBoundScaleFactor() const
    {
        return (mMovingAsymptoteLowerBoundScaleFactor);
    }
    void setMovingAsymptoteLowerBoundScaleFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteLowerBoundScaleFactor = aInput;
    }

    locus::ccsa::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    void setStoppingCriterion(const locus::ccsa::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    void solve()
    {
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        const ScalarType tCurrentObjectiveFunctionValue = mPrimalStageMng->evaluateObjective(tControl);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mPrimalStageMng->evaluateConstraints(tControl, mDualWork.operator*());
        mDataMng->setDual(mDualWork.operator*());
        mSubProblem->initializeAuxiliaryVariables(mDataMng.operator*());

        mNumIterationsDone = 0;
        while(1)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
            mPrimalStageMng->computeGradient(tCurrentControl, mControlWork.operator*());
            mDataMng->setCurrentObjectiveGradient(mControlWork.operator*());
            mPrimalStageMng->computeConstraintGradients(tCurrentControl, mWorkMultiVectorList.operator*());
            mDataMng->setCurrentConstraintGradients(mWorkMultiVectorList.operator*());

            if(this->checkStoppingCriteria() == true)
            {
                break;
            }

            this->updateSigmaParameters();

            const locus::MultiVector<ScalarType, OrdinalType> & tPreviousControl = mDataMng->getPreviousControl();
            locus::update(static_cast<ScalarType>(1),
                          tPreviousControl,
                          static_cast<ScalarType>(0),
                          mAntepenultimateControl.operator*());
            locus::update(static_cast<ScalarType>(1),
                          tCurrentControl,
                          static_cast<ScalarType>(0),
                          mControlWork.operator*());
            mDataMng->setPreviousControl(mControlWork.operator*());

            mSubProblem->solve(mPrimalStageMng.operator*(), mDataMng.operator*());

            mNumIterationsDone++;
        }
    }

private:
    void initialize(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualWork = aDataMng.getDual().create();
        mControlWork = aDataMng.getCurrentControl().create();
        mPreviousSigma = aDataMng.getCurrentControl().create();
        mAntepenultimateControl = aDataMng.getCurrentControl().create();
        mWorkMultiVectorList = aDataMng.getCurrentConstraintGradients().create();
    }
    bool checkStoppingCriteria()
    {
        bool tStop = false;

        mDataMng->computeStagnationMeasure();
        mDataMng->computeFeasibilityMeasure();
        mDataMng->computeStationarityMeasure();
        mDataMng->computeNormProjectedGradient();
        mDataMng->computeObjectiveStagnationMeasure();

        const locus::MultiVector<ScalarType, OrdinalType> & tDual = mDataMng->getDual();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        mDataMng->computeKarushKuhnTuckerConditionsInexactness(tControl, tDual);

        const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
        const ScalarType tFeasibilityMeasure = mDataMng->getFeasibilityMeasure();
        const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ScalarType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
        const ScalarType tObjectiveStagnationMeasure = mDataMng->getObjectiveStagnationMeasure();
        const ScalarType t_KKT_ConditionsInexactness = mDataMng->getKarushKuhnTuckerConditionsInexactness();

        if(tStagnationMeasure < this->getStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::CONTROL_STAGNATION);
        }
        else if(tStationarityMeasure < this->getStationarityTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::STATIONARITY_TOLERANCE);
        }
        else if( (tFeasibilityMeasure < this->getFeasibilityTolerance())
                && (tNormProjectedGradient < this->getOptimalityTolerance()) )
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::OPTIMALITY_AND_FEASIBILITY_MET);
        }
        else if(t_KKT_ConditionsInexactness < this->getKarushKuhnTuckerConditionsTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::KKT_CONDITIONS_TOLERANCE);
        }
        else if(mNumIterationsDone < this->getMaxNumIterations())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::MAX_NUMBER_ITERATIONS);
        }
        else if(tObjectiveStagnationMeasure < this->getObjectiveStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::OBJECTIVE_STAGNATION);
        }

        return (tStop);
    }
    void updateSigmaParameters()
    {
        assert(mControlWork.get() != nullptr);
        assert(mPreviousSigma.get() != nullptr);

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = mDataMng->getCurrentSigma();
        locus::update(static_cast<ScalarType>(1), tCurrentSigma, static_cast<ScalarType>(0), *mPreviousSigma);

        const OrdinalType tNumIterationsDone = this->getNumIterationsDone();
        if(tNumIterationsDone < static_cast<OrdinalType>(2))
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = mDataMng->getControlUpperBounds();
            locus::update(static_cast<ScalarType>(1), tUpperBounds, static_cast<ScalarType>(0), *mControlWork);
            const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = mDataMng->getControlLowerBounds();
            locus::update(static_cast<ScalarType>(-1), tLowerBounds, static_cast<ScalarType>(1), *mControlWork);
            locus::scale(static_cast<ScalarType>(0.5), mControlWork.operator*());
            mDataMng->setCurrentSigma(mControlWork.operator*());
        }
        else
        {
            const OrdinalType tNumVectors = mControlWork->getNumVectors();
            locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                const ScalarType tExpansionFactor = this->getMovingAsymptoteExpansionFactor();
                const ScalarType tContractionFactor = this->getMovingAsymptoteContractionFactor();

                const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
                const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = mDataMng->getControlUpperBounds();
                const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = mDataMng->getControlLowerBounds();
                const locus::MultiVector<ScalarType, OrdinalType> & tPreviousControl = mDataMng->getPreviousControl();
                const locus::MultiVector<ScalarType, OrdinalType> & tPreviousSigma = mPreviousSigma->operator[](tVectorIndex);
                const locus::MultiVector<ScalarType, OrdinalType> & tAntepenultimateControl = mAntepenultimateControl->operator[](tVectorIndex);

                const OrdinalType tNumberControls = mControlWork->operator[](tVectorIndex).size();
                locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = mControlWork->operator[](tVectorIndex);
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumberControls; tControlIndex++)
                {
                    ScalarType tValue = (tCurrentControl[tControlIndex] - tPreviousControl[tControlIndex])
                            * (tPreviousControl[tControlIndex] - tAntepenultimateControl[tControlIndex]);
                    if(tValue > static_cast<ScalarType>(0))
                    {
                        tCurrentSigma[tControlIndex] = tExpansionFactor * tPreviousSigma[tControlIndex];
                    }
                    else if(tValue < static_cast<ScalarType>(0))
                    {
                        tCurrentSigma[tControlIndex] = tContractionFactor * tPreviousSigma[tControlIndex];
                    }
                    else
                    {
                        tCurrentSigma[tControlIndex] = tPreviousSigma[tControlIndex];
                    }
                    // check that lower bound is satisfied
                    const ScalarType tLowerBoundScaleFactor = this->getMovingAsymptoteLowerBoundScaleFactor();
                    tValue = tLowerBoundScaleFactor * (tUpperBounds[tControlIndex] - tLowerBounds[tControlIndex]);
                    tCurrentSigma[tControlIndex] = std::max(tValue, tCurrentSigma[tControlIndex]);
                    // check that upper bound is satisfied
                    const ScalarType tUpperBoundScaleFactor = this->getMovingAsymptoteUpperBoundScaleFactor();
                    tValue = tUpperBoundScaleFactor * (tUpperBounds[tControlIndex] - tLowerBounds[tControlIndex]);
                    tCurrentSigma[tControlIndex] = std::min(tValue, tCurrentSigma[tControlIndex]);
                }
            }
            mDataMng->setCurrentSigma(mControlWork.operator*());
        }
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mOptimalityTolerance;
    ScalarType mStagnationTolerance;
    ScalarType mFeasibilityTolerance;
    ScalarType mStationarityTolerance;
    ScalarType mObjectiveStagnationTolerance;
    ScalarType mMovingAsymptoteExpansionFactor;
    ScalarType mMovingAsymptoteContractionFactor;
    ScalarType mKarushKuhnTuckerConditionsTolerance;
    ScalarType mMovingAsymptoteUpperBoundScaleFactor;
    ScalarType mMovingAsymptoteLowerBoundScaleFactor;

    locus::ccsa::stop_t mStoppingCriterion;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousSigma;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mAntepenultimateControl;
    std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> mWorkMultiVectorList;

    std::shared_ptr<locus::PrimalProblemStageMng<ScalarType, OrdinalType>> mPrimalStageMng;
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType>> mDataMng;
    std::shared_ptr<locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>> mSubProblem;

private:
    ConservativeConvexSeparableApproximationsAlgorithm(const locus::ConservativeConvexSeparableApproximationsAlgorithm<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableApproximationsAlgorithm<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableApproximationsAlgorithm<ScalarType, OrdinalType> & aRhs);
};

}

/**********************************************************************************************************/
/*********************************************** UNIT TESTS ***********************************************/
/**********************************************************************************************************/

namespace LocusTest
{

/* ******************************************************************* */
/* ************** METHOD OF MOVING ASYMPTOTES UNIT TESTS ************* */
/* ******************************************************************* */

TEST(LocusTest, ConservativeConvexSeparableAppxDataMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    locus::ConservativeConvexSeparableAppxDataMng<double> tDataMng(tDataFactory);

    // ********* TEST INTEGER AND SCALAR PARAMETERS *********
    size_t tOrdinalValue = 1;
    EXPECT_EQ(tOrdinalValue, tDataMng.getNumDualVectors());
    EXPECT_EQ(tOrdinalValue, tDataMng.getNumConstraints());
    EXPECT_EQ(tOrdinalValue, tDataMng.getNumControlVectors());

    double tScalarValue = 0.5;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tDataMng.getDualProblemBoundsScaleFactor(), tTolerance);
    tDataMng.setDualProblemBoundsScaleFactor(0.25);
    tScalarValue = 0.25;
    EXPECT_NEAR(tScalarValue, tDataMng.getDualProblemBoundsScaleFactor(), tTolerance);

    // ********* TEST CONSTRAINT GLOBALIZATION FACTORS *********
    tScalarValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tDualMultiVector(tNumVectors, tNumDuals, tScalarValue);
    LocusTest::checkMultiVectorData(tDualMultiVector, tDataMng.getConstraintGlobalizationFactors());
    tScalarValue = 2;
    locus::fill(tScalarValue, tDualMultiVector);
    tDataMng.setConstraintGlobalizationFactors(tDualMultiVector);
    tScalarValue = 0.5;
    locus::StandardVector<double> tDualVector(tNumDuals, tScalarValue);
    const size_t tVectorIndex = 0;
    tDataMng.setConstraintGlobalizationFactors(tVectorIndex, tDualVector);
    LocusTest::checkVectorData(tDualVector, tDataMng.getConstraintGlobalizationFactors(tVectorIndex));

    // ********* TEST OBJECTIVE FUNCTION *********
    tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarValue = 0.2;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tScalarValue = 0.25;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // ********* TEST INITIAL GUESS *********
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    tScalarValue = 0;
    locus::StandardMultiVector<double> tControlMultiVector(tNumVectors, tNumControls, tScalarValue);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 2;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setInitialGuess(tControlMultiVector);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 5;
    tDataMng.setInitialGuess(tScalarValue);
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 3;
    locus::StandardVector<double> tControlVector(tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentControl(tVectorIndex));

    tScalarValue = 33;
    tDataMng.setInitialGuess(tVectorIndex, tScalarValue);
    tControlVector.fill(tScalarValue);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentControl(tVectorIndex));

    // ********* TEST SET DUAL FUNCTIONS *********
    tScalarValue = 11;
    locus::fill(tScalarValue, tDualMultiVector);
    tDataMng.setDual(tDualMultiVector);
    LocusTest::checkMultiVectorData(tDualMultiVector, tDataMng.getDual());

    tScalarValue = 21;
    tDualVector.fill(tScalarValue);
    tDataMng.setDual(tVectorIndex, tDualVector);
    LocusTest::checkVectorData(tDualVector, tDataMng.getDual(tVectorIndex));

    // ********* TEST TRIAL STEP FUNCTIONS *********
    tScalarValue = 12;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setTrialStep(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getTrialStep());

    tScalarValue = 22;
    tControlVector.fill(tScalarValue);
    tDataMng.setTrialStep(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getTrialStep(tVectorIndex));

    // ********* TEST ACTIVE SET FUNCTIONS *********
    tScalarValue = 10;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setActiveSet(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getActiveSet());

    tScalarValue = 20;
    tControlVector.fill(tScalarValue);
    tDataMng.setActiveSet(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getActiveSet(tVectorIndex));

    // ********* TEST INACTIVE SET FUNCTIONS *********
    tScalarValue = 11;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setInactiveSet(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getInactiveSet());

    tScalarValue = 21;
    tControlVector.fill(tScalarValue);
    tDataMng.setInactiveSet(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getInactiveSet(tVectorIndex));

    // ********* TEST CURRENT CONTROL FUNCTIONS *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentControl(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 2;
    tControlVector.fill(tScalarValue);
    tDataMng.setCurrentControl(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentControl(tVectorIndex));

    // ********* TEST PREVIOUS CONTROL FUNCTIONS *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setPreviousControl(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getPreviousControl());

    tScalarValue = 3;
    tControlVector.fill(tScalarValue);
    tDataMng.setPreviousControl(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getPreviousControl(tVectorIndex));
}

TEST(LocusTest, DualProblemStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    std::shared_ptr<locus::MethodMovingAsymptotes<double>> tSubProblem =
            std::make_shared<locus::MethodMovingAsymptotes<double>>(tDataFactory);
    std::shared_ptr<locus::PrimalProblemStageMng<double>> tPrimalProblem =
            std::make_shared<locus::PrimalProblemStageMng<double>>(tDataFactory);
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<double>> tDataMng =
            std::make_shared<locus::ConservativeConvexSeparableAppxDataMng<double>>(tDataFactory);

    locus::ConservativeConvexSeparableApproximationsAlgorithm<double> tAlgorithm(tPrimalProblem, tDataMng, tSubProblem);
}

}
