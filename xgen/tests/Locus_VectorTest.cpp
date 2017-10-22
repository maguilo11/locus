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
#include "Locus_StandardVectorReductionOperations.hpp"

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
/******************************** NONLINEAR CONJUGATE GRADIENT ALGORITHM **********************************/
/**********************************************************************************************************/

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

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStageMngBase
{
public:
    virtual ~NonlinearConjugateGradientStageMngBase()
    {
    }

    virtual void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                         ScalarType aTolerance = std::numeric_limits<ScalarType>::max()) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

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

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStep
{
public:
    virtual ~NonlinearConjugateGradientStep()
    {
    }

    virtual void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                               locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class PolakRibiere : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    explicit PolakRibiere(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~PolakRibiere()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousGradient());
        tBeta = tBeta / locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    PolakRibiere(const locus::PolakRibiere<ScalarType, OrdinalType> & aRhs);
    locus::PolakRibiere<ScalarType, OrdinalType> & operator=(const locus::PolakRibiere<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class FletcherReeves : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    FletcherReeves(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~FletcherReeves()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                / locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    FletcherReeves(const locus::FletcherReeves<ScalarType, OrdinalType> & aRhs);
    locus::FletcherReeves<ScalarType, OrdinalType> & operator=(const locus::FletcherReeves<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class HestenesStiefel : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    HestenesStiefel(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~HestenesStiefel()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousGradient());
        ScalarType tDenominator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getTrialStep())
                - locus::dot(aDataMng.getPreviousGradient(), aDataMng.getTrialStep());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    HestenesStiefel(const locus::HestenesStiefel<ScalarType, OrdinalType> & aRhs);
    locus::HestenesStiefel<ScalarType, OrdinalType> & operator=(const locus::HestenesStiefel<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConjugateDescent : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    ConjugateDescent(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~ConjugateDescent()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = static_cast<ScalarType>(-1)
                * (locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                        / locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient()));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    ConjugateDescent(const locus::ConjugateDescent<ScalarType, OrdinalType> & aRhs);
    locus::ConjugateDescent<ScalarType, OrdinalType> & operator=(const locus::ConjugateDescent<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DaiYuan : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiYuan(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiYuan()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tDenominator = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tBeta = tNumerator / tDenominator;
        //tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiYuan(const locus::DaiYuan<ScalarType, OrdinalType> & aRhs);
    locus::DaiYuan<ScalarType, OrdinalType> & operator=(const locus::DaiYuan<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class LiuStorey : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    LiuStorey(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~LiuStorey()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousGradient());
        ScalarType tDenominator = static_cast<ScalarType>(-1)
                * locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    LiuStorey(const locus::LiuStorey<ScalarType, OrdinalType> & aRhs);
    locus::LiuStorey<ScalarType, OrdinalType> & operator=(const locus::LiuStorey<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Daniels : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    Daniels(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mHessianTimesVector(aDataFactory.control().create()),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~Daniels()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        aStageMng.applyVectorToHessian(aDataMng.getCurrentControl(),
                                       aDataMng.getTrialStep(),
                                       mHessianTimesVector.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), mHessianTimesVector.operator*());
        ScalarType tDenominator = locus::dot(aDataMng.getTrialStep(), mHessianTimesVector.operator*());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mHessianTimesVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    Daniels(const locus::Daniels<ScalarType, OrdinalType> & aRhs);
    locus::Daniels<ScalarType, OrdinalType> & operator=(const locus::Daniels<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DaiLiao : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiLiao(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaleFactor(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiLiao()
    {
    }

    void setScaleFactor(const ScalarType & aInput)
    {
        mScaleFactor = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentControlDotCurrentGradient = locus::dot(aDataMng.getCurrentControl(), aDataMng.getCurrentGradient());
        ScalarType tPreviousControlDotCurrentGradient = locus::dot(aDataMng.getPreviousControl(), aDataMng.getCurrentGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType tDeltaControlDotCurrentGradient = tCurrentControlDotCurrentGradient
                - tPreviousControlDotCurrentGradient;

        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (mScaleFactor * tDeltaControlDotCurrentGradient));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mScaleFactor;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiLiao(const locus::DaiLiao<ScalarType, OrdinalType> & aRhs);
    locus::DaiLiao<ScalarType, OrdinalType> & operator=(const locus::DaiLiao<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DaiYuanHybrid : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiYuanHybrid(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mWolfeConstant(static_cast<ScalarType>(1) / static_cast<ScalarType>(3)),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiYuanHybrid()
    {
    }

    void setWolfeConstant(const ScalarType & aInput)
    {
        mWolfeConstant = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());

        ScalarType tHestenesStiefelBeta = (tCurrentGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);

        ScalarType tDaiYuanBeta = tCurrentGradientDotCurrentGradient
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tScaleFactor = (static_cast<ScalarType>(1) - mWolfeConstant)
                / (static_cast<ScalarType>(1) + mWolfeConstant);
        tScaleFactor = static_cast<ScalarType>(-1) * tScaleFactor;
        ScalarType tScaledDaiYuanBeta = tScaleFactor * tDaiYuanBeta;

        ScalarType tBeta = std::max(tScaledDaiYuanBeta, std::min(tDaiYuanBeta, tHestenesStiefelBeta));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mWolfeConstant;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiYuanHybrid(const locus::DaiYuanHybrid<ScalarType, OrdinalType> & aRhs);
    locus::DaiYuanHybrid<ScalarType, OrdinalType> & operator=(const locus::DaiYuanHybrid<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class HagerZhang : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    HagerZhang(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mLowerBound(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~HagerZhang()
    {
    }

    void setLowerBound(const ScalarType & aInput)
    {
        mLowerBound = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType DeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;
        ScalarType tScaleFactor = static_cast<ScalarType>(2) * DeltaGradientDotDeltaGradient
                * tOneOverTrialStepDotDeltaGradient;

        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (tScaleFactor * tTrialStepDotCurrentGradient));
        ScalarType tTrialStepDotTrialStep = locus::dot(aDataMng.getTrialStep(), aDataMng.getTrialStep());
        ScalarType tNormTrialStep = std::sqrt(tTrialStepDotTrialStep);
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradientDotPreviousGradient);
        ScalarType tLowerBound = static_cast<ScalarType>(-1)
                / (tNormTrialStep * std::min(tNormPreviousGradient, mLowerBound));
        tBeta = std::max(tBeta, tLowerBound);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mLowerBound;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    HagerZhang(const locus::HagerZhang<ScalarType, OrdinalType> & aRhs);
    locus::HagerZhang<ScalarType, OrdinalType> & operator=(const locus::HagerZhang<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class PerryShanno : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    PerryShanno(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mLowerBound(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~PerryShanno()
    {
    }

    void setLowerBound(const ScalarType & aInput)
    {
        mLowerBound = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = this->computeBeta(aDataMng);
        ScalarType tAlpha = this->computeAlpha(aDataMng);
        ScalarType tTheta = this->computeTheta(aDataMng);

        locus::scale(tBeta, mScaledDescentDirection.operator*());
        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::update(tAlpha,
                      aDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::update(-tAlpha,
                      aDataMng.getPreviousGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::scale(tTheta, mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType computeBeta(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType tDeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;

        ScalarType tScaleFactor = static_cast<ScalarType>(2) * tDeltaGradientDotDeltaGradient
                * tOneOverTrialStepDotDeltaGradient;
        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (tScaleFactor * tTrialStepDotCurrentGradient));
        ScalarType tNormTrialStep = locus::dot(aDataMng.getTrialStep(), aDataMng.getTrialStep());
        tNormTrialStep = std::sqrt(tNormTrialStep);
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradientDotPreviousGradient);
        ScalarType tLowerBound = static_cast<ScalarType>(-1)
                / (tNormTrialStep * std::min(tNormPreviousGradient, mLowerBound));

        tBeta = std::max(tBeta, tLowerBound);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());
        return (tBeta);
    }
    ScalarType computeAlpha(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tAlpha = tTrialStepDotCurrentGradient / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        tAlpha = std::max(tAlpha, std::numeric_limits<ScalarType>::min());
        return (tAlpha);
    }
    ScalarType computeTheta(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tCurrentGradientDotCurrentControl = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentControl());
        ScalarType tPreviousGradientDotCurrentControl = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentControl());
        ScalarType tCurrentGradientDotPreviousControl = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousControl());
        ScalarType tPreviousGradientDotPreviousControl = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousControl());

        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tDeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;
        ScalarType tDeltaGradientDotDeltaControl = tCurrentGradientDotCurrentControl
                - tPreviousGradientDotCurrentControl - tCurrentGradientDotPreviousControl
                + tPreviousGradientDotPreviousControl;

        ScalarType tTheta = tDeltaGradientDotDeltaControl / tDeltaGradientDotDeltaGradient;
        tTheta = std::max(tTheta, std::numeric_limits<ScalarType>::min());
        return (tTheta);
    }

private:
    ScalarType mLowerBound;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    PerryShanno(const locus::PerryShanno<ScalarType, OrdinalType> & aRhs);
    locus::PerryShanno<ScalarType, OrdinalType> & operator=(const locus::PerryShanno<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class StateManager
{
public:
    virtual ~StateManager()
    {
    }

    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;

    virtual ScalarType getCurrentObjectiveValue() const = 0;
    virtual void setCurrentObjectiveValue(const ScalarType & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const = 0;
    virtual void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const = 0;
    virtual void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const = 0;
    virtual void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const = 0;
    virtual void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const = 0;
    virtual void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
};

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

template<typename ScalarType, typename OrdinalType = size_t>
class LineSearch
{
public:
    virtual ~LineSearch()
    {
    }

    virtual OrdinalType getNumIterationsDone() const = 0;
    virtual void setMaxNumIterations(const OrdinalType & aInput) = 0;
    virtual void setContractionFactor(const ScalarType & aInput) = 0;
    virtual void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class CubicLineSearch : public locus::LineSearch<ScalarType, OrdinalType>
{
public:
    explicit CubicLineSearch(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mContractionFactor(0.5),
            mArmijoRuleConstant(1e-4),
            mStagnationTolerance(1e-8),
            mInitialGradientDotTrialStep(0),
            mStepValues(3, static_cast<ScalarType>(0)),
            mTrialControl(aDataFactory.control().create()),
            mProjectedTrialStep(aDataFactory.control().create())
    {
    }
    virtual ~CubicLineSearch()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    ScalarType getStepValue() const
    {
        return (mStepValues[2]);
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStagnationTolerance(const OrdinalType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    void setContractionFactor(const ScalarType & aInput)
    {
        mContractionFactor = aInput;
    }

    void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng)
    {
        ScalarType tInitialStep = 1;
        locus::update(static_cast<ScalarType>(1),
                      aStateMng.getCurrentControl(),
                      static_cast<ScalarType>(0),
                      mTrialControl.operator*());
        locus::update(tInitialStep, aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

        ScalarType tTrialObjectiveValue = aStateMng.evaluateObjective(mTrialControl.operator*());
        // tTrialObjectiveValue[0] = current, tTrialObjectiveValue[1] = old, tTrialObjectiveValue[2] = trial
        const OrdinalType tSize = 3;
        std::vector<ScalarType> tObjectiveFunctionValues(tSize, 0.);
        tObjectiveFunctionValues[0] = aStateMng.getCurrentObjectiveValue();
        tObjectiveFunctionValues[2] = tTrialObjectiveValue;
        // step[0] = old, step[1] = current, step[2] = new
        mStepValues[2] = tInitialStep;
        mStepValues[1] = mStepValues[2];

        mNumIterationsDone = 1;
        mInitialGradientDotTrialStep = locus::dot(aStateMng.getCurrentGradient(), aStateMng.getTrialStep());
        while(mNumIterationsDone <= mMaxNumIterations)
        {
            ScalarType tSufficientDecreaseCondition = tObjectiveFunctionValues[0]
                    + (mArmijoRuleConstant * mStepValues[1] * mInitialGradientDotTrialStep);
            bool tSufficientDecreaseConditionSatisfied =
                    tObjectiveFunctionValues[2] < tSufficientDecreaseCondition ? true : false;
            bool tStepIsLessThanTolerance = mStepValues[2] < mStagnationTolerance ? true : false;
            if(tSufficientDecreaseConditionSatisfied || tStepIsLessThanTolerance)
            {
                break;
            }
            mStepValues[0] = mStepValues[1];
            mStepValues[1] = mStepValues[2];
            if(mNumIterationsDone == static_cast<OrdinalType>(1))
            {
                // first backtrack: do a quadratic fit
                ScalarType tDenominator = static_cast<ScalarType>(2)
                        * (tObjectiveFunctionValues[2] - tObjectiveFunctionValues[0] - mInitialGradientDotTrialStep);
                mStepValues[2] = -mInitialGradientDotTrialStep / tDenominator;
            }
            else
            {
                this->interpolate(tObjectiveFunctionValues);
            }
            this->checkCurrentStepValue();
            locus::update(static_cast<ScalarType>(1),
                          aStateMng.getCurrentControl(),
                          static_cast<ScalarType>(0),
                          mTrialControl.operator*());
            locus::update(mStepValues[2], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

            tTrialObjectiveValue = aStateMng.evaluateObjective(mTrialControl.operator*());
            tObjectiveFunctionValues[1] = tObjectiveFunctionValues[2];
            tObjectiveFunctionValues[2] = tTrialObjectiveValue;
            mNumIterationsDone++;
        }

        aStateMng.setCurrentObjectiveValue(tTrialObjectiveValue);
        aStateMng.setCurrentControl(mTrialControl.operator*());
    }

private:
    void checkCurrentStepValue()
    {
        const ScalarType tGamma = 0.1;
        if(mStepValues[2] > mContractionFactor * mStepValues[1])
        {
            mStepValues[2] = mContractionFactor * mStepValues[1];
        }
        else if(mStepValues[2] < tGamma * mStepValues[1])
        {
            mStepValues[2] = tGamma * mStepValues[1];
        }
        if(std::isfinite(mStepValues[2]) == false)
        {
            mStepValues[2] = tGamma * mStepValues[1];
        }
    }
    void interpolate(const std::vector<ScalarType> & aObjectiveFunctionValues)
    {
        ScalarType tPointOne = aObjectiveFunctionValues[2] - aObjectiveFunctionValues[0]
                - mStepValues[1] * mInitialGradientDotTrialStep;
        ScalarType tPointTwo = aObjectiveFunctionValues[1] - aObjectiveFunctionValues[0]
                - mStepValues[0] * mInitialGradientDotTrialStep;
        ScalarType tPointThree = static_cast<ScalarType>(1.) / (mStepValues[1] - mStepValues[0]);

        // find cubic unique minimum
        ScalarType tPointA = tPointThree
                * ((tPointOne / (mStepValues[1] * mStepValues[1])) - (tPointTwo / (mStepValues[0] * mStepValues[0])));
        ScalarType tPointB = tPointThree
                * ((tPointTwo * mStepValues[1] / (mStepValues[0] * mStepValues[0]))
                        - (tPointOne * mStepValues[0] / (mStepValues[1] * mStepValues[1])));
        ScalarType tPointC = tPointB * tPointB - static_cast<ScalarType>(3) * tPointA * mInitialGradientDotTrialStep;

        // cubic equation has unique minimum
        ScalarType tValueOne = (-tPointB + std::sqrt(tPointC)) / (static_cast<ScalarType>(3) * tPointA);
        // cubic equation is a quadratic
        ScalarType tValueTwo = -mInitialGradientDotTrialStep / (static_cast<ScalarType>(2) * tPointB);
        mStepValues[2] = tPointA != 0 ? mStepValues[2] = tValueOne : mStepValues[2] = tValueTwo;
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mContractionFactor;
    ScalarType mArmijoRuleConstant;
    ScalarType mStagnationTolerance;
    ScalarType mInitialGradientDotTrialStep;

    std::vector<ScalarType> mStepValues;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mProjectedTrialStep;

private:
    CubicLineSearch(const locus::CubicLineSearch<ScalarType, OrdinalType> & aRhs);
    locus::CubicLineSearch<ScalarType, OrdinalType> & operator=(const locus::CubicLineSearch<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class QuadraticLineSearch : public locus::LineSearch<ScalarType, OrdinalType>
{
public:
    explicit QuadraticLineSearch(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mStepLowerBound(1e-3),
            mStepUpperBound(0.5),
            mContractionFactor(0.5),
            mInitialTrialStepDotCurrentGradient(0),
            mStepValues(2, static_cast<ScalarType>(0)),
            mTrialControl(aDataFactory.control().create())
    {
    }
    virtual ~QuadraticLineSearch()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    ScalarType getStepValue() const
    {
        return (mStepValues[1]);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStepLowerBound(const ScalarType & aInput)
    {
        mStepLowerBound = aInput;
    }
    void setStepUpperBound(const ScalarType & aInput)
    {
        mStepUpperBound = aInput;
    }
    void setContractionFactor(const ScalarType & aInput)
    {
        mContractionFactor = aInput;
    }

    void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng)
    {
        OrdinalType tSize = 3;
        // tObjectiveFunction[0] = current trial value
        // tObjectiveFunction[1] = old trial value
        // tObjectiveFunction[2] = new trial value
        std::vector<ScalarType> tObjectiveFunction(tSize);
        tObjectiveFunction[0] = aStateMng.getCurrentObjectiveValue();

        ScalarType tNormTrialStep = locus::dot(aStateMng.getTrialStep(), aStateMng.getTrialStep());
        tNormTrialStep = std::sqrt(tNormTrialStep);
        mStepValues[1] = std::min(static_cast<ScalarType>(1),
                                  static_cast<ScalarType>(100) / (static_cast<ScalarType>(1) + tNormTrialStep));

        locus::update(static_cast<ScalarType>(1),
                      aStateMng.getCurrentControl(),
                      static_cast<ScalarType>(0),
                      *mTrialControl);
        locus::update(mStepValues[1], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

        tObjectiveFunction[2] = aStateMng.evaluateObjective(*mTrialControl);
        mInitialTrialStepDotCurrentGradient = locus::dot(aStateMng.getTrialStep(), aStateMng.getCurrentGradient());
        const ScalarType tAlpha = 1e-4;
        ScalarType tTargetObjectiveValue = tObjectiveFunction[0]
                - (tAlpha * mStepValues[1] * mInitialTrialStepDotCurrentGradient);

        mNumIterationsDone = 1;
        while(tObjectiveFunction[2] > tTargetObjectiveValue)
        {
            mStepValues[0] = mStepValues[1];
            ScalarType tStep = this->interpolate(tObjectiveFunction, mStepValues);
            mStepValues[1] = tStep;

            locus::update(static_cast<ScalarType>(1),
                          aStateMng.getCurrentControl(),
                          static_cast<ScalarType>(0),
                          *mTrialControl);
            locus::update(mStepValues[1], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

            tObjectiveFunction[1] = tObjectiveFunction[2];
            tObjectiveFunction[2] = aStateMng.evaluateObjective(*mTrialControl);

            mNumIterationsDone++;
            if(mNumIterationsDone >= mMaxNumIterations)
            {
                break;
            }
            tTargetObjectiveValue = tObjectiveFunction[0]
                    - (tAlpha * mStepValues[1] * mInitialTrialStepDotCurrentGradient);
        }

        aStateMng.setCurrentObjectiveValue(tObjectiveFunction[2]);
        aStateMng.setCurrentControl(*mTrialControl);
    }

private:
    ScalarType interpolate(const std::vector<ScalarType> & aObjectiveFunction, const std::vector<ScalarType> & aStepValues)
    {
        ScalarType tStepLowerBound = aStepValues[1] * mStepLowerBound;
        ScalarType tStepUpperBound = aStepValues[1] * mStepUpperBound;
        ScalarType tDenominator = static_cast<ScalarType>(2) * aStepValues[1]
                * (aObjectiveFunction[2] - aObjectiveFunction[0] - mInitialTrialStepDotCurrentGradient);

        ScalarType tStep = -mInitialTrialStepDotCurrentGradient / tDenominator;
        if(tStep < tStepLowerBound)
        {
            tStep = tStepLowerBound;
        }
        if(tStep > tStepUpperBound)
        {
            tStep = tStepUpperBound;
        }

        return (tStep);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mStepLowerBound;
    ScalarType mStepUpperBound;
    ScalarType mContractionFactor;
    ScalarType mInitialTrialStepDotCurrentGradient;

    // mStepValues[0] = old trial value
    // mStepValues[1] = new trial value
    std::vector<ScalarType> mStepValues;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;

private:
    QuadraticLineSearch(const locus::QuadraticLineSearch<ScalarType, OrdinalType> & aRhs);
    locus::QuadraticLineSearch<ScalarType, OrdinalType> & operator=(const locus::QuadraticLineSearch<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradient
{
public:
    NonlinearConjugateGradient(const std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> & aDataFactory,
                               const std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> & aDataMng,
                               const std::shared_ptr<locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType>> & aStageMng) :
            mMaxNumIterations(500),
            mNumIterationsDone(0),
            mGradientTolerance(1e-8),
            mStationarityTolerance(1e-8),
            mControlStagnationTolerance(std::numeric_limits<ScalarType>::epsilon()),
            mObjectiveStagnationTolerance(std::numeric_limits<ScalarType>::epsilon()),
            mStoppingCriteria(locus::algorithm::stop_t::NOT_CONVERGED),
            mControlWork(aDataFactory->control().create()),
            mTrialControl(aDataFactory->control().create()),
            mLineSearch(std::make_shared<locus::CubicLineSearch<ScalarType, OrdinalType>>(aDataFactory.operator*())),
            mStep(std::make_shared<locus::PolakRibiere<ScalarType, OrdinalType>>(aDataFactory.operator*())),
            mStateMng(std::make_shared<locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType>>(aDataMng, aStageMng))
    {
    }
    ~NonlinearConjugateGradient()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    locus::algorithm::stop_t getStoppingCriteria() const
    {
        return (mStoppingCriteria);
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setGradientTolerance(const ScalarType & aInput)
    {
        mGradientTolerance = aInput;
    }
    void setStationarityTolerance(const ScalarType & aInput)
    {
        mStationarityTolerance = aInput;
    }
    void setControlStagnationTolerance(const ScalarType & aInput)
    {
        mControlStagnationTolerance = aInput;
    }
    void setObjectiveStagnationTolerance(const ScalarType & aInput)
    {
        mObjectiveStagnationTolerance = aInput;
    }

    void setContractionFactor(const ScalarType & aInput)
    {
        mLineSearch->setContractionFactor(aInput);
    }
    void setDanielsMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::Daniels<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiLiaoMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiLiao<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiYuanMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiYuan<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setLiuStoreyMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::LiuStorey<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setHagerZhangMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::HagerZhang<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setPerryShannoMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::PerryShanno<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiYuanHybridMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiYuanHybrid<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setFletcherReevesMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::FletcherReeves<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setHestenesStiefelMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::HestenesStiefel<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setConjugateDescentMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::ConjugateDescent<ScalarType, OrdinalType>>(aDataFactory);
    }

    void solve()
    {
        assert(mStep.get() != nullptr);

        this->computeInitialState();
        // Perform first iteration (i.e. x_0)
        this->computeInitialDescentDirection();
        this->computeProjectedStep();
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        tDataMng.storePreviousState();

        mLineSearch->step(mStateMng.operator*());
        locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
        mStateMng->computeGradient(tDataMng.getCurrentControl(), mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
        this->computeProjectedGradient();

        bool tStop = false;
        if(this->checkStoppingCriteria() == true)
        {
            tStop = true;
        }

        mNumIterationsDone = 1;
        locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & tStageMng = mStateMng->getStageMng();
        while(tStop != true)
        {

            mStep->computeScaledDescentDirection(tDataMng, tStageMng);
            this->computeProjectedStep();
            tDataMng.storePreviousState();

            mLineSearch->step(mStateMng.operator*());
            locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
            mStateMng->computeGradient(tDataMng.getCurrentControl(), mControlWork.operator*());
            tDataMng.setCurrentGradient(mControlWork.operator*());
            this->computeProjectedGradient();

            mNumIterationsDone++;
            if(this->checkStoppingCriteria() == true)
            {
                tStop = true;
                break;
            }
        }
    }

private:
    void computeInitialState()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        ScalarType tValue = mStateMng->evaluateObjective(tControl);
        tDataMng.setCurrentObjectiveFunctionValue(tValue);

        mStateMng->computeGradient(tControl, mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
        this->computeProjectedGradient();
    }
    void computeProjectedStep()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(0), mTrialControl.operator*());
        locus::update(static_cast<ScalarType>(1),
                      tDataMng.getTrialStep(),
                      static_cast<ScalarType>(1),
                      mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = tDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = tDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());

        // Compute projected trial step
        locus::update(static_cast<ScalarType>(1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(-1), tControl, static_cast<ScalarType>(1), mControlWork.operator*());
        tDataMng.setTrialStep(mControlWork.operator*());
    }
    void computeProjectedGradient()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(0), mTrialControl.operator*());
        locus::update(static_cast<ScalarType>(1),
                      tDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = tDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = tDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());

        // Compute projected gradient
        locus::update(static_cast<ScalarType>(1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(-1), tControl, static_cast<ScalarType>(1), mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
    }
    void computeInitialDescentDirection()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        locus::update(static_cast<ScalarType>(-1),
                      tDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        tDataMng.setTrialStep(mControlWork.operator*());
    }
    bool checkStoppingCriteria()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();

        tDataMng.computeNormGradient();
        tDataMng.computeStagnationMeasure();
        tDataMng.computeStationarityMeasure();
        tDataMng.computeObjectiveStagnationMeasure();

        const ScalarType tNormGradient = tDataMng.getNormGradient();
        const ScalarType tStagnationMeasure = tDataMng.getStagnationMeasure();
        const ScalarType tStationarityMeasure = tDataMng.getStationarityMeasure();
        const ScalarType tObjectiveStagnationMeasure = tDataMng.getObjectiveStagnationMeasure();

        bool tStop = false;
        if(tStagnationMeasure < mControlStagnationTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::CONTROL_STAGNATION;
        }
        else if(tNormGradient < mGradientTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::NORM_GRADIENT;
        }
        else if(tStationarityMeasure < mStationarityTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::NORM_STEP;
        }
        else if(mNumIterationsDone >= mMaxNumIterations)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS;
        }
        else if(tObjectiveStagnationMeasure < mObjectiveStagnationTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::OBJECTIVE_STAGNATION;
        }

        return (tStop);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mGradientTolerance;
    ScalarType mStationarityTolerance;
    ScalarType mControlStagnationTolerance;
    ScalarType mObjectiveStagnationTolerance;

    locus::algorithm::stop_t mStoppingCriteria;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;

    std::shared_ptr<locus::LineSearch<ScalarType, OrdinalType>> mLineSearch;
    std::shared_ptr<locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>> mStep;
    std::shared_ptr<locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType>> mStateMng;

private:
    NonlinearConjugateGradient(const locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & aRhs);
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
/* ************* NONLINEAR CONJUGATE GRADIENT UNIT TESTS ************* */
/* ******************************************************************* */

TEST(LocusTest, NonlinearConjugateGradientDataMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);
    size_t tOrdinalValue = 1;
    EXPECT_EQ(tDataMng.getNumControlVectors(), tOrdinalValue);

    // ********* TEST OBJECTIVE FUNCTION VALUE *********
    const double tTolerance = 1e-6;
    double tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tScalarValue = 45;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarValue = 123;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // ********* TEST INITIAL GUESS FUNCTIONS *********
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    tScalarValue = 2;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tMultiVector(tNumVectors, tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tScalarValue);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.5;
    const size_t tVectorIndex = 0;
    tDataMng.setInitialGuess(tVectorIndex, tScalarValue);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setInitialGuess(tMultiVector);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 0.5;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tVector);

    // ********* TEST TRIAL STEP FUNCTIONS *********
    tScalarValue = 0.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setTrialStep(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tMultiVector);

    tScalarValue = 0.25;
    tVector.fill(tScalarValue);
    tDataMng.setTrialStep(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);

    // ********* TEST CURRENT CONTROL FUNCTIONS *********
    tScalarValue = 1.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentControl(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.25;
    tVector.fill(tScalarValue);
    tDataMng.setCurrentControl(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tVector);

    // ********* TEST PREVIOUS CONTROL FUNCTIONS *********
    tScalarValue = 1.21;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setPreviousControl(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tMultiVector);

    tScalarValue = 1.11;
    tVector.fill(tScalarValue);
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tVector);

    // ********* TEST CURRENT GRADIENT FUNCTIONS *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentGradient(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tMultiVector);

    tScalarValue = 3;
    tVector.fill(tScalarValue);
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentGradient(tVectorIndex), tVector);

    // ********* TEST PREVIOUS GRADIENT FUNCTIONS *********
    tScalarValue = 2.1;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setPreviousGradient(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tMultiVector);

    tScalarValue = 3.1;
    tVector.fill(tScalarValue);
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getPreviousGradient(tVectorIndex), tVector);

    // ********* TEST DEFAULT UPPER AND LOWER BOUNDS *********
    tScalarValue = -std::numeric_limits<double>::max();
    tVector.fill(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tVector);

    tScalarValue = std::numeric_limits<double>::max();
    tVector.fill(tScalarValue);
    locus::fill(tScalarValue,tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tVector);

    // ********* TEST LOWER BOUND FUNCTIONS *********
    tScalarValue = -10;
    tDataMng.setControlLowerBounds(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -9;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setControlLowerBounds(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -8;
    tVector.fill(tScalarValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tVector);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -7;
    tDataMng.setControlLowerBounds(tVectorIndex, tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    // ********* TEST UPPER BOUND FUNCTIONS *********
    tScalarValue = 10;
    tDataMng.setControlUpperBounds(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 9;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setControlUpperBounds(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 8;
    tVector.fill(tScalarValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tVector);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 7;
    tDataMng.setControlUpperBounds(tVectorIndex, tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    // ********* TEST COMPUTE CONTROL STAGNATION MEASURE *********
    tScalarValue = 3;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentControl(tMultiVector);
    tVector[0] = 2;
    tVector[1] = 2.5;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tDataMng.computeStagnationMeasure();
    tScalarValue = 1.;
    EXPECT_NEAR(tDataMng.getStagnationMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE OBJECTIVE STAGNATION MEASURE *********
    tScalarValue = 1.25;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    tScalarValue = 0.75;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    tDataMng.computeObjectiveStagnationMeasure();
    tScalarValue = 0.5;
    EXPECT_NEAR(tDataMng.getObjectiveStagnationMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE NORM GRADIENT *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentGradient(tMultiVector);
    tDataMng.computeNormGradient();
    tScalarValue = std::sqrt(2.);
    EXPECT_NEAR(tDataMng.getNormGradient(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE STATIONARITY MEASURE *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setTrialStep(tMultiVector);
    tDataMng.computeStationarityMeasure();
    tScalarValue = std::sqrt(8.);
    EXPECT_NEAR(tDataMng.getStationarityMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE STATIONARITY MEASURE *********
    tDataMng.storePreviousState();
    tScalarValue = 1.25;
    EXPECT_NEAR(tDataMng.getPreviousObjectiveFunctionValue(), tScalarValue, tTolerance);
    tScalarValue = 1;
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tMultiVector);
    tScalarValue = 3;
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tMultiVector);
}

TEST(LocusTest, NonlinearConjugateGradientStandardStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Test Evaluate Objective Function *********
    double tScalarValue = 2;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tScalarValue = 401;
    double tTolerance = 1e-6;
    size_t tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveFunctionEvaluations(), tOrdinalValue);
    EXPECT_NEAR(tStageMng.evaluateObjective(tControl), tScalarValue, tTolerance);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveFunctionEvaluations(), tOrdinalValue);

    // ********* Test Compute Gradient *********
    tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveGradientEvaluations(), tOrdinalValue);
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tStageMng.computeGradient(tControl, tGradient);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveGradientEvaluations(), tOrdinalValue);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 0) = 1602;
    tGoldVector(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // ********* Test Apply Vector to Hessian *********
    tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveHessianEvaluations(), tOrdinalValue);
    tScalarValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tScalarValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tStageMng.applyVectorToHessian(tControl, tVector, tHessianTimesVector);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveHessianEvaluations(), tOrdinalValue);
    tGoldVector(tVectorIndex, 0) = 3202;
    tGoldVector(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, PolakRibiere)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Polak-Ribiere Direction *********
    locus::PolakRibiere<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -103.4;
    tVector[1] = -250.8;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, FletcherReeves)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Fletcher-Reeves Direction *********
    locus::FletcherReeves<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -110;
    tVector[1] = -264;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, HestenesStiefel)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -3;
    tVector[1] = -2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hestenes-Stiefel Direction *********
    locus::HestenesStiefel<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 1.3333333333333333;
    tVector[1] = -1.333333333333333;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, ConjugateDescent)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Conjugate Descent Direction *********
    locus::ConjugateDescent<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -110;
    tVector[1] = -264;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiLiao)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -1;
    tVector[1] = 2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tVector[0] = 2;
    tVector[1] = 3;
    tDataMng.setCurrentControl(tVectorIndex, tVector);

    // ********* Allocate Dai-Liao Direction *********
    locus::DaiLiao<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 0.05;
    tVector[1] = -3.9;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, PerryShanno)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tVector[0] = 2;
    tVector[1] = 3;
    tDataMng.setCurrentControl(tVectorIndex, tVector);

    // ********* Allocate Perry-Shanno Direction *********
    locus::PerryShanno<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -0.419267707083;
    tVector[1] = -0.722989195678;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, LiuStorey)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Liu-Storey Direction *********
    locus::LiuStorey<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -103.4;
    tVector[1] = -250.8;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, HagerZhang)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::HagerZhang<double> tDirection(tDataFactory);
    // TEST 1: SCALE FACTOR SELECTED
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -14.367346938775;
    tVector[1] = -72.734693877551;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
    // TEST 2: SCALE FACTOR NOT SELECTED, LOWER BOUND USED INSTEAD
    tVector.fill(1e-1);
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 11;
    tVector[1] = -22;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiYuan)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -1;
    tVector[1] = 2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::DaiYuan<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -1.5;
    tVector[1] = -7;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiYuanHybrid)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::DaiYuanHybrid<double> tDirection(tDataFactory);
    // TEST 1: SCALED STEP
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 0.19642857142857;
    tVector[1] = -43.607142857142;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
    // TEST 2: UNSCALED STEP
    tVector[0] = -12;
    tVector[1] = -23;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 11;
    tVector[1] = 22;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 12.067522825323;
    tVector[1] = 8.009932778168;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, NonlinearConjugateGradientStateMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(tDataFactory);

    // ********* Allocate Nonlinear Conjugate Gradient State Manager *********
    locus::NonlinearConjugateGradientStateMng<double> tStateMng(tDataMng, tStageMng);

    // ********* Test Set Trial Step Function *********
    double tScalarValue = 0.1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tMultiVector(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setTrialStep(tMultiVector);
    LocusTest::checkMultiVectorData(tStateMng.getTrialStep(), tMultiVector);

    // ********* Test Set Current Control Function *********
    tScalarValue = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setCurrentControl(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getCurrentControl(), tControl);

    // ********* Test Set Current Control Function *********
    tScalarValue = 3;
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setCurrentGradient(tGradient);
    LocusTest::checkMultiVectorData(tStateMng.getCurrentGradient(), tGradient);

    // ********* Test Set Control Lower Bounds Function *********
    tScalarValue = std::numeric_limits<double>::min();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlLowerBounds(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getControlLowerBounds(), tControl);

    // ********* Test Set Control Upper Bounds Function *********
    tScalarValue = std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlUpperBounds(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getControlUpperBounds(), tControl);

    // ********* Test Evaluate Objective Function *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tControl);
    tScalarValue = 401;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tStateMng.evaluateObjective(tControl), tScalarValue, tTolerance);

    // ********* Test Set Current Objective Function *********
    tStateMng.setCurrentObjectiveValue(tScalarValue);
    EXPECT_NEAR(tStateMng.getCurrentObjectiveValue(), tScalarValue, tTolerance);

    // ********* Test Compute Gradient Function *********
    tScalarValue = 0;
    locus::fill(tScalarValue, tGradient);
    tStateMng.computeGradient(tControl, tGradient);
    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGold(tVectorIndex, 0) = 1602;
    tGold(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGold);

    // ********* Test Apply Vector to Hessian Function *********
    tScalarValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tScalarValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tStateMng.applyVectorToHessian(tControl, tVector, tHessianTimesVector);
    tGold(tVectorIndex, 0) = 3202;
    tGold(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGold);
}

TEST(LocusTest, QuadraticLineSearch)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(tDataFactory);

    // ********* Allocate Nonlinear Conjugate Gradient State Manager *********
    locus::NonlinearConjugateGradientStateMng<double> tStateMng(tDataMng, tStageMng);

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    double tScalarValue = std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlUpperBounds(tControl);
    tScalarValue = -std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlLowerBounds(tControl);

    const size_t tVectorIndex = 0;
    tControl(tVectorIndex, 0) = 1.997506234413967;
    tControl(tVectorIndex, 1) = 3.990024937655861;
    tStateMng.setCurrentControl(tControl);
    tScalarValue = tStateMng.evaluateObjective(tControl);
    const double tTolerance = 1e-6;
    double tGoldScalarValue = 0.99501869156216238;
    EXPECT_NEAR(tScalarValue, tGoldScalarValue, tTolerance);
    tStateMng.setCurrentObjectiveValue(tScalarValue);

    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tStateMng.computeGradient(tControl, tGradient);
    tStateMng.setCurrentGradient(tGradient);

    locus::StandardMultiVector<double> tTrialStep(tNumVectors, tNumControls);
    tTrialStep(tVectorIndex, 0) = -1.997506234413967;
    tTrialStep(tVectorIndex, 1) = -3.990024937655861;
    tStateMng.setTrialStep(tTrialStep);

    // ********* Allocate Quadratic Line Search *********
    locus::QuadraticLineSearch<double> tLineSearch(tDataFactory);
    tLineSearch.step(tStateMng);

    size_t tOrdinalValue = 7;
    EXPECT_EQ(tOrdinalValue, tLineSearch.getNumIterationsDone());
    tGoldScalarValue = 0.00243606117022465;
    EXPECT_NEAR(tLineSearch.getStepValue(), tGoldScalarValue, tTolerance);
    tGoldScalarValue = 0.99472430176791571;
    EXPECT_NEAR(tStateMng.getCurrentObjectiveValue(), tGoldScalarValue, tTolerance);
    tControl(tVectorIndex, 0) = 1.9926401870390293;
    tControl(tVectorIndex, 1) = 3.9803049928370093;
    LocusTest::checkMultiVectorData(tStateMng.getCurrentControl(), tControl);
}

TEST(LocusTest, NonlinearConjugateGradient_PolakRibiere_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    size_t tOrdinalValue = 37;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_PolakRibiere_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    size_t tOrdinalValue = 68;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_FletcherReeves_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setFletcherReevesMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 89;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_FletcherReeves_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setFletcherReevesMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 75;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HestenesStiefel_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHestenesStiefelMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 23;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HestenesStiefel_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHestenesStiefelMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 35;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HagerZhang_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHagerZhangMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 56;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HagerZhang_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHagerZhangMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 53;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuanHybrid_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanHybridMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 24;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuanHybrid_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanHybridMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 47;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuan_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 28;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuan_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 1.2; // NOTE: DIFFERENT INITIAL GUESS, DIVERGES IF INITIAL GUESS = 2
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 37;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_PerryShanno_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setPerryShannoMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 33;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_STEP, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_ConjugateDescentMethod_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setConjugateDescentMethod(tDataFactory.operator*());
    tAlgorithm.setContractionFactor(0.25);
    tAlgorithm.solve();

    size_t tOrdinalValue = 313;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_ConjugateDescentMethod_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setConjugateDescentMethod(tDataFactory.operator*());
    tAlgorithm.setContractionFactor(0.25);
    tAlgorithm.solve();

    size_t tOrdinalValue = 126;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_LiuStorey_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setLiuStoreyMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 50;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_LiuStorey_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setLiuStoreyMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 72;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_Daniels_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDanielsMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 28;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

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
