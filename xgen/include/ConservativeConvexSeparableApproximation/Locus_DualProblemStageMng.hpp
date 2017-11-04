/*
 * Locus_DualProblemStageMng.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DUALPROBLEMSTAGEMNG_HPP_
#define LOCUS_DUALPROBLEMSTAGEMNG_HPP_

#include <cmath>
#include <vector>
#include <limits>
#include <memory>
#include <numeric>
#include <cassert>
#include <algorithm>

#include "Locus_Vector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_ReductionOperations.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

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

    ScalarType getObjectiveCoefficientsR() const
    {
        return (mObjectiveCoefficientR);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getLowerAsymptotes() const
    {
        assert(mLowerAsymptotes.get() != nullptr);
        return (mLowerAsymptotes.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getUpperAsymptotes() const
    {
        assert(mUpperAsymptotes.get() != nullptr);
        return (mUpperAsymptotes.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialControlLowerBounds() const
    {
        assert(mTrialControlLowerBounds.get() != nullptr);
        return (mTrialControlLowerBounds.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialControlUpperBounds() const
    {
        assert(mTrialControlUpperBounds.get() != nullptr);
        return (mTrialControlUpperBounds.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getObjectiveCoefficientsP() const
    {
        assert(mObjectiveCoefficientsP.get() != nullptr);
        return (mObjectiveCoefficientsP.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getObjectiveCoefficientsQ() const
    {
        assert(mObjectiveCoefficientsQ.get() != nullptr);
        return (mObjectiveCoefficientsQ.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsP(const OrdinalType & aConstraintIndex) const
    {
        assert(mConstraintCoefficientsP.empty() == false);
        assert(mConstraintCoefficientsP[aConstraintIndex].get() != nullptr);
        return (mConstraintCoefficientsP[aConstraintIndex].operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsQ(const OrdinalType & aConstraintIndex) const
    {
        assert(mConstraintCoefficientsQ.empty() == false);
        assert(mConstraintCoefficientsQ[aConstraintIndex].get() != nullptr);
        return (mConstraintCoefficientsQ[aConstraintIndex].operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsR() const
    {
        assert(mConstraintCoefficientsR.get() != nullptr);
        return (mConstraintCoefficientsR.operator*());
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
            locus::Vector<ScalarType, OrdinalType> & tMyObjectiveCoefficientsP = mObjectiveCoefficientsP->operator[](tVectorIndex);
            locus::Vector<ScalarType, OrdinalType> & tMyObjectiveCoefficientsQ = mObjectiveCoefficientsQ->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentSigma = tCurrentSigma[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentObjectiveGradient = aDataMng.getCurrentObjectiveGradient(tVectorIndex);

            OrdinalType tNumControls = tMyCurrentSigma.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                        * tMyCurrentSigma[tControlIndex];
                tMyObjectiveCoefficientsP[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ScalarType>(0), tMyCurrentObjectiveGradient[tControlIndex])
                        + ((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ScalarType>(4));

                tMyObjectiveCoefficientsQ[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ScalarType>(0), -tMyCurrentObjectiveGradient[tControlIndex])
                        + ((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ScalarType>(4));
                mControlWorkVectorOne->operator[](tControlIndex) = (tMyObjectiveCoefficientsP[tControlIndex]
                        + tMyObjectiveCoefficientsQ[tControlIndex]) / tMyCurrentSigma[tControlIndex];
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
                ScalarType tValueTwo = tMyCoefficientsD[tIndex] * tMyAuxiliaryVariablesY[tIndex] * tMyAuxiliaryVariablesY[tIndex];
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

} // namespace locus

#endif /* LOCUS_DUALPROBLEMSTAGEMNG_HPP_ */
