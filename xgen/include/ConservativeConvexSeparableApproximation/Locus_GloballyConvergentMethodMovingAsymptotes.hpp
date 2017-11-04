/*
 * Locus_GloballyConvergentMethodMovingAsymptotes.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_GLOBALLYCONVERGENTMETHODMOVINGASYMPTOTES_HPP_
#define LOCUS_GLOBALLYCONVERGENTMETHODMOVINGASYMPTOTES_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>

#include "Locus_Bounds.hpp"
#include "Locus_Vector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_ReductionOperations.hpp"
#include "Locus_PrimalProblemStageMng.hpp"
#include "Locus_NonlinearConjugateGradientDualSolver.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"
#include "Locus_ConservativeConvexSeparableApproximation.hpp"

namespace locus
{

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
            mDualWorkOne(),
            mControlWorkOne(),
            mControlWorkTwo(),
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mControlWork(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mDeltaControl(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mPreviousTrialControl(aDataFactory.control().create()),
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
            locus::update(static_cast<ScalarType>(1), *mTrialControl, static_cast<ScalarType>(0), *mPreviousTrialControl);
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
        mDualWorkOne = mTrialDual->operator[](tVectorIndex).create();
        mControlWorkOne = mTrialControl->operator[](tVectorIndex).create();
        mControlWorkTwo = mTrialControl->operator[](tVectorIndex).create();
        locus::fill(static_cast<ScalarType>(1e-5), mMinConstraintGlobalizationFactors.operator*());
    }
    void updateObjectiveGlobalizationFactor(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mControlWorkOne->fill(static_cast<ScalarType>(0));
        mControlWorkTwo->fill(static_cast<ScalarType>(0));

        const OrdinalType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumVectors, 0);
        std::vector<ScalarType> tStorageTwo(tNumVectors, 0);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tCurrentGradient = aDataMng.getCurrentObjectiveGradient(tVectorIndex);

            const OrdinalType tNumControls = mControlWorkOne->size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tNumerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                ScalarType tDenominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                        - (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]);
                mControlWorkOne->operator[](tControlIndex) = tNumerator / tDenominator;

                tNumerator = ((tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                        * tCurrentGradient[tControlIndex] * tDeltaControl[tControlIndex])
                        + (tCurrentSigma[tControlIndex] * std::abs(tCurrentGradient[tControlIndex])
                                * (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]));
                mControlWorkTwo->operator[](tControlIndex) = tNumerator / tDenominator;
            }

            tStorageOne[tVectorIndex] = mControlReductionOperations->sum(mControlWorkOne.operator*());
            tStorageTwo[tVectorIndex] = mControlReductionOperations->sum(mControlWorkTwo.operator*());
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

        bool tUpdateConstraintGlobalizationFactors = false;
        const OrdinalType tDualVectorIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tConstraintValues = aDataMng.getCurrentConstraintValues(tDualVectorIndex);
        const locus::Vector<ScalarType, OrdinalType> & tTrialConstraintValues = mTrialConstraintValues->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType, OrdinalType> & tGlobalizationFactors = aDataMng.getConstraintGlobalizationFactors(tDualVectorIndex);

        const OrdinalType tNumConstraints = aDataMng.getNumConstraints();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            mControlWorkOne->fill(static_cast<ScalarType>(0));
            mControlWorkTwo->fill(static_cast<ScalarType>(0));

            const OrdinalType tNumVectors = mTrialControl->getNumVectors();
            std::vector<ScalarType> tStorageOne(tNumVectors, 0);
            std::vector<ScalarType> tStorageTwo(tNumVectors, 0);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                const locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tConstraintGradient =
                        aDataMng.getCurrentConstraintGradients(tConstraintIndex, tVectorIndex);
                assert(tConstraintGradient.size() == tDeltaControl.size());

                const OrdinalType tNumControls = mControlWorkOne->size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ScalarType numerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                    ScalarType denominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                            - (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]);
                    mControlWorkOne->operator[](tControlIndex) = numerator / denominator;

                    numerator = ((tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                            * tConstraintGradient[tControlIndex] * tDeltaControl[tControlIndex])
                            + (tCurrentSigma[tControlIndex] * std::abs(tConstraintGradient[tControlIndex])
                                    * (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]));
                    mControlWorkTwo->operator[](tControlIndex) = numerator / denominator;
                }

                tStorageOne[tVectorIndex] = mControlReductionOperations->sum(mControlWorkOne.operator*());
                tStorageTwo[tVectorIndex] = mControlReductionOperations->sum(mControlWorkTwo.operator*());
            }

            const ScalarType tInitialValue = 0;
            ScalarType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
            tFunctionEvaluationW = static_cast<ScalarType>(0.5) * tFunctionEvaluationW;

            ScalarType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
            tFunctionEvaluationV = tFunctionEvaluationV + tConstraintValues[tConstraintIndex];

            ScalarType tCcsaFunctionValue =
                    tFunctionEvaluationV + (tGlobalizationFactors[tConstraintIndex] * tFunctionEvaluationW);
            ScalarType tActualOverPredictedReduction =
                    (tTrialConstraintValues[tConstraintIndex] - tCcsaFunctionValue) / tFunctionEvaluationW;

            if(tActualOverPredictedReduction > static_cast<ScalarType>(0))
            {
                tUpdateConstraintGlobalizationFactors = true;
                ScalarType tValueOne = static_cast<ScalarType>(10) * tGlobalizationFactors[tConstraintIndex];
                ScalarType tValueTwo = static_cast<ScalarType>(1.1)
                        * (tGlobalizationFactors[tConstraintIndex] + tActualOverPredictedReduction);
                mDualWorkOne->operator[](tConstraintIndex) = std::min(tValueOne, tValueTwo);
            }
        }
        if(tUpdateConstraintGlobalizationFactors == true)
        {
            aDataMng.setConstraintGlobalizationFactors(tDualVectorIndex, mDualWorkOne.operator*());
        }
    }
    bool checkStoppingCriteria(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        aDataMng.computeKarushKuhnTuckerConditionsInexactness(mTrialControl.operator*(), mTrialDual.operator*());
        const ScalarType t_KKT_ConditionsInexactness = aDataMng.getKarushKuhnTuckerConditionsInexactness();

        const ScalarType tObjectiveStagnation =
                std::abs(mCurrentTrialObjectiveFunctionValue - mPreviousTrialObjectiveFunctionValue);

        locus::update(static_cast<ScalarType>(1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(-1),
                      mPreviousTrialControl.operator*(),
                      static_cast<ScalarType>(1),
                      mControlWork.operator*());
        ScalarType tControlStagnation = locus::dot(mControlWork.operator*(), mControlWork.operator*());
        tControlStagnation = std::sqrt(tControlStagnation);

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
        else if(tControlStagnation < this->getStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::CONTROL_STAGNATION);
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

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDeltaControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialConstraintValues;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mMinConstraintGlobalizationFactors;

    std::shared_ptr<locus::DualProblemSolver<ScalarType, OrdinalType>> mDualSolver;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    GloballyConvergentMethodMovingAsymptotes(const locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
    locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & operator=(const locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_GLOBALLYCONVERGENTMETHODMOVINGASYMPTOTES_HPP_ */
