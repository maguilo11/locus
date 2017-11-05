/*
 * Locus_ConservativeConvexSeparableAppxAlgorithm.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXALGORITHM_HPP_
#define LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXALGORITHM_HPP_

#include <memory>
#include <cassert>
#include <stdexcept>
#include <algorithm>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_MultiVectorList.hpp"
#include "Locus_PrimalProblemStageMng.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"
#include "Locus_ConservativeConvexSeparableApproximation.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxAlgorithm
{
public:
    ConservativeConvexSeparableAppxAlgorithm(const std::shared_ptr<locus::PrimalProblemStageMng<ScalarType, OrdinalType>> & aPrimalStageMng,
                                             const std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType>> & aDataMng,
                                             const std::shared_ptr<locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>> & aSubProblem) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mOptimalityTolerance(1e-4),
            mStagnationTolerance(1e-8),
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
            mStageMng(aPrimalStageMng),
            mDataMng(aDataMng),
            mSubProblem(aSubProblem)
    {
        this->initialize(aDataMng.operator*());
    }
    ~ConservativeConvexSeparableAppxAlgorithm()
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
        if(mDataMng->isInitialGuessSet() == false)
        {
            std::runtime_error("***** CONTROL INITIAL GUESS WAS NOT DEFINED ****");
        }

        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        const ScalarType tCurrentObjectiveFunctionValue = mStageMng->evaluateObjective(tControl);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mStageMng->evaluateConstraints(tControl, mDualWork.operator*());
        mDataMng->setDual(mDualWork.operator*());
        mSubProblem->initializeAuxiliaryVariables(mDataMng.operator*());

        while(1)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
            mStageMng->computeGradient(tCurrentControl, mControlWork.operator*());
            mDataMng->setCurrentObjectiveGradient(mControlWork.operator*());
            mStageMng->computeConstraintGradients(tCurrentControl, mWorkMultiVectorList.operator*());
            mDataMng->setCurrentConstraintGradients(mWorkMultiVectorList.operator*());

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

            mSubProblem->solve(mStageMng.operator*(), mDataMng.operator*());
            mStageMng->update(mDataMng.operator*());

            mNumIterationsDone++;
            if(this->checkStoppingCriteria() == true)
            {
                break;
            }
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
        mDataMng->computeNormInactiveGradient();
        mDataMng->computeObjectiveStagnationMeasure();

        const locus::MultiVector<ScalarType, OrdinalType> & tDual = mDataMng->getDual();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        mDataMng->computeKarushKuhnTuckerConditionsInexactness(tControl, tDual);

        const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
        const ScalarType tFeasibilityMeasure = mDataMng->getFeasibilityMeasure();
        const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ScalarType tNormInactiveGradient = mDataMng->getNormInactiveGradient();
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
                && (tNormInactiveGradient < this->getOptimalityTolerance()) )
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::OPTIMALITY_AND_FEASIBILITY_MET);
        }
        else if(t_KKT_ConditionsInexactness < this->getKarushKuhnTuckerConditionsTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::KKT_CONDITIONS_TOLERANCE);
        }
        else if(mNumIterationsDone >= this->getMaxNumIterations())
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

                const locus::Vector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tUpperBounds = mDataMng->getControlUpperBounds(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tLowerBounds = mDataMng->getControlLowerBounds(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tPreviousControl = mDataMng->getPreviousControl(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tPreviousSigma = mPreviousSigma->operator[](tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tAntepenultimateControl = mAntepenultimateControl->operator[](tVectorIndex);

                locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = mControlWork->operator[](tVectorIndex);

                const OrdinalType tNumberControls = tCurrentSigma.size();
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

    std::shared_ptr<locus::PrimalProblemStageMng<ScalarType, OrdinalType>> mStageMng;
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType>> mDataMng;
    std::shared_ptr<locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>> mSubProblem;

private:
    ConservativeConvexSeparableAppxAlgorithm(const locus::ConservativeConvexSeparableAppxAlgorithm<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableAppxAlgorithm<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableAppxAlgorithm<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXALGORITHM_HPP_ */
