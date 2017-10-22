/*
 * Locus_KelleySachsAugmentedLagrangian.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_KELLEYSACHSAUGMENTEDLAGRANGIAN_HPP_
#define LOCUS_KELLEYSACHSAUGMENTEDLAGRANGIAN_HPP_

#include <cmath>
#include <memory>
#include <cassert>

#include "Locus_Types.hpp"
#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_KelleySachsStepMng.hpp"
#include "Locus_KelleySachsAlgorithm.hpp"
#include "Locus_ProjectedSteihaugTointPcg.hpp"
#include "Locus_TrustRegionAlgorithmDataMng.hpp"
#include "Locus_AugmentedLagrangianStageMng.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class KelleySachsAugmentedLagrangian : public locus::KelleySachsAlgorithm<ScalarType, OrdinalType>
{
public:
    KelleySachsAugmentedLagrangian(const std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> & aDataFactory,
                                   const std::shared_ptr<locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType>> & aDataMng,
                                   const std::shared_ptr<locus::AugmentedLagrangianStageMng<ScalarType, OrdinalType>> & aStageMng) :
            locus::KelleySachsAlgorithm<ScalarType, OrdinalType>(*aDataFactory),
            mGammaConstant(1e-3),
            mOptimalityTolerance(1e-5),
            mFeasibilityTolerance(1e-4),
            mGradient(aDataFactory->control().create()),
            mStepMng(std::make_shared<locus::KelleySachsStepMng<ScalarType, OrdinalType>>(*aDataFactory)),
            mSolver(std::make_shared<locus::ProjectedSteihaugTointPcg<ScalarType, OrdinalType>>(*aDataFactory)),
            mDataMng(aDataMng),
            mStageMng(aStageMng)
    {
    }
    virtual ~KelleySachsAugmentedLagrangian()
    {
    }

    void setOptimalityTolerance(const ScalarType & aInput)
    {
        mOptimalityTolerance = aInput;
    }
    void setFeasibilityTolerance(const ScalarType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }

    void solve()
    {
        assert(mDataMng->isInitialGuessSet() == true);

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
        ScalarType tTolerance = mStepMng->getObjectiveInexactnessTolerance();
        ScalarType tCurrentObjectiveFunctionValue = mStageMng->evaluateObjective(tCurrentControl, tTolerance);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mStageMng->updateCurrentConstraintValues();

        mStageMng->computeGradient(tCurrentControl, *mGradient);
        mDataMng->setCurrentGradient(*mGradient);
        mDataMng->computeNormProjectedGradient();
        mDataMng->storeCurrentStageData();

        if(mStepMng->isInitialTrustRegionRadiusSetToNormProjectedGradient() == true)
        {
            ScalarType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
            mStepMng->setTrustRegionRadius(tNormProjectedGradient);
        }
        mDataMng->computeStationarityMeasure();

        OrdinalType tIteration = 0;
        while(1)
        {
            tIteration++;
            this->setNumIterationsDone(tIteration);
            // Compute adaptive constants to ensure superlinear convergence
            ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
            ScalarType tValue = std::pow(tStationarityMeasure, static_cast<ScalarType>(0.75));
            ScalarType tEpsilon = std::min(static_cast<ScalarType>(1e-3), tValue);
            mStepMng->setEpsilonConstant(tEpsilon);
            tValue = std::pow(tStationarityMeasure, static_cast<ScalarType>(0.95));
            ScalarType tEta = static_cast<ScalarType>(0.1) * std::min(static_cast<ScalarType>(1e-1), tValue);
            mStepMng->setEtaConstant(tEta);
            // Solve trust region subproblem
            mStepMng->solveSubProblem(*mDataMng, *mStageMng, *mSolver);
            // Update mid objective, control, and gradient information if necessary
            this->updateDataManager();
            // Update stage manager data
            mStageMng->update(mDataMng.operator*());
            if(this->checkStoppingCriteria() == true)
            {
                break;
            }
        }
    }

private:
    void updateDataManager()
    {
        // Store current objective function, control, and gradient values
        mDataMng->storeCurrentStageData();

        // Update inequality constraint values at mid point
        mStageMng->updateCurrentConstraintValues();
        // Compute gradient at new midpoint
        const locus::MultiVector<ScalarType, OrdinalType> & tMidControl = mStepMng->getMidPointControls();
        mStageMng->computeGradient(tMidControl, *mGradient);

        if(this->updateControl(*mGradient, *mStepMng, *mDataMng, *mStageMng) == true)
        {
            // Update new gradient and inequality constraint values since control
            // was successfully updated; else, keep mid gradient and thus mid control.
            mStageMng->updateCurrentConstraintValues();
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
            mStageMng->computeGradient(tCurrentControl, *mGradient);
            mDataMng->setCurrentGradient(*mGradient);
        }
        else
        {
            // Keep current objective function, control, and gradient values at mid point
            const ScalarType tMidObjectiveFunctionValue = mStepMng->getMidPointObjectiveFunctionValue();
            mDataMng->setCurrentObjectiveFunctionValue(tMidObjectiveFunctionValue);
            mDataMng->setCurrentControl(tMidControl);
            mDataMng->setCurrentGradient(*mGradient);
        }

        // Compute feasibility measure
        mStageMng->computeCurrentFeasibilityMeasure();
        // Compute norm of projected gradient
        mDataMng->computeNormProjectedGradient();
        // Compute stationarity measure
        mDataMng->computeStationarityMeasure();
        // Compute stagnation measure
        mDataMng->computeStagnationMeasure();
        // compute gradient inexactness bound
        ScalarType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
        mStepMng->updateGradientInexactnessTolerance(tNormProjectedGradient);
    }

    bool checkStoppingCriteria()
    {
        bool tStop = false;
        ScalarType tCurrentLagrangeMultipliersPenalty = mStageMng->getCurrentLagrangeMultipliersPenalty();
        ScalarType tTolerance = mGammaConstant * tCurrentLagrangeMultipliersPenalty;
        ScalarType tNormAugmentedLagrangianGradient = mDataMng->getNormProjectedGradient();
        if(tNormAugmentedLagrangianGradient <= tTolerance)
        {
            if(this->checkPrimaryStoppingCriteria() == true)
            {
                tStop = true;
            }
            else
            {
                // Update Lagrange multipliers and stop if penalty is below defined threshold/tolerance
                tStop = mStageMng->updateLagrangeMultipliers();
            }
        }
        else
        {
            const OrdinalType tIterationCount = this->getNumIterationsDone();
            const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
            const ScalarType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
            const ScalarType tOptimalityMeasure = mDataMng->getNormProjectedGradient();
            if( (tOptimalityMeasure < mOptimalityTolerance) && (tFeasibilityMeasure < mFeasibilityTolerance) )
            {
                this->setStoppingCriterion(locus::algorithm::stop_t::OPTIMALITY_AND_FEASIBILITY);
                tStop = true;
            }
            else if( tStagnationMeasure < this->getStagnationTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::CONTROL_STAGNATION);
            }
            else if( tIterationCount >= this->getMaxNumIterations() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS);
            }
        }

        return (tStop);
    }

    bool checkPrimaryStoppingCriteria()
    {
        bool tStop = false;
        if(this->checkNaN() == true)
        {
            // Stop optimization algorithm: NaN number detected
            tStop = true;
            mDataMng->resetCurrentStageDataToPreviousStageData();
        }
        else
        {
            const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
            const ScalarType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
            const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
            const ScalarType tOptimalityMeasure = mDataMng->getNormProjectedGradient();

            if( tStationarityMeasure <= this->getTrialStepTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::NORM_STEP);
            }
            else if( tStagnationMeasure < this->getStagnationTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::CONTROL_STAGNATION);
            }
            else if( (tOptimalityMeasure < mOptimalityTolerance) && (tFeasibilityMeasure < mFeasibilityTolerance) )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::OPTIMALITY_AND_FEASIBILITY);
            }
            else if( this->getNumIterationsDone() >= this->getMaxNumIterations() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS);
            }
        }

        return (tStop);
    }

    bool checkNaN()
    {
        const ScalarType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
        const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ScalarType tOptimalityMeasure = mDataMng->getNormProjectedGradient();
        const ScalarType tNormProjectedAugmentedLagrangianGradient = mDataMng->getNormProjectedGradient();

        bool tNaN_ValueDetected = false;
        if(std::isfinite(tStationarityMeasure) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_NORM_TRIAL_STEP);
        }
        else if(std::isfinite(tNormProjectedAugmentedLagrangianGradient) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_NORM_GRADIENT);
        }
        else if(std::isfinite(tOptimalityMeasure) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_OBJECTIVE_GRADIENT);
        }
        else if(std::isfinite(tFeasibilityMeasure) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_FEASIBILITY_VALUE);
        }

        return (tNaN_ValueDetected);
    }

private:
    ScalarType mGammaConstant;
    ScalarType mOptimalityTolerance;
    ScalarType mFeasibilityTolerance;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mGradient;

    std::shared_ptr<locus::KelleySachsStepMng<ScalarType, OrdinalType>> mStepMng;
    std::shared_ptr<locus::ProjectedSteihaugTointPcg<ScalarType,OrdinalType>> mSolver;
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<ScalarType,OrdinalType>> mDataMng;
    std::shared_ptr<locus::AugmentedLagrangianStageMng<ScalarType,OrdinalType>> mStageMng;

private:
    KelleySachsAugmentedLagrangian(const locus::KelleySachsAugmentedLagrangian<ScalarType, OrdinalType> & aRhs);
    locus::KelleySachsAugmentedLagrangian<ScalarType, OrdinalType> & operator=(const locus::KelleySachsAugmentedLagrangian<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_KELLEYSACHSAUGMENTEDLAGRANGIAN_HPP_ */
