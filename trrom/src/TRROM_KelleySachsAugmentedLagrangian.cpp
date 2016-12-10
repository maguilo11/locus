/*
 * TRROM_KelleySachsAugmentedLagrangian.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <string>

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_SteihaugTointNewtonIO.hpp"
#include "TRROM_ProjectedSteihaugTointPcg.hpp"
#include "TRROM_AugmentedLagrangianDataMng.hpp"
#include "TRROM_KelleySachsAugmentedLagrangian.hpp"

namespace trrom
{

KelleySachsAugmentedLagrangian::KelleySachsAugmentedLagrangian
(const std::tr1::shared_ptr<trrom::Data> & data_,
 const std::tr1::shared_ptr<trrom::AugmentedLagrangianDataMng> & data_mng_,
 const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_mng) :
        trrom::TrustRegionKelleySachs(data_),
        m_Gamma(1e-3),
        m_OptimalityTolerance(1e-3),
        m_FeasibilityTolerance(1e-3),
        m_WorkVector(data_->control()->create()),
        m_MidGradient(data_->control()->create()),
        m_IO(new trrom::SteihaugTointNewtonIO),
        m_StepMng(step_mng),
        m_Solver(new trrom::ProjectedSteihaugTointPcg(data_)),
        m_DataMng(data_mng_)
{
}

KelleySachsAugmentedLagrangian::~KelleySachsAugmentedLagrangian()
{
}

void KelleySachsAugmentedLagrangian::setOptimalityTolerance(double input_)
{
    m_OptimalityTolerance = input_;
}

void KelleySachsAugmentedLagrangian::setFeasibilityTolerance(double input_)
{
    m_FeasibilityTolerance = input_;
}

void KelleySachsAugmentedLagrangian::getMin()
{
    std::string name("KelleySachsAugmentedLagrangianDiagnostics.out");
    m_IO->openFile(name);

    double new_objective_value = m_DataMng->evaluateObjective();
    m_DataMng->updateInequalityConstraintValues();
    m_DataMng->setNewObjectiveFunctionValue(new_objective_value);
    m_DataMng->computeGradient();
    m_DataMng->getOldPrimal()->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_DataMng->getOldGradient()->update(1., *m_DataMng->getNewGradient(), 0.);
    double norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(norm_gradient);
    if(m_StepMng->isInitialTrustRegionRadiusSetToGradNorm() == true)
    {
        m_StepMng->setTrustRegionRadius(norm_gradient);
    }
    trrom::TrustRegionKelleySachs::computeStationarityMeasure(m_DataMng, m_Solver->getInactiveSet());
    m_IO->printInitialDiagnostics(m_DataMng);

    int itr = 1;
    while(1)
    {
        this->updateNumOptimizationItrDone(itr);
        // Compute adaptive constants to ensure superlinear convergence
        double measure = trrom::TrustRegionKelleySachs::getStationarityMeasure();
        double value = std::pow(measure, static_cast<double>(0.75));
        double epsilon = std::min(static_cast<double>(1e-3), value);
        m_StepMng->setEpsilon(epsilon);
        value = std::pow(measure, static_cast<double>(0.95));
        double eta = static_cast<double>(0.1) * std::min(static_cast<double>(1e-1), value);
        m_StepMng->setEta(eta);
        // Solve trust region subproblem
        m_StepMng->solveSubProblem(m_DataMng, m_Solver, m_IO);
        // Update mid point inequality values
        m_DataMng->updateInequalityConstraintValues();
        // Compute new midpoint gradient
        m_DataMng->computeGradient(m_StepMng->getMidPrimal(), m_MidGradient);
        // Update current primal and gradient information
        this->updateDataManager();
        if(this->checkStoppingCriteria() == true)
        {
            m_IO->printConvergedDiagnostics(m_DataMng, m_Solver, m_StepMng.get());
            break;
        }
        // Update stationarity measure
        trrom::TrustRegionKelleySachs::computeStationarityMeasure(m_DataMng, m_Solver->getInactiveSet());
        ++ itr;
    }

    m_IO->printSolution(m_DataMng->getNewPrimal());
    m_IO->closeFile();
}

void KelleySachsAugmentedLagrangian::printDiagnostics()
{
    m_IO->setDisplayOption(trrom::types::FINAL);
}

void KelleySachsAugmentedLagrangian::updateNumOptimizationItrDone(const int & input_)
{
    m_IO->setNumOptimizationItrDone(input_);
    trrom::TrustRegionKelleySachs::setNumOptimizationItrDone(input_);
}

void KelleySachsAugmentedLagrangian::updateDataManager()
{
    // set new objective function value
    double current_objective_value = m_DataMng->getNewObjectiveFunctionValue();
    m_DataMng->setOldObjectiveFunctionValue(current_objective_value);
    // update primal vector
    m_DataMng->getOldPrimal()->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_DataMng->getOldGradient()->update(1., *m_DataMng->getNewGradient(), 0.);

    if(trrom::TrustRegionKelleySachs::updatePrimal(m_StepMng, m_DataMng, m_MidGradient) == true)
    {
        // update new gradient and inequality constraint values since primal was
        // successfully updated; else, keep mid gradient and thus mid primal
        m_DataMng->updateInequalityConstraintValues();
        m_DataMng->computeGradient();
    }
    else
    {
        m_DataMng->getNewGradient()->update(1., *m_MidGradient, 0.);
    }

    // Compute stagnation measure
    m_WorkVector->update(1., *m_DataMng->getOldPrimal(), 0.);
    m_WorkVector->update(-1., *m_DataMng->getNewPrimal(), 1.);
    double stagnation_measure = m_WorkVector->norm();
    m_DataMng->setStagnationMeasure(stagnation_measure);
    // compute norm of projected gradient
    m_WorkVector->update(1., *m_DataMng->getNewGradient(), 0.);
    m_WorkVector->elementWiseMultiplication(*m_Solver->getInactiveSet());
    double norm_proj_gradient = m_WorkVector->norm();
    m_DataMng->setNormNewGradient(norm_proj_gradient);
    // compute gradient inexactness bound
    m_StepMng->updateGradientInexactnessTolerance(norm_proj_gradient);
    double gradient_inexactness_tolerance = m_DataMng->getGradientInexactnessTolerance();
    m_DataMng->setGradientInexactnessTolerance(gradient_inexactness_tolerance);
}

bool KelleySachsAugmentedLagrangian::checkStoppingCriteria()
{
    bool stop = false;
    double condition = m_Gamma * m_DataMng->getPenalty();
    double norm_augmented_lagrangian_gradient = m_DataMng->getNormNewGradient();
    if(norm_augmented_lagrangian_gradient <= condition)
    {
        if(this->checkPrimaryStoppingCriteria() == true)
        {
            stop = true;
        }
        else
        {
            // Update Lagrange multipliers and stop if penalty is below defined threshold/tolerance
            stop = m_DataMng->updateLagrangeMultipliers();
        }
    }
    else
    {
        double optimality_measure = m_DataMng->getNormLagrangianGradient();
        double feasibility_measure = m_DataMng->getNormInequalityConstraints();
        int iteration_count = trrom::TrustRegionKelleySachs::getNumOptimizationItrDone();
        if((optimality_measure < m_OptimalityTolerance) && (feasibility_measure < m_FeasibilityTolerance))
        {
            this->setStoppingCriterion(trrom::types::OPTIMALITY_AND_FEASIBILITY_SATISFIED);
            stop = true;
        }
        else if(iteration_count >= trrom::TrustRegionKelleySachs::getMaxNumOptimizationItr())
        {
            stop = true;
            this->setStoppingCriterion(trrom::types::MAX_NUM_ITR_REACHED);
        }
    }

    return (stop);
}

bool KelleySachsAugmentedLagrangian::checkPrimaryStoppingCriteria()
{
    bool stop = false;
    if(this->checkNaN() == true)
    {
        // Stop optimization algorithm: NaN number detected
        stop = true;
        this->resetCurrentStateToPreviousState(m_DataMng);
    }
    else
    {
        double stationarity_measure = this->getStationarityMeasure();
        double stagnation_measure = m_DataMng->getStagnationMeasure();
        double optimality_measure = m_DataMng->getNormLagrangianGradient();
        double feasibility_measure = m_DataMng->getNormInequalityConstraints();

        if(stationarity_measure <= this->getTrialStepTolerance())
        {
            stop = true;
            this->setStoppingCriterion(trrom::types::TRIAL_STEP_TOL_SATISFIED);
        }
        else if(stagnation_measure < this->getStagnationTolerance())
        {
            stop = true;
            this->setStoppingCriterion(trrom::types::STAGNATION_MEASURE);
        }
        else if((optimality_measure < m_OptimalityTolerance) && (feasibility_measure < m_FeasibilityTolerance))
        {
            stop = true;
            this->setStoppingCriterion(trrom::types::OPTIMALITY_AND_FEASIBILITY_SATISFIED);
        }
        else if(this->getNumOptimizationItrDone() >= this->getMaxNumOptimizationItr())
        {
            stop = true;
            this->setStoppingCriterion(trrom::types::MAX_NUM_ITR_REACHED);
            this->setStoppingCriterion(trrom::types::OPTIMALITY_AND_FEASIBILITY_SATISFIED);
        }
    }

    return (stop);
}

bool KelleySachsAugmentedLagrangian::checkNaN()
{
    bool nan_value_detected = false;
    double norm_proj_gradient = m_DataMng->getNormNewGradient();
    double stationarity_measure = this->getStationarityMeasure();
    double optimality_measure = m_DataMng->getNormLagrangianGradient();
    double feasibility_measure = m_DataMng->getNormInequalityConstraints();

    if(std::isfinite(stationarity_measure) == false)
    {
        nan_value_detected = true;

        this->setStoppingCriterion(trrom::types::NaN_TRIAL_STEP_NORM);
    }
    else if(std::isfinite(norm_proj_gradient) == false)
    {
        nan_value_detected = true;
        this->setStoppingCriterion(trrom::types::NaN_GRADIENT_NORM);
    }
    else if(std::isfinite(optimality_measure) == false)
    {
        nan_value_detected = true;
        this->setStoppingCriterion(trrom::types::NaN_OPTIMALITY_NORM);
    }
    else if(std::isfinite(feasibility_measure) == false)
    {
        nan_value_detected = true;
        this->setStoppingCriterion(trrom::types::NaN_FEASIBILITY_NORM);
    }

    return (nan_value_detected);
}

}
