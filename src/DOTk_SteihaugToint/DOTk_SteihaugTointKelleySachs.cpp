/*
 * DOTk_SteihaugTointKelleySachs.cpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_KelleySachsStepMng.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SteihaugTointNewtonIO.hpp"
#include "DOTk_SteihaugTointKelleySachs.hpp"
#include "DOTk_ProjectedSteihaugTointPcg.hpp"

namespace dotk
{

DOTk_SteihaugTointKelleySachs::DOTk_SteihaugTointKelleySachs(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                                             const std::shared_ptr<dotk::DOTk_KelleySachsStepMng> & step_mng) :
        dotk::DOTk_SteihaugTointNewton(),
        m_MaxNumUpdates(10),
        m_StationarityMeasure(0),
        m_WorkVector(data_mng_->getNewGradient()->clone()),
        m_IO(std::make_shared<dotk::DOTk_SteihaugTointNewtonIO>()),
        m_StepMng(step_mng),
        m_DataMng(data_mng_),
        m_BoundConstraint(std::make_shared<dotk::DOTk_BoundConstraints>()),
        m_Solver(std::make_shared<dotk::DOTk_ProjectedSteihaugTointPcg>(data_mng_->getPrimalStruc()))
{
}

DOTk_SteihaugTointKelleySachs::~DOTk_SteihaugTointKelleySachs()
{
}

void DOTk_SteihaugTointKelleySachs::setMaxNumUpdates(size_t input_)
{
    m_MaxNumUpdates = input_;
}

size_t DOTk_SteihaugTointKelleySachs::getMaxNumUpdates() const
{
    return (m_MaxNumUpdates);
}

void DOTk_SteihaugTointKelleySachs::setMaxNumSolverItr(size_t input_)
{
    m_Solver->setMaxNumItr(input_);
}

void DOTk_SteihaugTointKelleySachs::printDiagnosticsAndSolutionAtEveryItr()
{
    m_IO->setDisplayOption(dotk::types::ITERATION);
}

void DOTk_SteihaugTointKelleySachs::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->setDisplayOption(dotk::types::FINAL);
}

void DOTk_SteihaugTointKelleySachs::getMin()
{
    m_IO->openFile("DOTk_KelleySachsTrustRegionNewtonDiagnostics.out");

    Real new_objective_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(new_objective_value);
    m_DataMng->computeGradient();
    m_DataMng->getOldPrimal()->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_DataMng->getOldGradient()->update(1., *m_DataMng->getNewGradient(), 0.);
    Real norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(norm_gradient);
    if(m_StepMng->isInitialTrustRegionRadiusSetToGradNorm() == true)
    {
        m_StepMng->setTrustRegionRadius(norm_gradient);
    }
    this->computeStationarityMeasure();

    m_IO->printInitialDiagnostics(m_DataMng);

    size_t itr = 1;
    while(1)
    {
        this->updateNumOptimizationItrDone(itr);
        // Compute adaptive constants to ensure superlinear convergence
        Real value = std::pow(m_StationarityMeasure, static_cast<Real>(0.75));
        Real epsilon = std::min(static_cast<Real>(1e-3), value);
        m_StepMng->setEpsilon(epsilon);
        value = std::pow(m_StationarityMeasure, static_cast<Real>(0.95));
        Real eta = static_cast<Real>(0.1)*std::min(static_cast<Real>(1e-1), value);
        m_StepMng->setEta(eta);
        // Solve trust region subproblem
        m_StepMng->solveSubProblem(m_DataMng, m_Solver, m_IO);
        // Store current primal and gradient
        m_WorkVector->update(1., *m_DataMng->getNewPrimal(), 0.); // update current primal
        m_DataMng->getNewPrimal()->update(1., *m_StepMng->getMidPrimal(), 0.);
        // Compute new midpoint gradient
        m_DataMng->computeGradient();
        m_DataMng->getNewPrimal()->update(1., *m_WorkVector, 0.);  // reset current primal
        // Update current primal and gradient information
        this->updateDataManager();
        // Update stationarity measure
        this->computeStationarityMeasure();
        ++itr;
        if(this->checkStoppingCriteria() == true)
        {
            m_IO->printConvergedDiagnostics(m_DataMng, m_Solver, m_StepMng.get());
            break;
        }
    }

    m_IO->printSolution(m_DataMng->getNewPrimal());
    m_IO->closeFile();
}

void DOTk_SteihaugTointKelleySachs::updateNumOptimizationItrDone(const size_t & input_)
{
    m_IO->setNumOptimizationItrDone(input_);
    m_StepMng->setNumOptimizationItrDone(input_);
    dotk::DOTk_SteihaugTointNewton::setNumOptimizationItrDone(input_);
}

bool DOTk_SteihaugTointKelleySachs::checkStoppingCriteria()
{
    Real actual_reduction = m_StepMng->getActualReduction();
    Real norm_proj_gradient = m_DataMng->getNormNewGradient();
    Real objective_function_value = m_DataMng->getNewObjectiveFunctionValue();

    bool optimization_algorithm_converged = false;
    if(m_StationarityMeasure <= dotk::DOTk_SteihaugTointNewton::getTrialStepTolerance())
    {
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::TRIAL_STEP_TOL_SATISFIED);
    }
    else if(std::isfinite(m_StationarityMeasure) == false)
    {
        optimization_algorithm_converged = true;
        this->resetCurrentStateToPreviousState();
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    }
    else if(norm_proj_gradient < dotk::DOTk_SteihaugTointNewton::getGradientTolerance())
    {
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::GRADIENT_TOL_SATISFIED);
    }
    else if(std::isfinite(norm_proj_gradient) == false)
    {
        optimization_algorithm_converged = true;
        this->resetCurrentStateToPreviousState();
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::NaN_GRADIENT_NORM);
    }
    else if(std::abs(actual_reduction) <= dotk::DOTk_SteihaugTointNewton::getActualReductionTolerance())
    {
        // objective function stagnation
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED);
    }
    else if(objective_function_value <= dotk::DOTk_SteihaugTointNewton::getObjectiveTolerance())
    {
        // objective function stagnation
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED);
    }
    else if(dotk::DOTk_SteihaugTointNewton::getNumOptimizationItrDone()
            >= dotk::DOTk_SteihaugTointNewton::getMaxNumOptimizationItr())
    {
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::MAX_NUM_ITR_REACHED);
    }

    return (optimization_algorithm_converged);
}

void DOTk_SteihaugTointKelleySachs::computeStationarityMeasure()
{
    m_WorkVector->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_WorkVector->update(static_cast<Real>(-1.), *m_DataMng->getNewGradient(), 1.);
    m_BoundConstraint->project(*m_DataMng->getPrimalStruc()->getControlLowerBound(),
                               *m_DataMng->getPrimalStruc()->getControlUpperBound(),
                               *m_WorkVector);
    m_WorkVector->update(static_cast<Real>(1.), *m_DataMng->getNewPrimal(), -1.);
    m_WorkVector->elementWiseMultiplication(*m_Solver->getInactiveSet());

    m_StationarityMeasure = m_WorkVector->norm();
    m_DataMng->setNormTrialStep(m_StationarityMeasure);
}

void DOTk_SteihaugTointKelleySachs::resetCurrentStateToPreviousState()
{
    m_DataMng->getNewPrimal()->update(1., *m_DataMng->getOldPrimal(), 0.);
    m_DataMng->getNewGradient()->update(1., *m_DataMng->getOldGradient(), 0.);
    m_DataMng->setNewObjectiveFunctionValue(m_DataMng->getOldObjectiveFunctionValue());
}

bool DOTk_SteihaugTointKelleySachs::updatePrimal()
{
    bool primal_updated = false;

    Real xi = 1.;
    Real beta = 1e-2;
    Real alpha = beta;
    Real mu = 1. - 1e-4;
    Real mid_actual_reduction = m_StepMng->getActualReduction();
    Real mid_objective_value = m_StepMng->getMidObejectiveFunctionValue();

    size_t iteration = 0;
    size_t max_num_updates = this->getMaxNumUpdates();
    while(iteration < max_num_updates)
    {
        // Compute trial point
        Real lambda = -xi / alpha;
        m_WorkVector->update(1., *m_StepMng->getMidPrimal(), 0.);
        // NOTE: new gradient stores the current mid gradient
        m_WorkVector->update(lambda, *m_DataMng->getNewGradient(), 1.);
        m_BoundConstraint->project(*m_DataMng->getPrimalStruc()->getControlLowerBound(),
                                   *m_DataMng->getPrimalStruc()->getControlUpperBound(),
                                   *m_WorkVector);
        // Compute trial objective function
        Real trial_objective_value = m_DataMng->evaluateObjective(m_WorkVector);
        // Compute actual reduction
        Real actual_reduction = trial_objective_value - mid_objective_value;
        // Check convergence
        if(actual_reduction < -mu * mid_actual_reduction)
        {
            primal_updated = true;
            m_DataMng->getNewPrimal()->update(1., *m_WorkVector, 0.);
            m_StepMng->setActualReduction(actual_reduction);
            m_DataMng->setNewObjectiveFunctionValue(trial_objective_value);
            break;
        }
        // Compute scaling for next iteration
        if(iteration == 1)
        {
            xi = alpha;
        }
        else
        {
            xi = xi * beta;
        }
        ++iteration;
    }

    if(iteration >= max_num_updates)
    {
        m_DataMng->getNewPrimal()->update(1., *m_StepMng->getMidPrimal(), 0.);
        m_DataMng->setNewObjectiveFunctionValue(mid_objective_value);
    }

    return (primal_updated);
}

void DOTk_SteihaugTointKelleySachs::updateDataManager()
{
    Real current_objective_value = m_DataMng->getNewObjectiveFunctionValue();
    m_DataMng->setOldObjectiveFunctionValue(current_objective_value);
    m_DataMng->getOldPrimal()->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_DataMng->getOldGradient()->update(1., *m_DataMng->getNewGradient(), 0.);

    if(this->updatePrimal() == true)
    {
        // update new gradient since primal was successfully updated;
        // else, keep mid gradient and thus mid primal
        m_DataMng->computeGradient();
    }

    m_WorkVector->update(1., *m_DataMng->getNewGradient(), 0.);
    m_WorkVector->elementWiseMultiplication(*m_Solver->getInactiveSet());
    Real norm_proj_gradient = m_WorkVector->norm();
    m_DataMng->setNormNewGradient(norm_proj_gradient);
}

}
