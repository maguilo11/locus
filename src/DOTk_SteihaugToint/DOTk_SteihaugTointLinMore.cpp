/*
 * DOTk_SteihaugTointLinMore.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */


#include "vector.hpp"
#include "DOTk_SteihaugTointPcg.hpp"
#include "DOTk_TrustRegionStepMng.hpp"
#include "DOTk_SteihaugTointLinMore.hpp"
#include "DOTk_SteihaugTointNewtonIO.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_SteihaugTointLinMore::DOTk_SteihaugTointLinMore(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_TrustRegionStepMng> & step_mng) :
        dotk::DOTk_SteihaugTointNewton(),
        m_Solver(new dotk::DOTk_SteihaugTointPcg(data_mng_->getPrimalStruc())),
        m_IO(new dotk::DOTk_SteihaugTointNewtonIO),
        m_StepMng(step_mng),
        m_DataMng(data_mng_)
{
}

DOTk_SteihaugTointLinMore::~DOTk_SteihaugTointLinMore()
{
}

void DOTk_SteihaugTointLinMore::setSolverMaxNumItr(size_t input_)
{
    m_Solver->setMaxNumItr(input_);
}

size_t DOTk_SteihaugTointLinMore::getSolverMaxNumItr() const
{
    return (m_Solver->getMaxNumItr());
}

void DOTk_SteihaugTointLinMore::setSolverRelativeTolerance(Real input_)
{
    m_Solver->setRelativeTolerance(input_);
}

Real DOTk_SteihaugTointLinMore::getSolverRelativeTolerance() const
{
    return (m_Solver->getRelativeTolerance());
}

void DOTk_SteihaugTointLinMore::setSolverRelativeToleranceExponential(Real input_)
{
    m_Solver->setRelativeToleranceExponential(input_);
}

Real DOTk_SteihaugTointLinMore::getSolverRelativeToleranceExponential() const
{
    return (m_Solver->getRelativeToleranceExponential());
}

void DOTk_SteihaugTointLinMore::printDiagnosticsAndSolutionAtEveryItr()
{
    m_IO->setDisplayOption(dotk::types::ITERATION);
}

void DOTk_SteihaugTointLinMore::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->setDisplayOption(dotk::types::FINAL);
}

void DOTk_SteihaugTointLinMore::getMin()
{
    m_IO->openFile("DOTk_LinMoreTrustRegionNewtonDiagnostics.out");

    Real new_objective_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(new_objective_value);
    m_DataMng->computeGradient();
    m_DataMng->getOldPrimal()->copy(*m_DataMng->getNewPrimal());
    m_DataMng->getOldGradient()->copy(*m_DataMng->getNewGradient());
    Real norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(norm_gradient);
    if(m_StepMng->isInitialTrustRegionRadiusSetToGradNorm() == true)
    {
        m_StepMng->setTrustRegionRadius(norm_gradient);
    }

    m_IO->printInitialDiagnostics(m_DataMng);

    size_t itr = 1;
    while(1)
    {
        this->updateNumOptimizationItrDone(itr);
        m_StepMng->solveSubProblem(m_DataMng, m_Solver, m_IO);

        if(this->checkStoppingCriteria() == true)
        {
            m_IO->printConvergedDiagnostics(m_DataMng, m_Solver, m_StepMng.get());
            break;
        }
        ++itr;
    }

    m_IO->printSolution(m_DataMng->getNewPrimal());
    m_IO->closeFile();
}

void DOTk_SteihaugTointLinMore::updateNumOptimizationItrDone(const size_t & input_)
{
    m_IO->setNumOptimizationItrDone(input_);
    m_StepMng->setNumOptimizationItrDone(input_);
    dotk::DOTk_SteihaugTointNewton::setNumOptimizationItrDone(input_);
}

bool DOTk_SteihaugTointLinMore::checkStoppingCriteria()
{
    bool optimization_algorithm_converged = false;
    if(dotk::DOTk_SteihaugTointNewton::getNumOptimizationItrDone() < 1)
    {
        return (optimization_algorithm_converged);
    }
    Real grad_norm = m_DataMng->getNormNewGradient();
    Real trial_step_norm = m_DataMng->getNormTrialStep();
    if(std::isnan(trial_step_norm))
    {
        optimization_algorithm_converged = true;
        this->resetCurrentStateToPreviousState();
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    }
    else if(std::isnan(grad_norm))
    {
        optimization_algorithm_converged = true;
        this->resetCurrentStateToPreviousState();
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::NaN_GRADIENT_NORM);
    }
    else if(trial_step_norm < dotk::DOTk_SteihaugTointNewton::getTrialStepTolerance())
    {
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::TRIAL_STEP_TOL_SATISFIED);
    }
    else if(grad_norm < dotk::DOTk_SteihaugTointNewton::getGradientTolerance())
    {
        optimization_algorithm_converged = true;
        dotk::DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::GRADIENT_TOL_SATISFIED);
    }
    else if(m_DataMng->getNewObjectiveFunctionValue() < dotk::DOTk_SteihaugTointNewton::getObjectiveTolerance())
    {
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

void DOTk_SteihaugTointLinMore::resetCurrentStateToPreviousState()
{
    m_DataMng->getNewPrimal()->copy(*m_DataMng->getOldPrimal());
    m_DataMng->getNewGradient()->copy(*m_DataMng->getOldGradient());
    m_DataMng->setNewObjectiveFunctionValue(m_DataMng->getOldObjectiveFunctionValue());
}

}
