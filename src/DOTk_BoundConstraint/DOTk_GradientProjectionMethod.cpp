/*
 * DOTk_GradientProjectionMethod.cpp
 *
 *  Created on: Oct 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include <iomanip>
#include <sstream>
#include <iostream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_GradientProjectionMethod.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

GradientProjectionMethod::GradientProjectionMethod(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                   const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                                   const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_) :
        m_PrintOutputFile(false),
        m_IterationCount(0),
        m_MaxNumIterations(5000),
        m_ObjectiveTolerance(1e-8),
        m_ProjectedGradientTolerance(1e-8),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED),
        m_OutputFile(),
        m_Primal(primal_),
        m_Bounds(new dotk::DOTk_BoundConstraints),
        m_LineSearchStep(step_),
        m_Data(mng_)
{
}

GradientProjectionMethod::~GradientProjectionMethod()
{
}

void GradientProjectionMethod::getMin()
{
    this->checkBounds();

    // compute initial objective function value
    this->openOutputFile("DOTk_GradientProjectionDiagnostics.out");
    double objective_value = m_Data->evaluateObjective();
    m_Data->setNewObjectiveFunctionValue(objective_value);

    // compute initial gradient
    m_Data->computeGradient();
    double new_gradient_norm = m_Data->getNewGradient()->norm();
    m_Data->setNormNewGradient(new_gradient_norm);

    bool stop = false;
    this->diagnostics();
    while(stop == false)
    {
        // compute projected descent direction
        m_Data->getTrialStep()->update(1., *m_Data->getNewPrimal(), 0.);
        m_Data->getTrialStep()->update(static_cast<double>(-1.), *m_Data->getNewGradient(), 1.);
        m_Bounds->project(*m_Primal->getControlLowerBound(),
                          *m_Primal->getControlUpperBound(),
                          *m_Data->getTrialStep());
        m_Data->getTrialStep()->update(static_cast<double>(-1.), *m_Data->getNewPrimal(), 1.);

        // solve line search step sub-problem
        m_Data->getOldPrimal()->update(1., *m_Data->getNewPrimal(), 0.);
        m_Data->getOldGradient()->update(1., *m_Data->getNewGradient(), 0.);
        m_Data->setOldObjectiveFunctionValue(m_Data->getNewObjectiveFunctionValue());
        m_LineSearchStep->solveSubProblem(m_Data);

        // update gradient based on updated control
        m_Data->computeGradient();
        double new_gradient_norm = m_Data->getNewGradient()->norm();
        m_Data->setNormNewGradient(new_gradient_norm);

        m_IterationCount++;
        this->diagnostics();
        if(checkStoppinCritera() == true)
        {
            break;
        }
    }
    this->closeOutputFile();
    if(m_PrintOutputFile == true)
    {
        dotk::printControl(m_Data->getNewPrimal());
    }
}

void GradientProjectionMethod::printDiagnostics()
{
    m_PrintOutputFile = true;
}

void GradientProjectionMethod::setMaxNumIterations(size_t input_)
{
    m_MaxNumIterations = input_;
}

void GradientProjectionMethod::setObjectiveTolerance(double input_)
{
    m_ObjectiveTolerance = input_;
}

void GradientProjectionMethod::setProjectedGradientTolerance(double input_)
{
    m_ProjectedGradientTolerance = input_;
}

size_t GradientProjectionMethod::getIterationCount() const
{
    return (m_IterationCount);
}

dotk::types::stop_criterion_t GradientProjectionMethod::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

void GradientProjectionMethod::reset()
{
    m_Data->getNewPrimal()->update(1., *m_Data->getOldPrimal(), 0.);
    m_Data->getNewGradient()->update(1., *m_Data->getOldGradient(), 0.);
    m_Data->setNewObjectiveFunctionValue(m_Data->getOldObjectiveFunctionValue());
}

void GradientProjectionMethod::checkBounds()
{
    try
    {
        if(m_Primal->getControlLowerBound().use_count() <= 0)
        {
            std::ostringstream msg;
            msg << "\n\nDOTk ERROR: LOWER BOUND DATA IS NOT INITIALIZED. ERROR IN FILE: " << __FILE__ << ", LINE: "
                    << __LINE__ << ": ABORT\n\n";
            throw msg.str().c_str();
        }

        if(m_Primal->getControlUpperBound().use_count() <= 0)
        {
            std::ostringstream msg;
            msg << "\n\nDOTk ERROR: UPPER BOUND DATA IS NOT INITIALIZED. ERROR IN FILE: " << __FILE__ << ", LINE: "
                    << __LINE__ << ": ABORT\n\n";
            throw msg.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void GradientProjectionMethod::diagnostics()
{
    if(m_PrintOutputFile == false)
    {
        return;
    }

    const Real norm_gradient = m_Data->getNormNewGradient();
    const Real objective_func_value = m_Data->getNewObjectiveFunctionValue();
    const size_t num_objective_evaluations = m_Data->getRoutinesMng()->getObjectiveFunctionEvaluationCounter();
    if(m_IterationCount == 0)
    {
        m_OutputFile << std::setw(10) << std::right << "Iteration" << std::setw(12) << std::right << "Func-count"
                << std::setw(12) << std::right << "F(x)" << std::setw(12) << std::right << "norm(G)" << std::setw(12)
                << std::right << "norm(P)" << std::setw(15) << std::right << "LineSrch-Step" << std::setw(14)
                << std::right << "LineSrch-Itr" << "\n" << std::flush;
        m_OutputFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << m_IterationCount
                << std::setw(12) << std::right << num_objective_evaluations << std::setw(12) << std::right
                << objective_func_value << std::setw(12) << std::right << norm_gradient << std::setw(12) << std::right
                << "*" << std::setw(15) << std::right << "*" << std::setw(14) << std::right << "*" << "\n"
                << std::flush;
    }
    else
    {
        const Real trial_step_norm = m_Data->getNormTrialStep();
        m_OutputFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << m_IterationCount
                << std::setw(12) << std::right << num_objective_evaluations << std::setw(12) << std::right
                << objective_func_value << std::setw(12) << std::right << norm_gradient << std::setw(12) << std::right
                << trial_step_norm << std::setw(15) << std::right << m_LineSearchStep->step() << std::setw(14)
                << std::right << m_LineSearchStep->iterations() << "\n" << std::flush;
    }
}

void GradientProjectionMethod::openOutputFile(const char* const name_)
{
    if(m_PrintOutputFile == false)
    {
        return;
    }
    m_OutputFile.open(name_, std::ios::out | std::ios::trunc);
}

void GradientProjectionMethod::closeOutputFile()
{
    if(m_PrintOutputFile == false)
    {
        return;
    }
    m_OutputFile.close();
}

void GradientProjectionMethod::setStoppingCriterion(dotk::types::stop_criterion_t input_)
{
    m_StoppingCriterion = input_;
}

bool GradientProjectionMethod::checkStoppinCritera()
{
    bool stop = false;
    double new_gradient_norm = m_Data->getNormNewGradient();
    double new_projected_gradient_norm = m_Data->getTrialStep()->norm();
    m_Data->setNormTrialStep(new_projected_gradient_norm);
    double new_objective_function_value = m_Data->getNewObjectiveFunctionValue();

    if(std::isfinite(new_gradient_norm) == false)
    {
        this->reset();
        this->setStoppingCriterion(dotk::types::NaN_GRADIENT_NORM);
        stop = true;
    }
    else if(std::isfinite(new_projected_gradient_norm) == false)
    {
        this->reset();
        this->setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
        stop = true;
    }
    else if(new_projected_gradient_norm < m_ProjectedGradientTolerance)
    {
        this->setStoppingCriterion(dotk::types::TRIAL_STEP_TOL_SATISFIED);
        stop = true;
    }
    else if(new_objective_function_value < m_ObjectiveTolerance)
    {
        this->setStoppingCriterion(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED);
        stop = true;
    }
    else if(m_IterationCount >= m_MaxNumIterations)
    {
        this->setStoppingCriterion(dotk::types::MAX_NUM_ITR_REACHED);
        stop = true;
    }

    return (stop);
}

}
