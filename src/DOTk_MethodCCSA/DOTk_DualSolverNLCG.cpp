/*
 * DOTk_DualSolverNLCG.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <vector>
#include <limits>
#include <string>
#include <iostream>
#include <algorithm>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_DataMngNonlinearCG.hpp"
#include "DOTk_ScaleParametersNLCG.hpp"

namespace dotk
{

DOTk_DualSolverNLCG::DOTk_DualSolverNLCG(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_DualSolverCCSA(dotk::ccsa::dual_solver_t::NONLINEAR_CG),
        m_NonlinearCgType(dotk::types::POLAK_RIBIERE_NLCG),
        m_TrialDual(primal_->dual()->clone()),
        m_ProjectedStep(primal_->dual()->clone()),
        m_DualLowerBound(primal_->dual()->clone()),
        m_DualUpperBound(primal_->dual()->clone()),
        m_Bounds(new dotk::DOTk_BoundConstraints()),
        m_DataMng(new dotk::DOTk_DataMngNonlinearCG(primal_->dual()))
{
    this->initialize(primal_);
}

DOTk_DualSolverNLCG::~DOTk_DualSolverNLCG()
{
}

dotk::types::nonlinearcg_t DOTk_DualSolverNLCG::getNonlinearCgType() const
{
    return (m_NonlinearCgType);
}

void DOTk_DualSolverNLCG::setNonlinearCgType(dotk::types::nonlinearcg_t input_)
{
    switch(input_)
    {
        case dotk::types::FLETCHER_REEVES_NLCG:
        {
            this->setFletcherReevesNLCG();
            break;
        }
        case dotk::types::POLAK_RIBIERE_NLCG:
        {
            this->setPolakRibiereNLCG();
            break;
        }
        case dotk::types::HESTENES_STIEFEL_NLCG:
        {
            this->setHestenesStiefelNLCG();
            break;
        }
        case dotk::types::DAI_YUAN_NLCG:
        {
            this->setDaiYuanNLCG();
            break;
        }
        case dotk::types::LIU_STOREY_NLCG:
        {
            this->setLiuStoreyNLCG();
            break;
        }
        case dotk::types::CONJUGATE_DESCENT_NLCG:
        {
            this->setConjugateDescentNLCG();
            break;
        }
        case dotk::types::DANIELS_NLCG:
        case dotk::types::DAI_LIAO_NLCG:
        case dotk::types::UNDEFINED_NLCG:
        case dotk::types::HAGER_ZHANG_NLCG:
        case dotk::types::PERRY_SHANNO_NLCG:
        case dotk::types::DAI_YUAN_HYBRID_NLCG:
        default:
        {
            this->setPolakRibiereNLCG();
            break;
        }
    }
}

void DOTk_DualSolverNLCG::setFletcherReevesNLCG()
{
    m_NonlinearCgType = dotk::types::FLETCHER_REEVES_NLCG;
}

void DOTk_DualSolverNLCG::setPolakRibiereNLCG()
{
    m_NonlinearCgType = dotk::types::POLAK_RIBIERE_NLCG;
}

void DOTk_DualSolverNLCG::setHestenesStiefelNLCG()
{
    m_NonlinearCgType = dotk::types::HESTENES_STIEFEL_NLCG;
}

void DOTk_DualSolverNLCG::setDaiYuanNLCG()
{
    m_NonlinearCgType = dotk::types::DAI_YUAN_NLCG;
}

void DOTk_DualSolverNLCG::setLiuStoreyNLCG()
{
    m_NonlinearCgType = dotk::types::LIU_STOREY_NLCG;
}

void DOTk_DualSolverNLCG::setConjugateDescentNLCG()
{
    m_NonlinearCgType = dotk::types::CONJUGATE_DESCENT_NLCG;
}

Real DOTk_DualSolverNLCG::getNewObjectiveFunctionValue() const
{
    return (m_DataMng->m_NewObjectiveFunctionValue);
}

Real DOTk_DualSolverNLCG::getOldObjectiveFunctionValue() const
{
    return (m_DataMng->m_OldObjectiveFunctionValue);
}

void DOTk_DualSolverNLCG::reset()
{
    m_DataMng->reset();
}
void DOTk_DualSolverNLCG::solve(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & solution_)
{
    m_DataMng->m_NewDual->update(1., *solution_, 0.);
    m_DataMng->m_NewObjectiveFunctionValue = objective_->value(*m_DataMng->m_NewDual);
    objective_->gradient(*solution_, *m_DataMng->m_NewGradient);

    m_DataMng->m_NewSteepestDescent->update(-1., *m_DataMng->m_NewGradient, 0.);
    m_TrialDual->update(1., *m_DataMng->m_NewDual, 0.);
    m_TrialDual->update(static_cast<Real>(1), *m_DataMng->m_NewSteepestDescent, 1.);
    m_Bounds->computeProjectedGradient(*m_TrialDual,
                                       *m_DualLowerBound,
                                       *m_DualUpperBound,
                                       *m_DataMng->m_NewSteepestDescent);
    m_DataMng->m_NewTrialStep->update(1., *m_DataMng->m_NewSteepestDescent, 0.);

    m_DataMng->storeCurrentState();
    this->step(objective_);
    if(this->stoppingCriteriaSatisfied() == true)
    {
        return;
    }
    dotk::DOTk_DualSolverCCSA::resetIterationCount();
    size_t max_num_iterations = dotk::DOTk_DualSolverCCSA::getMaxNumIterations();
    while(dotk::DOTk_DualSolverCCSA::getIterationCount() < max_num_iterations)
    {
        objective_->gradient(*m_DataMng->m_NewDual, *m_DataMng->m_NewGradient);

        m_DataMng->m_NewSteepestDescent->update(-1., *m_DataMng->m_NewGradient, 0.);

        m_TrialDual->update(1., *m_DataMng->m_NewDual, 0.);
        m_TrialDual->update(static_cast<Real>(1), *m_DataMng->m_NewSteepestDescent, 1.);
        m_Bounds->computeProjectedGradient(*m_TrialDual,
                                           *m_DualLowerBound,
                                           *m_DualUpperBound,
                                           *m_DataMng->m_NewSteepestDescent);

        Real scale_parameter = this->computeScaling();
        if(std::isfinite(scale_parameter) == false)
        {
            break;
        }

        m_DataMng->m_NewTrialStep->update(1., *m_DataMng->m_NewSteepestDescent, 0.);
        m_DataMng->m_NewTrialStep->update(scale_parameter, *m_DataMng->m_OldTrialStep, 1.);
        m_DataMng->storeCurrentState();
        this->step(objective_);

        dotk::DOTk_DualSolverCCSA::updateIterationCount();
        if(this->stoppingCriteriaSatisfied() == true)
        {
            break;
        }
    }

    if(dotk::DOTk_DualSolverCCSA::getIterationCount() >= max_num_iterations)
    {
        dotk::DOTk_DualSolverCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::MAX_NUMBER_ITERATIONS);
    }
    solution_->update(1., *m_DataMng->m_NewDual, 0.);
}

void DOTk_DualSolverNLCG::step(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_)
{
    Real alpha = 1e-4;
    std::vector<Real> step_values(2, 0.);
    // objective_function_values[0] = current value;
    // objective_function_values[1] = old trial value;
    // objective_function_values[2] = new trial value
    std::vector<Real> objective_function_values(3, 0.);
    objective_function_values[0] = m_DataMng->m_NewObjectiveFunctionValue;

    Real norm_trial_step = m_DataMng->m_NewTrialStep->norm();
    step_values[1] = std::min(static_cast<Real>(1.),
                              static_cast<Real>(100.) / (static_cast<Real>(1.) + norm_trial_step));

    m_TrialDual->update(1., *m_DataMng->m_NewDual, 0.);
    m_TrialDual->update(step_values[1], *m_DataMng->m_NewTrialStep, 1.);
    m_Bounds->project(*m_DualLowerBound, *m_DualUpperBound, *m_TrialDual);
    m_Bounds->computeProjectedStep(*m_TrialDual, *m_DataMng->m_NewDual, *m_ProjectedStep);

    objective_function_values[2] = objective_->value(*m_TrialDual);

    Real initial_projected_step_dot_gradient = m_ProjectedStep->dot(*m_DataMng->m_NewGradient);
    Real goal_objective_function_value = objective_function_values[0]
            - alpha * step_values[1] * initial_projected_step_dot_gradient;

    this->resetLineSearchIterationCount();
    size_t max_num_line_search_iterations = this->getMaxNumLineSearchIterations();
    while(objective_function_values[2] > goal_objective_function_value)
    {
        step_values[0] = step_values[1];
        Real new_step = quadraticInterpolationModel(step_values,
                                                    objective_function_values,
                                                    initial_projected_step_dot_gradient);
        step_values[1] = new_step;

        m_TrialDual->update(1., *m_DataMng->m_NewDual, 0.);
        m_TrialDual->update(step_values[1], *m_DataMng->m_NewTrialStep, 1.);
        m_Bounds->project(*m_DualLowerBound, *m_DualUpperBound, *m_TrialDual);
        m_Bounds->computeProjectedStep(*m_TrialDual, *m_DataMng->m_NewDual, *m_ProjectedStep);

        objective_function_values[1] = objective_function_values[2];
        objective_function_values[2] = objective_->value(*m_TrialDual);
        if(this->getLineSearchIterationCount() >= max_num_line_search_iterations)
        {
            return;
        }
        goal_objective_function_value = objective_function_values[0]
                - alpha * step_values[1] * initial_projected_step_dot_gradient;
        this->updateLineSearchIterationCount();
    }
    m_DataMng->m_NewObjectiveFunctionValue = objective_function_values[2];
    m_DataMng->m_NewDual->update(1., *m_TrialDual, 0.);
}

Real DOTk_DualSolverNLCG::computeScaling()
{
    Real scale = 0;
    switch(getNonlinearCgType())
    {
        case dotk::types::FLETCHER_REEVES_NLCG:
        {
            scale = dotk::nlcg::fletcherReeves(*m_DataMng->m_NewSteepestDescent, *m_DataMng->m_OldSteepestDescent);
            break;
        }
        case dotk::types::POLAK_RIBIERE_NLCG:
        {
            scale = dotk::nlcg::polakRibiere(*m_DataMng->m_NewSteepestDescent, *m_DataMng->m_OldSteepestDescent);
            break;
        }
        case dotk::types::HESTENES_STIEFEL_NLCG:
        {
            scale = dotk::nlcg::hestenesStiefel(*m_DataMng->m_NewSteepestDescent,
                                                *m_DataMng->m_OldSteepestDescent,
                                                *m_DataMng->m_OldTrialStep);
            break;
        }
        case dotk::types::DAI_YUAN_NLCG:
        {
            scale = dotk::nlcg::daiYuan(*m_DataMng->m_NewSteepestDescent,
                                        *m_DataMng->m_OldSteepestDescent,
                                        *m_DataMng->m_OldTrialStep);
            break;
        }
        case dotk::types::LIU_STOREY_NLCG:
        {
            scale = dotk::nlcg::liuStorey(*m_DataMng->m_NewSteepestDescent,
                                          *m_DataMng->m_OldSteepestDescent,
                                          *m_DataMng->m_OldTrialStep);
            break;
        }
        case dotk::types::CONJUGATE_DESCENT_NLCG:
        {
            scale = dotk::nlcg::conjugateDescent(*m_DataMng->m_NewSteepestDescent,
                                                 *m_DataMng->m_OldSteepestDescent,
                                                 *m_DataMng->m_OldTrialStep);
            break;
        }
        case dotk::types::DANIELS_NLCG:
        case dotk::types::DAI_LIAO_NLCG:
        case dotk::types::UNDEFINED_NLCG:
        case dotk::types::HAGER_ZHANG_NLCG:
        case dotk::types::PERRY_SHANNO_NLCG:
        case dotk::types::DAI_YUAN_HYBRID_NLCG:
        default:
        {
            std::string msg(" NONLINEAR CG STEP TYPE WAS NOT DEFINED. **** \n");
            std::cerr << "\n **** ERROR IN: " << __FILE__ << ", LINE: "
                    << __LINE__ << ", MSG: "<< msg.c_str() << std::flush;
            break;
        }
    }

    return (scale);
}

void DOTk_DualSolverNLCG::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->getDualLowerBound().use_count() < 1)
    {
        m_DualLowerBound->fill(0.);
    }
    else
    {
        m_DualLowerBound->update(1., *primal_->getDualLowerBound(), 0.);
    }
    if(primal_->getDualUpperBound().use_count() < 1)
    {
        m_DualUpperBound->fill(std::numeric_limits<Real>::max());
    }
    else
    {
        m_DualUpperBound->update(1., *primal_->getDualUpperBound(), 0.);
    }
}

bool DOTk_DualSolverNLCG::stoppingCriteriaSatisfied()
{
    bool criteria_satisfied = false;
    Real norm_trial_step = m_ProjectedStep->norm();
    Real norm_gradient = m_DataMng->m_NewSteepestDescent->norm();

    if(norm_gradient < dotk::DOTk_DualSolverCCSA::getGradientTolerance())
    {
        criteria_satisfied = true;
        dotk::DOTk_DualSolverCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::GRADIENT_TOLERANCE);
    }
    else if(norm_trial_step < dotk::DOTk_DualSolverCCSA::getTrialStepTolerance())
    {
        criteria_satisfied = true;
        dotk::DOTk_DualSolverCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::STEP_TOLERANCE);
    }

    return (criteria_satisfied);
}

Real DOTk_DualSolverNLCG::quadraticInterpolationModel(const std::vector<Real> step_values_,
                                                      const std::vector<Real> objective_function_values_,
                                                      const Real & initial_projected_step_dot_gradient_)
{
    // Recall the following:
    //  objective_function_values[0] = current value;
    //  objective_function_values[1] = old trial value;
    //  objective_function_values[2] = new trial value
    Real value = 0;
    Real step_lower_bound = step_values_[1] * this->getLineSearchStepLowerBound();
    Real step_upper_bound = step_values_[1] * this->getLineSearchStepUpperBound();
    // Quadratic interpolation model
    value = -initial_projected_step_dot_gradient_
            / (static_cast<Real>(2) * step_values_[1]
                    * (objective_function_values_[2] - objective_function_values_[0]
                            - initial_projected_step_dot_gradient_));
    if(value < step_lower_bound)
    {
        value = step_lower_bound;
    }
    if(value > step_upper_bound)
    {
        value = step_upper_bound;
    }

    return (value);
}

}
