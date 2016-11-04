/*
 * DOTk_BacktrackingCubicInterpolation.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_BacktrackingCubicInterpolation.hpp"

namespace dotk
{

DOTk_BacktrackingCubicInterpolation::DOTk_BacktrackingCubicInterpolation(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_) :
        dotk::DOTk_LineSearch(dotk::types::line_search_t::BACKTRACKING_CUBIC_INTRP),
        m_ArmijoRuleConstant(1e-4),
        m_TrialPrimal(vector_->clone())
{
}

DOTk_BacktrackingCubicInterpolation::~DOTk_BacktrackingCubicInterpolation()
{
}

Real DOTk_BacktrackingCubicInterpolation::getConstant() const
{
    return (m_ArmijoRuleConstant);
}

void DOTk_BacktrackingCubicInterpolation::setConstant(Real value_)
{
    m_ArmijoRuleConstant = value_;
}

void DOTk_BacktrackingCubicInterpolation::getBacktrackingCubicFit(const Real innr_gradient_trialStep_,
                                                                  const std::vector<Real> & objective_func_val_,
                                                                  std::vector<Real> & step_)
{
    Real point1 = objective_func_val_[2] - objective_func_val_[0] - step_[1] * innr_gradient_trialStep_;
    Real point2 = objective_func_val_[1] - objective_func_val_[0] - step_[0] * innr_gradient_trialStep_;
    Real point3 = static_cast<Real>(1.) / (step_[1] - step_[0]);
    // find cubic unique minimum
    Real a = point3 * ((point1 / (step_[1] * step_[1])) - (point2 / (step_[0] * step_[0])));
    Real b = point3 * ((point2 * step_[1] / (step_[0] * step_[0])) - (point1 * step_[0] / (step_[1] * step_[1])));
    Real c = b * b - static_cast<Real>(3.) * a * innr_gradient_trialStep_;
    step_[2] = a != 0 ?
    // cubic equation has unique minimum
            step_[2] = (-b + std::sqrt(c)) / (static_cast<Real>(3.) * a):
            // cubic equation is a quadratic
            step_[2] = -innr_gradient_trialStep_ / (static_cast<Real>(2.) * b);
}

void DOTk_BacktrackingCubicInterpolation::checkBacktrackingStep(std::vector<Real> & step_)
{
    const Real gamma = 0.1;
    Real rho = dotk::DOTk_LineSearch::getContractionFactor();
    if(step_[2] > rho * step_[1])
    {
        step_[2] = rho * step_[1];
    }
    else if(step_[2] < gamma * step_[1])
    {
        step_[2] = gamma * step_[1];
    }
    if(std::isnan(step_[2]))
    {
        step_[2] = gamma * step_[1];
    }
}

void DOTk_BacktrackingCubicInterpolation::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real initial_step = 1;
    m_TrialPrimal->copy(*mng_->getOldPrimal());
    m_TrialPrimal->axpy(initial_step, *mng_->getTrialStep());

    Real new_objective_func_val = mng_->evaluateObjective(m_TrialPrimal);
    // objective_fun_val[0] = original, objective_fun_val[1] = old, objective_fun_val[2] = current
    std::vector<Real> objective_fun_val(3, 0.);
    objective_fun_val[0] = mng_->getOldObjectiveFunctionValue();
    objective_fun_val[2] = new_objective_func_val;
    // step[0] = old, step[1] = current, step[2] = new
    std::vector<Real> steps(3, 0.);
    steps[2] = initial_step;
    steps[1] = steps[2];

    size_t itr = 1;
    Real gradient_dot_step = mng_->getNewGradient()->dot(*mng_->getTrialStep());
    while(itr <= dotk::DOTk_LineSearch::getMaxNumLineSearchItr())
    {
        dotk::DOTk_LineSearch::setNumLineSearchItrDone(itr);
        Real sufficient_decrease_condition = objective_fun_val[0] + this->getConstant() * steps[1] * gradient_dot_step;
        bool sufficient_decrease_condition_satisfied =
                objective_fun_val[2] < sufficient_decrease_condition ? true : false;
        bool step_is_less_than_tolerance = steps[2] < dotk::DOTk_LineSearch::getStepStagnationTol() ? true : false;
        if(sufficient_decrease_condition_satisfied || step_is_less_than_tolerance)
        {
            break;
        }
        steps[0] = steps[1];
        steps[1] = steps[2];
        if(itr == 1)
        {
            // first backtrack: do a quadratic fit
            steps[2] = -gradient_dot_step
                    / (static_cast<Real>(2.) * (objective_fun_val[2] - objective_fun_val[0] - gradient_dot_step));
        }
        else
        {
            this->getBacktrackingCubicFit(gradient_dot_step, objective_fun_val, steps);
        }
        this->checkBacktrackingStep(steps);
        m_TrialPrimal->copy(*mng_->getOldPrimal());
        m_TrialPrimal->axpy(steps[2], *mng_->getTrialStep());

        new_objective_func_val = mng_->evaluateObjective(m_TrialPrimal);
        objective_fun_val[1] = objective_fun_val[2];
        objective_fun_val[2] = new_objective_func_val;
        ++itr;
    }
    this->setNewObjectiveFunctionValue(new_objective_func_val);
    mng_->getNewPrimal()->copy(*m_TrialPrimal);
    dotk::DOTk_LineSearch::setStepSize(steps[2]);
}

}
