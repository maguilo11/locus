/*
 * DOTk_HagerZhangLineSearch.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "vector.hpp"
#include "DOTk_HagerZhangLineSearch.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_HagerZhangLineSearch::DOTk_HagerZhangLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_) :
        dotk::DOTk_LineSearch(dotk::types::line_search_t::LINE_SEARCH_HAGER_ZHANG),
        m_MaxShrinkIntervalIterations(5),
        m_ArmijoRuleConstant(0.1),
        m_CurvatureConstant(0.9),
        m_IntervalUpdateParameter(1e-6),
        m_BisectionUpdateParameter(0.5),
        m_ObjectiveFunctionErrorEstimateParameter(2 / 3),
        m_StepInterval()
{
    m_StepInterval.insert(std::pair<dotk::types::bound_t, Real>(dotk::types::LOWER_BOUND, 0.));
    m_StepInterval.insert(std::pair<dotk::types::bound_t, Real>(dotk::types::UPPER_BOUND, 0.));
}

DOTk_HagerZhangLineSearch::~DOTk_HagerZhangLineSearch()
{
}

void DOTk_HagerZhangLineSearch::setMaxShrinkIntervalIterations(size_t value_)
{
    m_MaxShrinkIntervalIterations = value_;
}

size_t DOTk_HagerZhangLineSearch::getMaxShrinkIntervalIterations() const
{
    return (m_MaxShrinkIntervalIterations);
}

void DOTk_HagerZhangLineSearch::setConstant(Real value_)
{
    m_ArmijoRuleConstant = value_;
}

Real DOTk_HagerZhangLineSearch::getConstant() const
{
    return (m_ArmijoRuleConstant);
}

void DOTk_HagerZhangLineSearch::setCurvatureConstant(Real value_)
{
    m_CurvatureConstant = value_;
}

Real DOTk_HagerZhangLineSearch::getCurvatureConstant() const
{
    return (m_CurvatureConstant);
}

void DOTk_HagerZhangLineSearch::setIntervalUpdateParameter(Real value_)
{
    m_IntervalUpdateParameter = value_;
}

Real DOTk_HagerZhangLineSearch::getIntervalUpdateParameter() const
{
    return (m_IntervalUpdateParameter);
}

void DOTk_HagerZhangLineSearch::setBisectionUpdateParameter(Real value_)
{
    m_BisectionUpdateParameter = value_;
}

Real DOTk_HagerZhangLineSearch::getBisectionUpdateParameter() const
{
    return (m_BisectionUpdateParameter);
}

void DOTk_HagerZhangLineSearch::setObjectiveFunctionErrorEstimateParameter(Real value_)
{
    m_ObjectiveFunctionErrorEstimateParameter = value_;
}

Real DOTk_HagerZhangLineSearch::getObjectiveFunctionErrorEstimateParameter() const
{
    return (m_ObjectiveFunctionErrorEstimateParameter);
}

void DOTk_HagerZhangLineSearch::setStepInterval(dotk::types::bound_t type_, Real value_)
{
    m_StepInterval.find(type_)->second = value_;
}

Real DOTk_HagerZhangLineSearch::getStepInterval(dotk::types::bound_t type_)
{
    return (m_StepInterval.find(type_)->second);
}

Real DOTk_HagerZhangLineSearch::secantStep(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                                           const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real step = 0.;
    primal_new_->copy(*primal_old_);
    primal_new_->axpy(this->getStepInterval(dotk::types::LOWER_BOUND), *trial_step_);
    mng_->computeGradient(primal_new_, gradient_new_);
    Real grad_dot_step_lb = gradient_new_->dot(*trial_step_);
    primal_new_->copy(*primal_old_);
    primal_new_->axpy(this->getStepInterval(dotk::types::UPPER_BOUND), *trial_step_);
    mng_->computeGradient(primal_new_, gradient_new_);
    Real grad_dot_step_ub = gradient_new_->dot(*trial_step_);
    step = (this->getStepInterval(dotk::types::LOWER_BOUND) * grad_dot_step_ub)
           - ((this->getStepInterval(dotk::types::UPPER_BOUND) * grad_dot_step_lb)
                   / grad_dot_step_ub) - grad_dot_step_lb;
    return (step);
}

void DOTk_HagerZhangLineSearch::doubleSecantStep(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                                                 const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                                                 const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                                                 const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                                                 const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real current_step_lower_bound = this->getStepInterval(dotk::types::LOWER_BOUND);
    Real current_step_upper_bound = this->getStepInterval(dotk::types::UPPER_BOUND);
    Real step = this->secantStep(trial_step_, primal_old_, primal_new_, gradient_new_, mng_);
    dotk::DOTk_LineSearch::setStepSize(step);
    this->updateInterval(step, trial_step_, primal_old_, primal_new_, gradient_new_, mng_);
    Real new_step_lower_bound = this->getStepInterval(dotk::types::LOWER_BOUND);
    Real new_step_upper_bound = this->getStepInterval(dotk::types::UPPER_BOUND);
    Real step_bar = 0.;
    Real diff_step_minus_new_step_upper_bound = step - new_step_upper_bound;
    bool is_step_close_to_new_upper_bound =
            diff_step_minus_new_step_upper_bound < std::numeric_limits<Real>::epsilon() ? true : false;
    if(is_step_close_to_new_upper_bound)
    {
        this->setStepInterval(dotk::types::LOWER_BOUND, current_step_upper_bound);
        this->setStepInterval(dotk::types::UPPER_BOUND, new_step_upper_bound);
        step_bar = this->secantStep(trial_step_, primal_old_, primal_new_, gradient_new_, mng_);
        dotk::DOTk_LineSearch::setStepSize(step_bar);
    }
    Real diff_step_minus_new_step_lower_bound = step - new_step_lower_bound;
    bool is_step_close_to_new_lower_bound =
            diff_step_minus_new_step_lower_bound < std::numeric_limits<Real>::epsilon() ? true : false;
    if(is_step_close_to_new_lower_bound)
    {
        this->setStepInterval(dotk::types::LOWER_BOUND, current_step_lower_bound);
        this->setStepInterval(dotk::types::UPPER_BOUND, new_step_lower_bound);
        step_bar = this->secantStep(trial_step_, primal_old_, primal_new_, gradient_new_, mng_);
        dotk::DOTk_LineSearch::setStepSize(step_bar);
    }
    if(is_step_close_to_new_upper_bound || is_step_close_to_new_lower_bound)
    {
        this->updateInterval(step_bar, trial_step_, primal_old_, primal_new_, gradient_new_, mng_);
    }
}

void DOTk_HagerZhangLineSearch::updateInterval(const Real & step_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                                               const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    bool is_step_smaller_than_lower_bound = step_ < this->getStepInterval(dotk::types::LOWER_BOUND) ? true : false;
    bool is_step_greater_than_upper_bound = step_ > this->getStepInterval(dotk::types::UPPER_BOUND) ? true : false;
    if(is_step_smaller_than_lower_bound || is_step_greater_than_upper_bound)
    {
        return;
    }
    primal_new_->copy(*primal_old_);
    primal_new_->axpy(step_, *trial_step_);
    Real new_fval = mng_->evaluateObjective(primal_new_);
    dotk::DOTk_LineSearch::setNewObjectiveFunctionValue(new_fval);
    mng_->computeGradient(primal_new_, gradient_new_);
    Real grad_dot_step = gradient_new_->dot(*trial_step_);
    if(grad_dot_step >= std::numeric_limits<Real>::min())
    {
        this->setStepInterval(dotk::types::LOWER_BOUND, this->getStepInterval(dotk::types::LOWER_BOUND));
        this->setStepInterval(dotk::types::UPPER_BOUND, step_);
        return;
    }
    Real error_estimate_in_fval = dotk::DOTk_LineSearch::getOldObjectiveFunctionValue()
                                  + this->getObjectiveFunctionErrorEstimateParameter();
    bool is_opposite_slope_condition1_ok = grad_dot_step < std::numeric_limits<Real>::min() ? true : false;
    bool is_opposite_slope_condition2_ok = new_fval <= error_estimate_in_fval ? true : false;
    if(is_opposite_slope_condition1_ok && is_opposite_slope_condition2_ok)
    {
        this->setStepInterval(dotk::types::LOWER_BOUND, step_);
        this->setStepInterval(dotk::types::UPPER_BOUND, this->getStepInterval(dotk::types::UPPER_BOUND));
        return;
    }
    bool is_opposite_slope_condition3_ok = new_fval > error_estimate_in_fval ? true : false;
    if(is_opposite_slope_condition1_ok && is_opposite_slope_condition3_ok)
    {
        shrinkInterval(step_, trial_step_, primal_old_, primal_new_, gradient_new_, mng_);
        return;
    }
}

void DOTk_HagerZhangLineSearch::shrinkInterval(const Real & step_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                                               const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real theta = this->getIntervalUpdateParameter();
    Real new_interval_lower_bound = this->getStepInterval(dotk::types::LOWER_BOUND);
    Real new_interval_upper_bound = step_;
    size_t iterations = 1;
    while(1)
    {
        Real secant_step = ((static_cast<Real>(1.0) - theta) * new_interval_lower_bound)
                + (theta * new_interval_upper_bound);
        primal_new_->copy(*primal_old_);
        primal_new_->axpy(secant_step, *trial_step_);
        Real new_fval = mng_->evaluateObjective(primal_new_);
        dotk::DOTk_LineSearch::setNewObjectiveFunctionValue(new_fval);
        mng_->computeGradient(primal_new_, gradient_new_);
        Real innr_gradient_trialStep = gradient_new_->dot(*trial_step_);
        if(innr_gradient_trialStep >= std::numeric_limits<Real>::min())
        {
            this->setStepInterval(dotk::types::LOWER_BOUND, new_interval_lower_bound);
            this->setStepInterval(dotk::types::UPPER_BOUND, secant_step);
            dotk::DOTk_LineSearch::setStepSize(secant_step);
            break;
        }
        bool is_opposite_slope_condition1_ok =
                innr_gradient_trialStep < std::numeric_limits<Real>::min() ? true : false;
        Real error_estimate_in_fval =
                dotk::DOTk_LineSearch::getOldObjectiveFunctionValue() + this->getObjectiveFunctionErrorEstimateParameter();
        bool is_opposite_slope_condition2_ok = new_fval <= error_estimate_in_fval ? true : false;
        if(is_opposite_slope_condition1_ok && is_opposite_slope_condition2_ok)
        {
            new_interval_lower_bound = secant_step;
        }
        bool is_opposite_slope_condition3_ok = new_fval > error_estimate_in_fval ? true : false;
        if(is_opposite_slope_condition1_ok && is_opposite_slope_condition3_ok)
        {
            new_interval_upper_bound = secant_step;
        }
        ++iterations;
    }
}

void DOTk_HagerZhangLineSearch::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    dotk::DOTk_LineSearch::setOldObjectiveFunctionValue(mng_->getOldObjectiveFunctionValue());

    dotk::DOTk_LineSearch::setStepSize(static_cast<Real>(1.0));
    this->setStepInterval(dotk::types::LOWER_BOUND, std::numeric_limits<Real>::min());
    this->setStepInterval(dotk::types::UPPER_BOUND, dotk::DOTk_LineSearch::getStepSize());
    mng_->getNewPrimal()->copy(*mng_->getOldPrimal());
    mng_->getNewPrimal()->axpy(dotk::DOTk_LineSearch::getStepSize(), *mng_->getTrialStep());
    Real new_objective_func_val = mng_->evaluateObjective(mng_->getNewPrimal());
    this->setNewObjectiveFunctionValue(new_objective_func_val);
    mng_->computeGradient(mng_->getNewPrimal(), mng_->getNewGradient());

    size_t itr = 1;
    Real grad_old_dot_step = mng_->getOldGradient()->dot(*mng_->getTrialStep());
    while(itr <= dotk::DOTk_LineSearch::getMaxNumLineSearchItr())
    {
        Real delta_objective = dotk::DOTk_LineSearch::getNewObjectiveFunctionValue()
                - dotk::DOTk_LineSearch::getOldObjectiveFunctionValue();
        bool armijo_check =
                delta_objective <= (this->getConstant() * this->getStepSize() * grad_old_dot_step) ? true : false;
        Real grad_new_dot_step = mng_->getNewGradient()->dot(*mng_->getTrialStep());
        bool curvature_check = grad_new_dot_step >= (this->getCurvatureConstant() * grad_old_dot_step) ? true : false;
        if(armijo_check && curvature_check)
        {
            break;
        }
        if(dotk::DOTk_LineSearch::getStepSize() < dotk::DOTk_LineSearch::getStepStagnationTol())
        {
            break;
        }
        this->doubleSecantStep(mng_->getTrialStep(),
                               mng_->getOldPrimal(),
                               mng_->getNewPrimal(),
                               mng_->getNewGradient(),
                               mng_);
        Real ubound_minus_lbound = this->getStepInterval(dotk::types::UPPER_BOUND)
                - this->getStepInterval(dotk::types::LOWER_BOUND);
        bool is_bisection_step_required =
                ubound_minus_lbound > (this->getBisectionUpdateParameter() * ubound_minus_lbound) ? true: false;
        if(is_bisection_step_required)
        {
            Real step = (this->getStepInterval(dotk::types::LOWER_BOUND)
                    + this->getStepInterval(dotk::types::UPPER_BOUND)) / static_cast<Real>(2.);
            this->updateInterval(step,
                                 mng_->getTrialStep(),
                                 mng_->getOldPrimal(),
                                 mng_->getNewPrimal(),
                                 mng_->getNewGradient(),
                                 mng_);
            dotk::DOTk_LineSearch::setStepSize(step);
        }
        ++itr;
    }

    dotk::DOTk_LineSearch::setStepSize(dotk::DOTk_LineSearch::getStepSize());
}

}
