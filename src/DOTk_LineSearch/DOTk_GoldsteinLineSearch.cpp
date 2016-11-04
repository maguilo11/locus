/*
 * DOTk_GoldsteinLineSearch.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_GoldsteinLineSearch.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_GoldsteinLineSearch::DOTk_GoldsteinLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_) :
        dotk::DOTk_LineSearch(dotk::types::line_search_t::BACKTRACKING_GOLDSTEIN),
        m_GoldsteinConstant(0.9),
        m_TrialPrimal(vector_->clone())
{
}

DOTk_GoldsteinLineSearch::~DOTk_GoldsteinLineSearch()
{
}

Real DOTk_GoldsteinLineSearch::getConstant() const
{
    return (m_GoldsteinConstant);
}

void DOTk_GoldsteinLineSearch::setConstant(Real value_)
{
    m_GoldsteinConstant = value_;
}

void DOTk_GoldsteinLineSearch::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real step = 1;
    m_TrialPrimal->copy(*mng_->getOldPrimal());
    m_TrialPrimal->axpy(step, *mng_->getTrialStep());

    size_t itr = 1;
    Real gradient_dot_step = mng_->getNewGradient()->dot(*mng_->getTrialStep());
    Real new_objective_func_val = mng_->evaluateObjective(m_TrialPrimal);
    Real contraction_factor = dotk::DOTk_LineSearch::getContractionFactor();
    while(itr <= dotk::DOTk_LineSearch::getMaxNumLineSearchItr())
    {
        dotk::DOTk_LineSearch::setNumLineSearchItrDone(itr);
        Real sufficient_decrease = mng_->getOldObjectiveFunctionValue()
                + (step * this->getConstant() * gradient_dot_step);

        bool sufficient_decrease_condition_satisfied = sufficient_decrease <= new_objective_func_val ? true: false;
        Real control_step_from_below = mng_->getOldObjectiveFunctionValue()
                + (step * contraction_factor * gradient_dot_step);
        bool control_step_from_below_condition_satisfied =
                new_objective_func_val <= control_step_from_below ? true: false;
        if(control_step_from_below_condition_satisfied && sufficient_decrease_condition_satisfied)
        {
            break;
        }

        bool is_step_lower_than_tolerance = step < dotk::DOTk_LineSearch::getStepStagnationTol() ? true : false;
        if(is_step_lower_than_tolerance)
        {
            break;
        }
        step *= contraction_factor;
        m_TrialPrimal->copy(*mng_->getOldPrimal());
        m_TrialPrimal->axpy(step, *mng_->getTrialStep());
        new_objective_func_val = mng_->evaluateObjective(m_TrialPrimal);
        ++itr;
    }

    this->setNewObjectiveFunctionValue(new_objective_func_val);
    mng_->getNewPrimal()->copy(*m_TrialPrimal);
    dotk::DOTk_LineSearch::setStepSize(step);
}

}
