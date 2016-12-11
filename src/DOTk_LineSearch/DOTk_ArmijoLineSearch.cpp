/*
 * DOTk_ArmijoLineSearch.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_ArmijoLineSearch.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_ArmijoLineSearch::DOTk_ArmijoLineSearch(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_LineSearch(dotk::types::line_search_t::BACKTRACKING_ARMIJO),
        m_ArmijoRuleConstant(1e-4),
        m_TrialPrimal(vector_->clone())
{
}

DOTk_ArmijoLineSearch::~DOTk_ArmijoLineSearch()
{
}

Real DOTk_ArmijoLineSearch::getConstant() const
{
    return (m_ArmijoRuleConstant);
}

void DOTk_ArmijoLineSearch::setConstant(Real value_)
{
    m_ArmijoRuleConstant = value_;
}

void DOTk_ArmijoLineSearch::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real step = 1.;
    m_TrialPrimal->copy(*mng_->getOldPrimal());
    m_TrialPrimal->axpy(step, *mng_->getTrialStep());

    size_t itr = 1;
    Real new_objective_func_val = mng_->evaluateObjective(m_TrialPrimal);
    Real contraction_factor = dotk::DOTk_LineSearch::getContractionFactor();
    Real gradient_dot_step = mng_->getNewGradient()->dot(*mng_->getTrialStep());
    while(itr <= dotk::DOTk_LineSearch::getMaxNumLineSearchItr())
    {
        dotk::DOTk_LineSearch::setNumLineSearchItrDone(itr);
        Real delta_objective_func = mng_->getOldObjectiveFunctionValue()
                + this->getConstant() * step * gradient_dot_step;

        bool sufficient_decrease_condition_satisfied =
                new_objective_func_val < delta_objective_func ? true : false;
        bool is_step_lower_than_tolerance = step < dotk::DOTk_LineSearch::getStepStagnationTol() ? true : false;
        if(sufficient_decrease_condition_satisfied || is_step_lower_than_tolerance)
        {
            break;
        }

        step *= contraction_factor;
        m_TrialPrimal->copy(*mng_->getOldPrimal());
        m_TrialPrimal->axpy(step, *mng_->getTrialStep());
        new_objective_func_val = mng_->evaluateObjective(m_TrialPrimal);
        ++itr;
    }

    dotk::DOTk_LineSearch::setNewObjectiveFunctionValue(new_objective_func_val);
    mng_->getNewPrimal()->copy(*m_TrialPrimal);
    dotk::DOTk_LineSearch::setStepSize(step);
}

}
