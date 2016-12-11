/*
 * DOTk_GoldenSectionLineSearch.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_GoldenSectionLineSearch.hpp"

namespace dotk
{

DOTk_GoldenSectionLineSearch::DOTk_GoldenSectionLineSearch(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_LineSearch(dotk::types::line_search_t::GOLDENSECTION),
        m_Step(4, 0.),
        m_ObjectiveFuncVal(4, 0.),
        m_TrialPrimal(vector_->clone())
{
}

DOTk_GoldenSectionLineSearch::~DOTk_GoldenSectionLineSearch()
{
}

void DOTk_GoldenSectionLineSearch::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    /* step[0] = step_lower_bound (a_LB)
     step[1] = step_upper_bound (a_UB)
     step[2] = kappa_lower_bound (k_LB)
     step[3] = kappa_lower_bound (k_UB) */
    m_Step.assign(m_Step.size(), 0.);
    /* objective_func_val_[0] = f(x + a_LB*trial_step)
     objective_func_val_[1] = f(x + a_UB*trial_step)
     objective_func_val_[2] = f(x + k_LB*trial_step)
     objective_func_val_[3] = f(x + k_UB*trial_step) */
    m_ObjectiveFuncVal.assign(m_Step.size(), 0.);
    // f(x + a_LB*trial_step)
    m_Step[0] = std::numeric_limits<Real>::min();
    m_ObjectiveFuncVal[0] = mng_->getOldObjectiveFunctionValue();
    // f(x + a_UB*trial_step)
    m_Step[1] = static_cast<Real>(1.);
    m_ObjectiveFuncVal[1] = mng_->getNewObjectiveFunctionValue();

    size_t itr = 1;
    const Real tau = (static_cast<Real>(3.) - std::sqrt(static_cast<Real>(5.))) / static_cast<Real>(2.);
    while(itr <= dotk::DOTk_LineSearch::getMaxNumLineSearchItr())
    {
        dotk::DOTk_LineSearch::setNumLineSearchItrDone(itr);

        // f(x + k_LB*trial_step)
        m_Step[2] = m_Step[0] + tau * (m_Step[1] - m_Step[0]);
        m_TrialPrimal->copy(*mng_->getOldPrimal());
        m_TrialPrimal->axpy(m_Step[2], *mng_->getTrialStep());
        m_ObjectiveFuncVal[2] = mng_->evaluateObjective(m_TrialPrimal);

        // f(primal + k_UB*trial_step)
        m_Step[3] = m_Step[1] - tau * (m_Step[1] - m_Step[0]);
        m_TrialPrimal->copy(*mng_->getOldPrimal());
        m_TrialPrimal->axpy(m_Step[3], *mng_->getTrialStep());
        m_ObjectiveFuncVal[3] = mng_->evaluateObjective(m_TrialPrimal);

        this->checkGoldenSectionStep();
        Real newStep_minus_oldStepd = m_Step[1] - m_Step[0];
        bool has_step_search_converged =
                newStep_minus_oldStepd < dotk::DOTk_LineSearch::getStepStagnationTol() ? true: false;
        if(has_step_search_converged)
        {
            break;
        }
        ++itr;
    }

    // Select the optimal step length
    Real step = m_ObjectiveFuncVal[2] < m_ObjectiveFuncVal[3] ? m_Step[2] : m_Step[3];
    // update state vector
    m_TrialPrimal->copy(*mng_->getOldPrimal());
    m_TrialPrimal->axpy(step, *mng_->getTrialStep());

    // Save the optimal state
    Real new_objective_func_value =
            m_ObjectiveFuncVal[2] < m_ObjectiveFuncVal[3] ? m_ObjectiveFuncVal[2] : m_ObjectiveFuncVal[3];
    dotk::DOTk_LineSearch::setNewObjectiveFunctionValue(new_objective_func_value);
    mng_->getNewPrimal()->copy(*m_TrialPrimal);
    mng_->getTrialStep()->copy(*mng_->getTrialStep());
    dotk::DOTk_LineSearch::setStepSize(step);
}

void DOTk_GoldenSectionLineSearch::checkGoldenSectionStep()
{
    if(m_ObjectiveFuncVal[2] < m_ObjectiveFuncVal[3])
    {
        // f(x + k_LB*s) < f(x + k_UB*s)
        if(m_ObjectiveFuncVal[0] <= m_ObjectiveFuncVal[2])
        {
            // f(x + a_UB*s) <= f(x + k_LB*s)
            m_Step[1] = m_Step[2];
            m_ObjectiveFuncVal[1] = m_ObjectiveFuncVal[2];
        }
        else
        {
            // f(x + a_UB*s) > f(x + k_UB*s)
            m_Step[1] = m_Step[3];
            m_ObjectiveFuncVal[1] = m_ObjectiveFuncVal[3];
        }
    }
    else if(m_ObjectiveFuncVal[2] > m_ObjectiveFuncVal[3])
    {
        // f(x + k_LB*s) > f(x + k_UB*s)
        if(m_ObjectiveFuncVal[3] >= m_ObjectiveFuncVal[1])
        {
            // f(x + k_UB*s) >= f(x + a_UB*s)
            m_Step[0] = m_Step[3];
            m_ObjectiveFuncVal[0] = m_ObjectiveFuncVal[3];
        }
        else
        {
            // f(x + k_UB*s) < f(x + a_UB*s)
            m_Step[0] = m_Step[2];
            m_ObjectiveFuncVal[0] = m_ObjectiveFuncVal[2];
        }
    }
    else if(m_ObjectiveFuncVal[2] == m_ObjectiveFuncVal[3])
    {
        // f(x + k_LB*s) == f(x + k_UB*s)
        m_Step[0] = m_Step[2];
        m_ObjectiveFuncVal[0] = m_ObjectiveFuncVal[2];
        m_Step[1] = m_Step[3];
        m_ObjectiveFuncVal[1] = m_ObjectiveFuncVal[3];
    }
}

}
