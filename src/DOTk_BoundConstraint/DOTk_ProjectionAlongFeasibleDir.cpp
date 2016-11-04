/*
 * DOTk_ProjectionAlongFeasibleDir.cpp
 *
 *  Created on: Sep 19, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_ProjectionAlongFeasibleDir.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_ProjectionAlongFeasibleDir::DOTk_ProjectionAlongFeasibleDir(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_BoundConstraint(primal_, dotk::types::constraint_method_t::PROJECTION_ALONG_FEASIBLE_DIR),
        m_LowerBounds(primal_->control()->clone()),
        m_UpperBounds(primal_->control()->clone()),
        m_TrialPrimal(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_ProjectionAlongFeasibleDir::~DOTk_ProjectionAlongFeasibleDir()
{
}

void DOTk_ProjectionAlongFeasibleDir::getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                   const std::tr1::shared_ptr<dotk::vector<Real> > & feasible_dir_)
{
    m_TrialPrimal->copy(*primal_);
    m_TrialPrimal->axpy(dotk::DOTk_BoundConstraint::getStepSize(), *feasible_dir_);
    dotk::DOTk_BoundConstraint::project(m_LowerBounds, m_UpperBounds, m_TrialPrimal);
    feasible_dir_->copy(*m_TrialPrimal);
    feasible_dir_->axpy(static_cast<Real>(-1.0), *primal_);
}

void DOTk_ProjectionAlongFeasibleDir::constraint(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                                 const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_TrialPrimal->copy(*mng_->getNewPrimal());
    dotk::DOTk_BoundConstraint::computeScaledTrialStep(step_, mng_, m_TrialPrimal);
    this->getDirection(mng_->getNewPrimal(), mng_->getTrialStep());
    step_->step(mng_);
    Real new_objective_function_value = step_->getNewObjectiveFunctionValue();
    mng_->setNewObjectiveFunctionValue(new_objective_function_value);
}

void DOTk_ProjectionAlongFeasibleDir::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->getControlLowerBound().use_count() > 0)
    {
        m_LowerBounds->copy(*primal_->getControlLowerBound());
    }
    else
    {
        std::perror("\n**** Error in DOTk_ProjectionAlongFeasibleDir::initialize. User did not define control lower bounds. ABORT. ****\n");
        std::abort();
    }
    if(primal_->getControlUpperBound().use_count() > 0)
    {
        m_UpperBounds->copy(*primal_->getControlUpperBound());
    }
    else
    {
        std::perror("\n**** Error in DOTk_ProjectionAlongFeasibleDir::initialize. User did not define control upper bounds. ABORT. ****\n");
        std::abort();
    }
}

}
