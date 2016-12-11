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

void DOTk_ProjectionAlongFeasibleDir::getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & feasible_dir_)
{
    m_TrialPrimal->update(1., *primal_, 0.);
    m_TrialPrimal->update(dotk::DOTk_BoundConstraint::getStepSize(), *feasible_dir_, 1.);
    dotk::DOTk_BoundConstraint::project(m_LowerBounds, m_UpperBounds, m_TrialPrimal);
    feasible_dir_->update(1., *m_TrialPrimal, 0.);
    feasible_dir_->update(static_cast<Real>(-1.0), *primal_, 1.);
}

void DOTk_ProjectionAlongFeasibleDir::constraint(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                                 const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_TrialPrimal->update(1., *mng_->getNewPrimal(), 0.);
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
        m_LowerBounds->update(1., *primal_->getControlLowerBound(), 0.);
    }
    else
    {
        std::perror("\n**** Error in DOTk_ProjectionAlongFeasibleDir::initialize. User did not define control lower bounds. ABORT. ****\n");
        std::abort();
    }
    if(primal_->getControlUpperBound().use_count() > 0)
    {
        m_UpperBounds->update(1., *primal_->getControlUpperBound(), 0.);
    }
    else
    {
        std::perror("\n**** Error in DOTk_ProjectionAlongFeasibleDir::initialize. User did not define control upper bounds. ABORT. ****\n");
        std::abort();
    }
}

}
