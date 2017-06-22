/*
 * DOTk_FeasibleDirection.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_FeasibleDirection.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_FeasibleDirection::DOTk_FeasibleDirection(const std::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_BoundConstraint(primal_, dotk::types::constraint_method_t::FEASIBLE_DIR),
        m_LowerBounds(primal_->control()->clone()),
        m_UpperBounds(primal_->control()->clone()),
        m_TrialPrimal(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_FeasibleDirection::~DOTk_FeasibleDirection()
{
}

void DOTk_FeasibleDirection::getDirection(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                          const std::shared_ptr<dotk::Vector<Real> > & feasible_dir_)
{
    Real factor = 1.;
    m_TrialPrimal->update(1., *primal_, 0.);
    m_TrialPrimal->update(factor, *feasible_dir_, 1.);
    dotk::DOTk_BoundConstraint::computeActiveSet(m_LowerBounds, m_UpperBounds, m_TrialPrimal);

    size_t itr = 1;
    while(1)
    {
        dotk::DOTk_BoundConstraint::setNumFeasibleItr(itr);
        if(dotk::DOTk_BoundConstraint::isFeasible(m_LowerBounds, m_UpperBounds, m_TrialPrimal) == true)
        {
            feasible_dir_->scale(factor);
            break;
        }
        factor = factor * dotk::DOTk_BoundConstraint::getContractionStep();
        m_TrialPrimal->update(1., *primal_, 0.);
        m_TrialPrimal->update(factor, *feasible_dir_, 1.);
        ++itr;
    }
}

void DOTk_FeasibleDirection::constraint(const std::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                        const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    this->getDirection(mng_->getNewPrimal(), mng_->getTrialStep());
    step_->step(mng_);
    mng_->setNewObjectiveFunctionValue(step_->getNewObjectiveFunctionValue());
}

void DOTk_FeasibleDirection::initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->getControlLowerBound().use_count() > 0)
    {
        m_LowerBounds->update(1., *primal_->getControlLowerBound(), 0.);
    }
    else
    {
        std::perror("\n**** Error in DOTk_FeasibleDirection::initialize. User did not define control lower bounds. ABORT. ****\n");
        std::abort();
    }
    if(primal_->getControlUpperBound().use_count() > 0)
    {
        m_UpperBounds->update(1., *primal_->getControlUpperBound(), 0.);
    }
    else
    {
        std::perror("\n**** Error in DOTk_FeasibleDirection::initialize. User did not define control upper bounds. ABORT. ****\n");
        std::abort();
    }
}

}
