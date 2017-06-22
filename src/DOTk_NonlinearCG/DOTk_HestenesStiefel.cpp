/*
 * DOTk_HestenesStiefel.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_HestenesStiefel.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_HestenesStiefel::DOTk_HestenesStiefel() :
    dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::HESTENES_STIEFEL_NLCG)
{
}

DOTk_HestenesStiefel::~DOTk_HestenesStiefel()
{
}

Real DOTk_HestenesStiefel::computeScaleFactor(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                              const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                              const std::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real beta = (new_grad_->dot(*new_grad_) - new_grad_->dot(*old_grad_))
            / (new_grad_->dot(*dir_) - old_grad_->dot(*dir_));
    //beta = std::max(beta, std::numeric_limits<Real>::min());
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_HestenesStiefel::getDirection(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                        const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                        const std::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, dir_);
    dir_->update(static_cast<Real>(-1.0), *new_grad_, beta);
}

void DOTk_HestenesStiefel::direction(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real value = dotk::DOTk_DescentDirection::computeCosineAngle(mng_->getOldGradient(), mng_->getTrialStep());
    if(dotk::DOTk_DescentDirection::isTrialStepOrthogonalToSteepestDescent(value) == true)
    {
        dotk::DOTk_DescentDirection::steepestDescent(mng_->getNewGradient(), mng_->getTrialStep());
        return;
    }
    this->getDirection(mng_->getOldGradient(), mng_->getNewGradient(), mng_->getTrialStep());
}

}
