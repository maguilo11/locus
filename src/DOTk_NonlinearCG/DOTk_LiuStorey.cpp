/*
 * DOTk_LiuStorey.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_LiuStorey.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LiuStorey::DOTk_LiuStorey() :
        dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::LIU_STOREY_NLCG)
{
}

DOTk_LiuStorey::~DOTk_LiuStorey()
{
}

Real DOTk_LiuStorey::computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                        const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                        const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    /// Do I need the negative sign?
    Real new_grad_dot_new_grad = new_grad_->dot(*new_grad_);
    Real new_grad_dot_old_grad = new_grad_->dot(*old_grad_);
    Real dir_dot_old_grad = dir_->dot(*old_grad_);
    Real beta = -(new_grad_dot_new_grad - new_grad_dot_old_grad) / dir_dot_old_grad;
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_LiuStorey::getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, dir_);
    dir_->update(static_cast<Real>(-1.0), *new_grad_, beta);
}

void DOTk_LiuStorey::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
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
