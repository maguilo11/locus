/*
 * DOTk_DaiYuan.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_DaiYuan.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_DaiYuan::DOTk_DaiYuan() :
    dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::DAI_YUAN_NLCG)
{
}

DOTk_DaiYuan::~DOTk_DaiYuan()
{
}

Real DOTk_DaiYuan::computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
{
    Real new_grad_dot_new_grad = new_grad_->dot(*new_grad_);
    Real dir_dot_new_grad = dir_->dot(*new_grad_);
    Real dir_dot_old_grad = dir_->dot(*old_grad_);
    Real beta = new_grad_dot_new_grad / (dir_dot_new_grad - dir_dot_old_grad);
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_DaiYuan::getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, dir_);
    dir_->scale(beta);
    dir_->axpy(static_cast<Real>(-1.0), *new_grad_);
}

void DOTk_DaiYuan::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
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