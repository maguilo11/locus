/*
 * DOTk_DaiYuanHybrid.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_DaiYuanHybrid.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_DaiYuanHybrid::DOTk_DaiYuanHybrid() :
        dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::DAI_YUAN_HYBRID_NLCG),
        mWolfeConstant(1. / 3.)
{
}

DOTk_DaiYuanHybrid::~DOTk_DaiYuanHybrid()
{
}

Real DOTk_DaiYuanHybrid::getWolfeConstant() const
{
    return (mWolfeConstant);
}

void DOTk_DaiYuanHybrid::setWolfeConstant(Real value_)
{
    mWolfeConstant = value_;
}
Real DOTk_DaiYuanHybrid::computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                            const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                            const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real new_grad_dot_new_grad = new_grad_->dot(*new_grad_);
    Real new_grad_dot_old_grad = new_grad_->dot(*old_grad_);
    Real dir_dot_new_grad = dir_->dot(*new_grad_);
    Real dir_dot_old_grad = dir_->dot(*old_grad_);
    Real beta_hestenes_stiefel = (new_grad_dot_new_grad - new_grad_dot_old_grad)
                                 / (dir_dot_new_grad - dir_dot_old_grad);
    Real beta_dai_yuan = new_grad_dot_new_grad / (dir_dot_new_grad - dir_dot_old_grad);
    Real hybrid_scale_factor = -((static_cast<Real>(1.) - this->getWolfeConstant())
                                 / (static_cast<Real>(1.) + this->getWolfeConstant()));
    Real beta_scaled_dai_yuan = hybrid_scale_factor * beta_dai_yuan;
    Real beta = std::max(beta_scaled_dai_yuan, std::min(beta_hestenes_stiefel, beta_dai_yuan));
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_DaiYuanHybrid::getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                      const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                      const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, dir_);
    dir_->scale(beta);
    dir_->axpy(static_cast<Real>(-1.0), *new_grad_);
}

void DOTk_DaiYuanHybrid::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
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
