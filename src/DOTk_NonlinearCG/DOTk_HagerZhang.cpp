/*
 * DOTk_HagerZhang.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_HagerZhang.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_HagerZhang::DOTk_HagerZhang() :
        dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::HAGER_ZHANG_NLCG),
        mLowerBoundLimit(0.1)
{
}

DOTk_HagerZhang::~DOTk_HagerZhang()
{
}

void DOTk_HagerZhang::setLowerBoundLimit(Real value_)
{
    mLowerBoundLimit = value_;
}

Real DOTk_HagerZhang::getLowerBoundLimit() const
{
    return (mLowerBoundLimit);
}

Real DOTk_HagerZhang::computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                         const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                         const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real dir_dot_new_grad = dir_->dot(*new_grad_);
    Real dir_dot_old_grad = dir_->dot(*old_grad_);
    Real norm_grad_old = old_grad_->norm();
    Real old_grad_dot_old_grad = norm_grad_old * norm_grad_old;
    Real norm_new_grad = new_grad_->norm();
    Real new_grad_dot_new_grad = norm_new_grad * norm_new_grad;
    Real old_grad_dot_new_grad = old_grad_->dot(*new_grad_);

    Real one_over_dir_dot_dgrad = static_cast<Real>(1.0) / (dir_dot_new_grad - dir_dot_old_grad);
    Real dgrad_dot_new_grad = new_grad_dot_new_grad - old_grad_dot_new_grad;
    Real dgrad_dot_dgrad = new_grad_dot_new_grad - old_grad_dot_new_grad - old_grad_dot_new_grad
                           + old_grad_dot_old_grad;
    Real scaling = static_cast<Real>(2.0) * dgrad_dot_dgrad * one_over_dir_dot_dgrad;
    Real beta = one_over_dir_dot_dgrad * (dgrad_dot_new_grad - scaling * dir_dot_new_grad);
    // check lower bound on scale factor
    Real norm_dir = dir_->norm();
    Real lower_bound = static_cast<Real>(-1.0)
                       / (norm_dir * std::min(norm_grad_old, this->getLowerBoundLimit()));
    beta = std::max(beta, lower_bound);
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_HagerZhang::getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, dir_);
    dir_->update(static_cast<Real>(-1.0), *new_grad_, beta);
}

void DOTk_HagerZhang::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
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
