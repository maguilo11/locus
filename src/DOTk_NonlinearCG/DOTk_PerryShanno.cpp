/*
 * DOTk_PerryShanno.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_PerryShanno.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_PerryShanno::DOTk_PerryShanno() :
        dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::PERRY_SHANNO_NLCG),
        mAlpha(0.),
        mTheta(0.),
        mLowerBoundLimit(0.1)
{
}

DOTk_PerryShanno::~DOTk_PerryShanno()
{
}

void DOTk_PerryShanno::setAlphaScaleFactor(Real value_)
{
    mAlpha = value_;
}

Real DOTk_PerryShanno::getAlphaScaleFactor() const
{
    return (mAlpha);
}

void DOTk_PerryShanno::setThetaScaleFactor(Real value_)
{
    mTheta = value_;
}

Real DOTk_PerryShanno::getThetaScaleFactor() const
{
    return (mTheta);
}

void DOTk_PerryShanno::setLowerBoundLimit(Real value_)
{
    mLowerBoundLimit = value_;
}

Real DOTk_PerryShanno::getLowerBoundLimit() const
{
    return (mLowerBoundLimit);
}

Real DOTk_PerryShanno::computeAlphaScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
{
    Real dir_dot_new_grad = dir_->dot(*new_grad_);
    Real dir_dot_old_grad = dir_->dot(*old_grad_);
    Real alpha = dir_dot_new_grad / (dir_dot_new_grad - dir_dot_old_grad);
    this->setAlphaScaleFactor(alpha);
    return (alpha);
}

Real DOTk_PerryShanno::computeThetaScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                                               const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_)
{
    Real nwgrad_dot_nwprimal = new_grad_->dot(*new_primal_);
    Real odgrad_dot_nwprimal = old_grad_->dot(*new_primal_);
    Real nwgrad_dot_olprimal = new_grad_->dot(*old_primal_);
    Real odgrad_dot_odprimal = old_grad_->dot(*old_primal_);
    Real nwgrad_dot_nwgrad = new_grad_->dot(*new_grad_);
    Real odgrad_dot_nwgrad = old_grad_->dot(*new_grad_);
    Real odgrad_dot_odgrad = old_grad_->dot(*old_grad_);
    Real dgrad_dot_dgrad = nwgrad_dot_nwgrad - odgrad_dot_nwgrad - odgrad_dot_nwgrad + odgrad_dot_odgrad;
    Real dgrad_dot_dprimal = nwgrad_dot_nwprimal - odgrad_dot_nwprimal - nwgrad_dot_olprimal + odgrad_dot_odprimal;
    Real theta = dgrad_dot_dprimal / dgrad_dot_dgrad;
    this->setThetaScaleFactor(theta);
    return (theta);
}

Real DOTk_PerryShanno::computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                          const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                          const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
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

void DOTk_PerryShanno::getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, dir_);
    Real alpha = this->computeAlphaScaleFactor(old_grad_, new_grad_, dir_);
    Real theta = this->computeThetaScaleFactor(old_grad_, new_grad_, old_primal_, new_primal_);

    // compute Perry-Shannon direction
    dir_->scale(beta);
    dir_->axpy(static_cast<Real>(-1.0), *new_grad_);
    dir_->axpy(alpha, *new_grad_);
    dir_->axpy(-alpha, *old_grad_);
    dir_->scale(theta);
}

void DOTk_PerryShanno::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real value = dotk::DOTk_DescentDirection::computeCosineAngle(mng_->getOldGradient(), mng_->getTrialStep());
    if(dotk::DOTk_DescentDirection::isTrialStepOrthogonalToSteepestDescent(value) == true)
    {
        dotk::DOTk_DescentDirection::steepestDescent(mng_->getNewGradient(), mng_->getTrialStep());
        return;
    }
    this->getDirection(mng_->getOldGradient(),
                       mng_->getNewGradient(),
                       mng_->getOldPrimal(),
                       mng_->getNewPrimal(),
                       mng_->getTrialStep());
}

}
