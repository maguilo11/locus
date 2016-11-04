/*
 * DOTk_DaiLiao.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_DaiLiao.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_DaiLiao::DOTk_DaiLiao() :
        dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::DAI_LIAO_NLCG),
        mConstant(0.1)
{
}

DOTk_DaiLiao::~DOTk_DaiLiao()
{
}

Real DOTk_DaiLiao::getConstant() const
{
    return (mConstant);
}

void DOTk_DaiLiao::setConstant(Real value_)
{
    mConstant = value_;
}

Real DOTk_DaiLiao::computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
{
    Real innr_direction_newGrad = dir_->dot(*new_grad_);
    Real innr_direction_oldGrad = dir_->dot(*old_grad_);
    Real innr_newPrimal_newGrad = new_primal_->dot(*new_grad_);
    Real innr_oldPrimal_newGrad = old_primal_->dot(*new_grad_);
    Real innr_newGrad_newGrad = new_grad_->dot(*new_grad_);
    Real innr_oldGrad_newGrad = old_grad_->dot(*new_grad_);

    Real one_over_innrDirectionDeltaGrad = static_cast<Real>(1.0) / (innr_direction_newGrad - innr_direction_oldGrad);
    Real innr_deltaGrad_newGrad = innr_newGrad_newGrad - innr_oldGrad_newGrad;
    Real innr_deltaPrimal_newGrad = innr_newPrimal_newGrad - innr_oldPrimal_newGrad;
    Real beta = one_over_innrDirectionDeltaGrad
                * (innr_deltaGrad_newGrad - (this->getConstant() * innr_deltaPrimal_newGrad));
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_DaiLiao::getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & dir_)
{
    Real beta = this->computeScaleFactor(old_grad_, new_grad_, old_primal_, new_primal_, dir_);
    dir_->scale(beta);
    dir_->axpy(static_cast<Real>(-1.0), *new_grad_);
}

void DOTk_DaiLiao::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
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
