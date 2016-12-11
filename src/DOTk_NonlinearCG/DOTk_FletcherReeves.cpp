/*
 * DOTk_FletcherReeves.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_FletcherReeves.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_FletcherReeves::DOTk_FletcherReeves() :
        dotk::DOTk_DescentDirection(dotk::types::nonlinearcg_t::FLETCHER_REEVES_NLCG)
{
}

DOTk_FletcherReeves::~DOTk_FletcherReeves()
{
}

Real DOTk_FletcherReeves::computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_)
{
    Real beta = new_grad_->dot(*new_grad_) /
            old_grad_->dot(*old_grad_);
    dotk::DOTk_DescentDirection::setScaleFactor(beta);
    return (beta);
}

void DOTk_FletcherReeves::getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                                       const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                       const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{

    Real beta = this->computeScaleFactor(old_grad_, new_grad_);
    dir_->scale(beta);
    dir_->axpy(static_cast<Real>(-1.0), *new_grad_);
}

void DOTk_FletcherReeves::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real value = dotk::DOTk_DescentDirection::computeCosineAngle(mng_->getOldGradient(), mng_->getTrialStep());
    if( dotk::DOTk_DescentDirection::isTrialStepOrthogonalToSteepestDescent(value) == true )
    {
        dotk::DOTk_DescentDirection::steepestDescent(mng_->getNewGradient(), mng_->getTrialStep());
        return;
    }
    this->getDirection(mng_->getOldGradient(), mng_->getNewGradient(), mng_->getTrialStep());
}

}
