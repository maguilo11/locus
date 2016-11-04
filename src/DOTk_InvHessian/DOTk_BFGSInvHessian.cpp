/*
 * DOTk_BFGSInvHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_BFGSInvHessian.hpp"

namespace dotk
{

DOTk_BFGSInvHessian::DOTk_BFGSInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_) :
        dotk::DOTk_SecondOrderOperator(),
        mDeltaPrimal(vector_->clone()),
        mDeltaGradient(vector_->clone()),
        m_InvHessTimesVec(vector_->clone())
{
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::BFGS_INV_HESS);
}

DOTk_BFGSInvHessian::~DOTk_BFGSInvHessian()
{
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_BFGSInvHessian::getDeltaGrad() const
{
    return (mDeltaGradient);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_BFGSInvHessian::getDeltaPrimal() const
{
    return (mDeltaPrimal);
}

void DOTk_BFGSInvHessian::getInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                        const std::tr1::shared_ptr<dotk::vector<Real> > & inv_hess_times_vector_)
{
    inv_hess_times_vector_->copy(*vector_);
    Real dprimal_dot_vec = mDeltaPrimal->dot(*vector_);
    Real value = std::fabs(dprimal_dot_vec - static_cast<Real>(0.0));
    bool zero_dprimal_dot_vec = value <= std::numeric_limits<Real>::min() ? true: false;
    if((zero_dprimal_dot_vec == true) || (dotk::DOTk_SecondOrderOperator::getNumOptimizationItrDone() == 1))
    {
        return;
    }

    // get Mv_new = (I - rho*s*y) * (I - rho*y*s)*vector_ + rho*s*s^t*vector_
    Real rho = static_cast<Real>(1.0) / mDeltaGradient->dot(*mDeltaPrimal);
    Real alpha = rho * dprimal_dot_vec;
    // compute rho*s*s^t*vector_
    inv_hess_times_vector_->copy(*mDeltaPrimal);
    inv_hess_times_vector_->scale(alpha);
    // compute gamma*I*(vector_ - alpha*y)
    m_InvHessTimesVec->copy(*vector_);
    m_InvHessTimesVec->axpy(-alpha, *mDeltaGradient);
    // compute beta = rho *dotk::scalar::dot(Yvec,(vector_ - alpha*y))
    Real beta = rho * mDeltaGradient->dot(*m_InvHessTimesVec);
    // compute (I - rho*s*y)^t * (I - rho*s*y)*vector_
    inv_hess_times_vector_->axpy(-beta, *mDeltaPrimal);
    inv_hess_times_vector_->axpy(static_cast<Real>(1.0), *m_InvHessTimesVec);

    Real norm_invHess_times_vec = inv_hess_times_vector_->norm();
    bool negative_curvature_detected = norm_invHess_times_vec < std::numeric_limits<Real>::min() ? true: false;
    if(negative_curvature_detected == true)
    {
        inv_hess_times_vector_->copy(*vector_);
    }
}

void DOTk_BFGSInvHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), mDeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(),
                                                         mng_->getOldGradient(),
                                                         mDeltaGradient);
    this->getInvHessian(vector_, matrix_times_vector_);
}

}
