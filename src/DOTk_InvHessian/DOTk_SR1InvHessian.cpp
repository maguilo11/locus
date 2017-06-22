/*
 * DOTk_SR1InvHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_SR1InvHessian.hpp"

namespace dotk
{

DOTk_SR1InvHessian::DOTk_SR1InvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_SecondOrderOperator(),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_InvHessTimesVec(vector_->clone())
{
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::SR1_INV_HESS);
}

DOTk_SR1InvHessian::~DOTk_SR1InvHessian()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_SR1InvHessian::getDeltaGrad() const
{
    return (m_DeltaGradient);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_SR1InvHessian::getDeltaPrimal() const
{
    return (m_DeltaPrimal);
}

void DOTk_SR1InvHessian::getInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                       const std::shared_ptr<dotk::Vector<Real> > & inv_hess_times_vector_)
{
    inv_hess_times_vector_->update(1., *vector_, 0.);
    Real dprimal_dot_dgrad = m_DeltaPrimal->dot(*m_DeltaGradient);
    Real value = std::fabs(dprimal_dot_dgrad - static_cast<Real>(0.0));
    bool zero_dprimal_dot_vec = value <= std::numeric_limits<Real>::min() ? true : false;
    if(zero_dprimal_dot_vec == true)
    {
        return;
    }

    Real barzilai_borwein_step = dotk::DOTk_SecondOrderOperator::getBarzilaiBorweinStep(m_DeltaPrimal, m_DeltaGradient);
    Real barzilai_borwein_step_lower_bound = dotk::DOTk_SecondOrderOperator::getDiagonalScaleFactor() /
            dotk::DOTk_SecondOrderOperator::getLowerBoundOnDiagonalScaleFactor();
    barzilai_borwein_step = std::min(barzilai_borwein_step, barzilai_borwein_step_lower_bound);
    inv_hess_times_vector_->scale(barzilai_borwein_step);
    // use secant equation to approximate M_k * s_k (i.e. M_k+1 y_k = M_k s_k ) assuming B_k = I
    m_InvHessTimesVec->update(1., *m_DeltaPrimal, 0.);
    Real alpha = dprimal_dot_dgrad - m_DeltaGradient->dot(*m_DeltaGradient);
    // get alpha = (dot(s,vector_) - dot(invHessTimesVec,vector_) ) / ( ->dot(s,y) - ->dot(invHessTimesVec,y)
    Real kappa = (m_DeltaPrimal->dot(*vector_) - m_InvHessTimesVec->dot(*vector_)) / alpha;
    // get invHessian_times_vector_(k+1) = vector_(k) + dot(kappa*s,vector_)(k) - dot(kappa*invHessTimesVec,vector_)(k), k=iteration counter
    inv_hess_times_vector_->update(kappa, *m_DeltaPrimal, 1.);
    inv_hess_times_vector_->update(-kappa, *m_InvHessTimesVec, 1.);

    Real norm_invHess_times_vec = inv_hess_times_vector_->norm();
    bool negative_curvature_detected = norm_invHess_times_vec < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_detected == true)
    {
        inv_hess_times_vector_->update(1., *vector_, 0.);
    }
}

void DOTk_SR1InvHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                               const std::shared_ptr<dotk::Vector<Real> > & vector_,
                               const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    this->getInvHessian(vector_, matrix_times_vector_);
}

}
