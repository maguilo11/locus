/*
 * DOTk_SR1Hessian.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SR1Hessian.hpp"

namespace dotk
{

DOTk_SR1Hessian::DOTk_SR1Hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_SecondOrderOperator(),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_HessTimesVec(vector_->clone())
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::hessian_t::SR1_HESS);
}

DOTk_SR1Hessian::~DOTk_SR1Hessian()
{
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_SR1Hessian::getDeltaGrad() const
{
    return (m_DeltaGradient);
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_SR1Hessian::getDeltaPrimal() const
{
    return (m_DeltaPrimal);
}

void DOTk_SR1Hessian::getHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                 const std::tr1::shared_ptr<dotk::Vector<Real> > & hess_times_vector_)
{
    hess_times_vector_->update(1., *vector_, 0.);
    Real dgrad_dot_dprimal = m_DeltaGradient->dot(*m_DeltaPrimal);
    Real value = std::fabs(dgrad_dot_dprimal - static_cast<Real>(0.0));
    bool zero_dgrad_dot_dprimal = value <= std::numeric_limits<Real>::min() ? true : false;
    if(zero_dgrad_dot_dprimal == true)
    {
        return;
    }

    Real barzilai_borwein_step = static_cast<Real>(1.0) /
            dotk::DOTk_SecondOrderOperator::getBarzilaiBorweinStep(m_DeltaPrimal, m_DeltaGradient);
    Real barzilai_borwein_step_lower_bound = dotk::DOTk_SecondOrderOperator::getDiagonalScaleFactor() /
            dotk::DOTk_SecondOrderOperator::getLowerBoundOnDiagonalScaleFactor();
    barzilai_borwein_step_lower_bound = std::min(barzilai_borwein_step, barzilai_borwein_step_lower_bound);
    hess_times_vector_->scale(barzilai_borwein_step_lower_bound);
    // use secant equation to approximate H_k * y_k (i.e. H_k+1 s_k = H_k y_k ) assuming H_k = I
    m_HessTimesVec->update(1., *m_DeltaGradient, 0.);
    Real alpha = dgrad_dot_dprimal - m_DeltaPrimal->dot(*m_DeltaPrimal);
    // get beta = (dot(y,vector_) - dot(Hs,vector_) ) / ( dot(y,s) - dot(Hs,s)
    Real beta = (m_DeltaGradient->dot(*vector_) - m_HessTimesVec->dot(*vector_)) / alpha;
    // get hess_times_vector_(k+1) = vector_(k) + dot(gamma*y,vector_)(k) - dot(gamma*Hs,vector_)(k), k=iteration counter
    hess_times_vector_->update(beta, *m_DeltaGradient, 1.);
    hess_times_vector_->update(-beta, *m_HessTimesVec, 1.);

    Real norm_Hv = hess_times_vector_->norm();
    bool negative_curvature_detected = norm_Hv < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_detected == true)
    {
        hess_times_vector_->update(1., *vector_, 0.);
    }
}

void DOTk_SR1Hessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    this->getHessian(vector_, matrix_times_vector_);
}

}
