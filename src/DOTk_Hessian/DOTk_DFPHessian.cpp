/*
 * DOTk_DFPHessian.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DFPHessian.hpp"

namespace dotk
{

DOTk_DFPHessian::DOTk_DFPHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_) :
        dotk::DOTk_SecondOrderOperator(),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_HessTimesVec(vector_->clone())
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::hessian_t::DFP_HESS);
}

DOTk_DFPHessian::~DOTk_DFPHessian()
{
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_DFPHessian::getDeltaGrad() const
{
    return (m_DeltaGradient);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_DFPHessian::getDeltaPrimal() const
{
    return (m_DeltaPrimal);
}

void DOTk_DFPHessian::getHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & hess_times_vec_)
{
    hess_times_vec_->copy(*vec_);
    Real innr_deltaGrad_vector = m_DeltaGradient->dot(*vec_);
    Real value = std::fabs(innr_deltaGrad_vector - static_cast<Real>(0.0));
    bool zero_innr_deltaGrad_vector = value <= std::numeric_limits<Real>::min() ? true : false;
    if((zero_innr_deltaGrad_vector == true) || (dotk::DOTk_SecondOrderOperator::getNumOptimizationItrDone() == 1))
    {
        return;
    }
    // get Hv_new = (I - rho*y*s) * (I - rho*s*y)*vec_ + rho*y*y^t*vec_
    Real rho = static_cast<Real>(1.) / m_DeltaGradient->dot(*m_DeltaPrimal);
    Real alpha = rho * innr_deltaGrad_vector;
    // compute rho*y*y^t*vec_
    hess_times_vec_->copy(*m_DeltaGradient);
    hess_times_vec_->scale(alpha);
    // compute gamma*I*(vec_ - alpha*s)
    m_HessTimesVec->copy(*vec_);
    m_HessTimesVec->axpy(-alpha, *m_DeltaPrimal);
    // compute beta = rho * dot(Svec,(vec_ - alpha*s))
    Real beta = rho * m_HessTimesVec->dot(*m_DeltaPrimal);
    // compute (I - rho*y*s)^t * (I - rho*s*y)*vec_
    hess_times_vec_->axpy(-beta, *m_DeltaGradient);
    hess_times_vec_->axpy(static_cast<Real>(1.0), *m_HessTimesVec);
    Real norm_Hv = hess_times_vec_->norm();
    bool negative_curvature_detected = norm_Hv < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_detected == true)
    {
        hess_times_vec_->copy(*vec_);
    }
}

void DOTk_DFPHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    this->getHessian(vec_, matrix_times_vec_);
}

}
