/*
 * DOTk_BarzilaiBorweinHessian.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>
#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_BarzilaiBorweinHessian.hpp"

namespace dotk
{

DOTk_BarzilaiBorweinHessian::DOTk_BarzilaiBorweinHessian(const dotk::Vector<Real> & vector_) :
        dotk::DOTk_SecondOrderOperator(),
        m_DeltaPrimal(vector_.clone()),
        m_DeltaGradient(vector_.clone())
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::hessian_t::BARZILAIBORWEIN_HESS);
}

DOTk_BarzilaiBorweinHessian::~DOTk_BarzilaiBorweinHessian()
{
}

void DOTk_BarzilaiBorweinHessian::computeDeltaPrimal(const std::shared_ptr<dotk::Vector<Real> > & new_primal_,
                                                     const std::shared_ptr<dotk::Vector<Real> > & old_primal_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(new_primal_, old_primal_, m_DeltaPrimal);
}

void DOTk_BarzilaiBorweinHessian::computeDeltaGradient(const std::shared_ptr<dotk::Vector<Real> > & new_gradient_,
                                                       const std::shared_ptr<dotk::Vector<Real> > & old_gradient_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(new_gradient_, old_gradient_, m_DeltaGradient);
}

void DOTk_BarzilaiBorweinHessian::getHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                             const std::shared_ptr<dotk::Vector<Real> > & mat_times_vec_)
{
    mat_times_vec_->update(1., *vector_, 0.);
    Real innr_deltaGrad_deltaPrimal = m_DeltaGradient->dot(*m_DeltaPrimal);
    bool negative_curvature_detected =
            innr_deltaGrad_deltaPrimal <= std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_detected)
    {
        return;
    }
    Real scaling = static_cast<Real>(1.0)
            / dotk::DOTk_SecondOrderOperator::getBarzilaiBorweinStep(m_DeltaPrimal, m_DeltaGradient);
    mat_times_vec_->scale(scaling);
}
void DOTk_BarzilaiBorweinHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                        const std::shared_ptr<dotk::Vector<Real> > & vec_,
                                        const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_)
{
    this->computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal());
    this->computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient());
    this->getHessian(vec_, matrix_times_vec_);
}

}
