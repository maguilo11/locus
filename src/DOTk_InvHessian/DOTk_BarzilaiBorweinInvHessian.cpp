/*
 * DOTk_BarzilaiBorweinInvHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_BarzilaiBorweinInvHessian.hpp"

namespace dotk
{

DOTk_BarzilaiBorweinInvHessian::DOTk_BarzilaiBorweinInvHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_SecondOrderOperator(),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone())
{
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::BARZILAIBORWEIN_INV_HESS);
}

DOTk_BarzilaiBorweinInvHessian::~DOTk_BarzilaiBorweinInvHessian()
{
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_BarzilaiBorweinInvHessian::getDeltaGrad() const
{
    return (m_DeltaGradient);
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_BarzilaiBorweinInvHessian::getDeltaPrimal() const
{
    return (m_DeltaPrimal);
}

void DOTk_BarzilaiBorweinInvHessian::getInvHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & inv_hess_times_vector_)
{
    inv_hess_times_vector_->update(1., *vector_, 0.);
    Real dgrad_dot_dprimal = m_DeltaGradient->dot(*m_DeltaPrimal);
    Real value = std::fabs(dgrad_dot_dprimal - static_cast<Real>(0.0));
    bool zero_dgrad_dot_dprimal = value <= std::numeric_limits<Real>::min() ? true : false;
    if(zero_dgrad_dot_dprimal == true)
    {
        return;
    }
    Real scaling = dotk::DOTk_SecondOrderOperator::getBarzilaiBorweinStep(m_DeltaPrimal, m_DeltaGradient);
    inv_hess_times_vector_->scale(scaling);
}

void DOTk_BarzilaiBorweinInvHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    this->getInvHessian(vector_, matrix_times_vector_);
}

}
