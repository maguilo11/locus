/*
 * DOTk_LDFPHessian.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LDFPHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LDFPHessian::DOTk_LDFPHessian(const dotk::Vector<Real> & vector_,
                                   size_t max_secant_storage_) :
        dotk::DOTk_SecondOrderOperator(max_secant_storage_),
        m_Alpha(max_secant_storage_, 0.),
        m_RhoStorage(max_secant_storage_, 0.),
        m_DeltaPrimal(vector_.clone()),
        m_DeltaGradient(vector_.clone()),
        m_DeltaPrimalStorage(new dotk::serial::DOTk_RowMatrix<Real>(vector_, max_secant_storage_)),
        m_DeltaGradientStorage(new dotk::serial::DOTk_RowMatrix<Real>(vector_, max_secant_storage_))
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::LDFP_HESS);
}

DOTk_LDFPHessian::~DOTk_LDFPHessian()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LDFPHessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LDFPHessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LDFPHessian::getHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                  const std::shared_ptr<dotk::Vector<Real> > & hess_times_vec_)
{
    Int storage_size = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    m_Alpha.assign(m_Alpha.size(), 0.);
    hess_times_vec_->update(1., *vector_, 0.);
    for(Int index = storage_size; index >= 0; index--)
    {
        m_Alpha[index] = m_RhoStorage[index] * m_DeltaGradientStorage->basis(index)->dot(*hess_times_vec_);
        hess_times_vec_->update(-m_Alpha[index], *(m_DeltaPrimalStorage->basis(index)), static_cast<Real>(1.));
    }
    for(Int index = 0; index <= storage_size; ++index)
    {
        Real beta = m_RhoStorage[index] * m_DeltaPrimalStorage->basis(index)->dot(*hess_times_vec_);
        Real kappa = m_Alpha[index] - beta;
        hess_times_vec_->update(kappa, *m_DeltaGradientStorage->basis(index), 1.);
    }
    Real norm_Hv = hess_times_vec_->norm();
    bool negative_curvature_detected = norm_Hv < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_detected)
    {
        hess_times_vec_->update(1., *vector_, 0.);
    }
}

void DOTk_LDFPHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                             const std::shared_ptr<dotk::Vector<Real> > & vector_,
                             const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal,
                                                        m_DeltaGradient,
                                                        m_RhoStorage,
                                                        m_DeltaPrimalStorage,
                                                        m_DeltaGradientStorage);
    this->getHessian(vector_, matrix_times_vec_);
}

}
