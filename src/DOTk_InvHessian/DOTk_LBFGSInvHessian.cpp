/*
 * DOTk_LBFGSInvHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LBFGSInvHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LBFGSInvHessian::DOTk_LBFGSInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                           size_t max_secant_storage_) :
        dotk::DOTk_SecondOrderOperator(max_secant_storage_),
        m_Alpha(max_secant_storage_, 0.),
        m_RhoStorage(new std::vector<Real>(max_secant_storage_, 0.)),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_DeltaPrimalStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaGradientStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_))
{
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::LBFGS_INV_HESS);
}

DOTk_LBFGSInvHessian::~DOTk_LBFGSInvHessian()
{
}

const std::tr1::shared_ptr<std::vector<Real> > & DOTk_LBFGSInvHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LBFGSInvHessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LBFGSInvHessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LBFGSInvHessian::getInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                         const std::tr1::shared_ptr<dotk::vector<Real> > & inv_hess_times_vector_)
{
    int storage_size = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    inv_hess_times_vector_->copy(*vector_);
    bool is_secant_information_stored = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() <= 0 ? true : false;
    if(is_secant_information_stored == true)
    {
        return;
    }

    m_Alpha.assign(m_Alpha.size(), 0.);
    for (int index_i = storage_size; index_i >= 0; index_i--)
    {
        m_Alpha[index_i] = (*m_RhoStorage)[index_i] * m_DeltaPrimalStorage->basis(index_i)->dot(*inv_hess_times_vector_);
        inv_hess_times_vector_->axpy(-m_Alpha[index_i], *m_DeltaGradientStorage->basis(index_i));
    }

    for(int index_j = 0; index_j <= storage_size; ++index_j)
    {
        Real beta = (*m_RhoStorage)[index_j] * m_DeltaGradientStorage->basis(index_j)->dot(*inv_hess_times_vector_);
        Real kappa = m_Alpha[index_j] - beta;
        inv_hess_times_vector_->axpy(kappa, *m_DeltaPrimalStorage->basis(index_j));
    }

    Real norm_invhess_times_vec = inv_hess_times_vector_->norm();
    bool negative_curvature_detected = norm_invhess_times_vec < std::numeric_limits<Real>::min() ? true: false;
    if (negative_curvature_detected == true)
    {
        inv_hess_times_vector_->copy(*vector_);
    }
}

void DOTk_LBFGSInvHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal,
                                                        m_DeltaGradient,
                                                        *m_RhoStorage,
                                                        m_DeltaPrimalStorage,
                                                        m_DeltaGradientStorage);
    this->getInvHessian(vector_, matrix_times_vector_);
}

}
