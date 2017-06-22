/*
 * DOTk_LDFPInvHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LDFPInvHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LDFPInvHessian::DOTk_LDFPInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                             size_t max_secant_storage_) :
        dotk::DOTk_SecondOrderOperator(max_secant_storage_),
        m_RhoStorage(new std::vector<Real>(max_secant_storage_, 0.)),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_MatrixA(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_MatrixB(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaPrimalStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaGradientStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_))
{
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::LDFP_INV_HESS);
}

DOTk_LDFPInvHessian::~DOTk_LDFPInvHessian()
{
}

const std::shared_ptr<std::vector<Real> > & DOTk_LDFPInvHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LDFPInvHessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LDFPInvHessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LDFPInvHessian::getInvHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                        const std::shared_ptr<dotk::Vector<Real> > & inv_hess_times_vector_)
{
    Int index = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    inv_hess_times_vector_->update(1., *vector_, 0.);
    bool is_secant_information_stored = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() <= 0 ? true : false;
    if(is_secant_information_stored == true)
    {
        return;
    }

    // lower bound on gamma*sigma, where sigma in (B + sigma*I)
    Real lower_bound_on_barzilai_borwein_step = dotk::DOTk_SecondOrderOperator::getDiagonalScaleFactor()
            / dotk::DOTk_SecondOrderOperator::getLowerBoundOnDiagonalScaleFactor();
    Real barzilai_borwein_step = (*m_RhoStorage)[index]
            / m_DeltaPrimalStorage->basis(index)->dot(*m_DeltaPrimalStorage->basis(index));
    barzilai_borwein_step = std::min(barzilai_borwein_step, lower_bound_on_barzilai_borwein_step);
    inv_hess_times_vector_->scale(barzilai_borwein_step);

    for(Int index_i = 0; index_i < dotk::DOTk_SecondOrderOperator::getNumUpdatesStored(); ++index_i)
    {
        Real alpha = static_cast<Real>(1.0) / std::sqrt((*m_RhoStorage)[index_i]);
        m_MatrixB->basis(index_i)->update(1., *m_DeltaPrimalStorage->basis(index_i), 0.);
        m_MatrixB->basis(index_i)->scale(alpha);
        m_MatrixA->basis(index_i)->update(1., *m_DeltaGradientStorage->basis(index_i), 0.);
        m_MatrixA->basis(index_i)->scale(barzilai_borwein_step);

        for(Int index_j = 0; index_j <= index_i - 1; ++index_j)
        {
            alpha = m_MatrixB->basis(index_j)->dot(*m_DeltaGradientStorage->basis(index_i));
            m_MatrixA->basis(index_i)->update(alpha, *m_MatrixB->basis(index_j), 1.);
            alpha = static_cast<Real>(-1.0) * (m_MatrixA->basis(index_j)->dot(*m_DeltaGradientStorage->basis(index_i)));
            m_MatrixA->basis(index_i)->update(alpha, *m_MatrixA->basis(index_j), 1.);
        }

        alpha = static_cast<Real>(1.0)
                / std::sqrt(m_DeltaGradientStorage->basis(index_i)->dot(*m_MatrixA->basis(index_i)));
        m_MatrixA->basis(index_i)->scale(alpha);
        // compute application of direction to BFGS DOTk Hessian
        alpha = m_MatrixB->basis(index_i)->dot(*vector_);
        inv_hess_times_vector_->update(alpha, *m_MatrixB->basis(index_i), 1.);
        alpha = m_MatrixA->basis(index_i)->dot(*vector_);
        inv_hess_times_vector_->update(-alpha, *m_MatrixA->basis(index_i), 1.);
    }

    Real norm_invHess_times_vec = inv_hess_times_vector_->norm();
    bool negative_curvature_encountered = norm_invHess_times_vec < std::numeric_limits<Real>::min() ? true: false;
    if(negative_curvature_encountered == true)
    {
        inv_hess_times_vector_->update(1., *vector_, 0.);
    }
}

void DOTk_LDFPInvHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_)
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
