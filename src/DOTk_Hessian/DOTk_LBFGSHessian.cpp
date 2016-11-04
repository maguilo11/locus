/*
 * DOTk_LBFGSHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LBFGSHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LBFGSHessian::DOTk_LBFGSHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
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
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::LBFGS_HESS);
}

DOTk_LBFGSHessian::~DOTk_LBFGSHessian()
{
}

const std::tr1::shared_ptr<std::vector<Real> > & DOTk_LBFGSHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LBFGSHessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LBFGSHessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LBFGSHessian::getHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & hess_times_vec_)
{
    /// Limited memory Broyden–Fletcher–Goldfarb–Shanno (BFGS) method \n
    /// In:  \n
    ///      vec_ = Trial step at the current iteration, unchanged on exist. \n
    ///      (std::vector<Real>) \n
    /// Out: \n
    ///      hess_times_vec_ = DOTk_SecondOrderOperator-vector product, i.e. application of the DOTk_SecondOrderOperator operator to the trial step. \n
    ///      (std::vector<Real>) \n
    ///
    Int index = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    hess_times_vec_->copy(*vector_);
    if(dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() == 0)
    {
        return;
    }
    Real inv_barzilai_borwein_step = m_DeltaGradientStorage->basis(index)->dot(*m_DeltaGradientStorage->basis(index));
    inv_barzilai_borwein_step = inv_barzilai_borwein_step / (*m_RhoStorage)[index];
    // lower bound on gamma*sigma, where sigma in (B + sigma*I)
    Real barzilai_borwein_step_lower_bound = dotk::DOTk_SecondOrderOperator::getDiagonalScaleFactor()
            / dotk::DOTk_SecondOrderOperator::getLowerBoundOnDiagonalScaleFactor();
    inv_barzilai_borwein_step = std::min(inv_barzilai_borwein_step, barzilai_borwein_step_lower_bound);
    hess_times_vec_->scale(inv_barzilai_borwein_step);

    for(Int index_i = 0; index_i < dotk::DOTk_SecondOrderOperator::getNumUpdatesStored(); ++index_i)
    {
        Real alpha = static_cast<Real>(1.) / std::sqrt((*m_RhoStorage)[index_i]);
        m_MatrixB->basis(index_i)->copy(*m_DeltaGradientStorage->basis(index_i));
        m_MatrixB->basis(index_i)->scale(alpha);
        m_MatrixA->basis(index_i)->copy(*m_DeltaPrimalStorage->basis(index_i));
        m_MatrixA->basis(index_i)->scale(inv_barzilai_borwein_step);

        for(Int index_j = 0; index_j <= index_i - 1; ++index_j)
        {
            alpha = m_MatrixB->basis(index_j)->dot(*m_DeltaPrimalStorage->basis(index_i));
            m_MatrixA->basis(index_i)->axpy(alpha, *m_MatrixB->basis(index_j));
            alpha = static_cast<Real>(-1.0) * m_MatrixA->basis(index_j)->dot(*m_DeltaPrimalStorage->basis(index_i));
            m_MatrixA->basis(index_i)->axpy(alpha, *m_MatrixA->basis(index_j));
        }

        alpha = static_cast<Real>(1.) / std::sqrt(m_DeltaPrimalStorage->basis(index_i)->dot(*m_MatrixA->basis(index_i)));
        m_MatrixA->basis(index_i)->scale(alpha);
        // compute application of direction to BFGS DOTk Hessian
        alpha = m_MatrixB->basis(index_i)->dot(*vector_);
        hess_times_vec_->axpy(alpha, *m_MatrixB->basis(index_i));
        alpha = m_MatrixA->basis(index_i)->dot(*vector_);
        hess_times_vec_->axpy(-alpha, *m_MatrixA->basis(index_i));
    }
    Real norm_Hv = hess_times_vec_->norm();
    bool negative_curvature_detected = norm_Hv < std::numeric_limits<Real>::min() ? true: false;
    if(negative_curvature_detected)
    {
        hess_times_vec_->copy(*vector_);
    }
}

void DOTk_LBFGSHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                              const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                              const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(),
                                                         mng_->getOldGradient(),
                                                         m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal,
                                                        m_DeltaGradient,
                                                        *m_RhoStorage,
                                                        m_DeltaPrimalStorage,
                                                        m_DeltaGradientStorage);
    this->getHessian(vector_, matrix_times_vec_);
}

}
