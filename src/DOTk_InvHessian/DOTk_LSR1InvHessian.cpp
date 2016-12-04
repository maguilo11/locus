/*
 * DOTk_LSR1InvHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LSR1InvHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LSR1InvHessian::DOTk_LSR1InvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                         size_t max_secant_storage_) :
        dotk::DOTk_SecondOrderOperator(max_secant_storage_),
        m_RhoStorage(new std::vector<Real>(max_secant_storage_, 0.)),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_MatrixA(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaPrimalStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaGradientStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_))
{
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::LSR1_INV_HESS);
}

DOTk_LSR1InvHessian::~DOTk_LSR1InvHessian()
{
}

const std::tr1::shared_ptr<std::vector<Real> > & DOTk_LSR1InvHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LSR1InvHessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LSR1InvHessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LSR1InvHessian::unrollingSR1(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                       const std::tr1::shared_ptr<dotk::vector<Real> > & inv_hess_times_vector_)
{
    /// Memory efficient SR1 formula, use for limited memory SOL_Hessian approximations \n
    /// \n
    /// In:  \n
    ///      vector_ = Trial step at the current iteration, unchanged on exist. \n
    ///      (std::vector_<Real>) \n
    /// In/Out: \n
    ///      inv_hess_times_vector_ = SOL_Hessian-vector_ product, i.e. application of \n
    ///                                 the SOL_Hessian operator to the trial step. \n
    ///      (std::vector_<Real>) \n
    ///
    int updates = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored();
    for(int index = 0; index < updates; ++index)
    {
        m_MatrixA->basis(index)->copy(*m_DeltaGradientStorage->basis(index));
    }

    for(int index_i = 0; index_i < updates; ++index_i)
    {
        Real dgrad_dot_vec = m_DeltaPrimalStorage->basis(index_i)->dot(*vector_);
        Real rowA_dot_vec = m_MatrixA->basis(index_i)->dot(*vector_);
        Real dprimal_dot_dgrad_outer =
                m_DeltaPrimalStorage->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_i));
        Real rowA_dot_dgrad_outer = m_MatrixA->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_i));
        Real alpha = (dgrad_dot_vec - rowA_dot_vec) / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
        inv_hess_times_vector_->axpy(alpha, *m_DeltaPrimalStorage->basis(index_i));
        inv_hess_times_vector_->axpy(-alpha, *m_MatrixA->basis(index_i));

        for(int index_j = updates - 1; index_j > index_i; --index_j)
        {
            Real dgrad_dot_dprimal_inner =
                    m_DeltaPrimalStorage->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_j));
            Real rowA_dot_dgrad_inner = m_MatrixA->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_j));
            Real beta = (dgrad_dot_dprimal_inner - rowA_dot_dgrad_inner)
                    / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
            m_MatrixA->basis(index_j)->axpy(beta, *m_DeltaPrimalStorage->basis(index_i));
            m_MatrixA->basis(index_j)->axpy(-beta, *m_MatrixA->basis(index_i));
        }
    }
}

void DOTk_LSR1InvHessian::getInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                        const std::tr1::shared_ptr<dotk::vector<Real> > & inv_hess_times_vector_)
{
    /// Limited memory Symmetric Rank-1 (LSR1) DOTk Hessian approximation method. \n
    /// The symmetric rank-1 updating formula satisfies the following secant equation: \n
    /// \n
    ///               (dX_k - M_k*dG_k)*(dX_k - M_k*dG_k)^T \n
    /// M_k+1 = M_k + ------------------------------------  \n
    ///                     (dX_k - H_k*dG_k)^T * dG_k      \n
    /// \n
    /// In:  \n
    ///      vector_ = direction at the current iteration, unchanged on exist. \n
    ///      (std::tr1::shared_ptr<dotk::vector<Real> >) \n
    /// Out: \n
    ///      inv_hess_times_vector_ = inverse DOTk Hessian-vector product, i.e. application of the inverse DOTk Hessian \n
    ///      operator to the trial step. \n
    ///      (std::tr1::shared_ptr<dotk::vector<Real> >) \n
    ///
    inv_hess_times_vector_->copy(*vector_);
    bool is_secant_information_stored = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() <= 0 ? true: false;
    if(is_secant_information_stored == true)
    {
        return;
    }

    this->unrollingSR1(vector_, inv_hess_times_vector_);

    Real norm_invHess_times_vec = inv_hess_times_vector_->norm();
    bool negative_curvature_detected = norm_invHess_times_vec < std::numeric_limits<Real>::min() ? true: false;
    if(negative_curvature_detected == true)
    {
        inv_hess_times_vector_->copy(*vector_);
    }
}

void DOTk_LSR1InvHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_)
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
    this->getInvHessian(vector_, matrix_times_vector_);
}

}
