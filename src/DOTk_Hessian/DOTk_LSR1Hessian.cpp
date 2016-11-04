/*
 * DOTk_LSR1Hessian.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LSR1Hessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LSR1Hessian::DOTk_LSR1Hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                   size_t max_secant_storage_) :
        dotk::DOTk_SecondOrderOperator(max_secant_storage_),
        m_DeltaPrimal(vector_->clone()),
        m_DeltaGradient(vector_->clone()),
        m_MatrixA(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaPrimalStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_)),
        m_DeltaGradientStorage(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, max_secant_storage_))
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::LSR1_HESS);
}

DOTk_LSR1Hessian::~DOTk_LSR1Hessian()
{
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LSR1Hessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LSR1Hessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LSR1Hessian::unrollingSR1(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & hess_times_vector_)
{
    /// Memory efficient SR1 formula, use for limited memory DOTk Hessian approximations \n
    /// \n
    /// In:  \n
    ///      vector_ = Trial step at the current iteration, unchanged on exist. \n
    ///      (std::vector_<Real>) \n
    /// In/Out: \n
    ///      hess_times_vector_ = DOTk Hessian-vector_ product, i.e. application of \n
    ///                              the DOTk Hessian operator to the trial step. \n
    ///      (std::vector_<Real>) \n
    ///
    size_t updates = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored();
    for(size_t index = 0; index < updates; ++index)
    {
        m_MatrixA->basis(index)->copy(*m_DeltaPrimalStorage->basis(index));
    }

    for(size_t index_i = 0; index_i < updates; ++index_i)
    {
        Real dgrad_dot_vector = m_DeltaGradientStorage->basis(index_i)->dot(*vector_);
        Real rowA_times_vector = m_MatrixA->basis(index_i)->dot(*vector_);
        Real dprimal_dot_dgrad_outer = m_DeltaGradientStorage->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_i));
        Real rowA_dot_dgrad_outer = m_MatrixA->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_i));
        Real alpha = (dgrad_dot_vector - rowA_times_vector) / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
        hess_times_vector_->axpy(alpha, *m_DeltaGradientStorage->basis(index_i));
        hess_times_vector_->axpy(-alpha, *m_MatrixA->basis(index_i));

        for(Int index_j = updates - 1; index_j > index_i; --index_j)
        {
            Real dgrad_dot_dprimal_inner =
                    m_DeltaGradientStorage->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_j));
            Real rowA_dot_dgrad_inner = m_MatrixA->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_j));
            Real beta = (dgrad_dot_dprimal_inner - rowA_dot_dgrad_inner)
                    / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
            m_MatrixA->basis(index_j)->axpy(beta, *m_DeltaGradientStorage->basis(index_i));
            m_MatrixA->basis(index_j)->axpy(-beta, *m_MatrixA->basis(index_i));
        }
    }
}

void DOTk_LSR1Hessian::getHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & hess_times_vector_)
{
    hess_times_vector_->copy(*vector_);
    if(dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() == 0)
    {
        return;
    }
    this->unrollingSR1(vector_, hess_times_vector_);
    Real norm_Hess_times_vec = hess_times_vector_->norm();
    bool negative_curvature_encountered = norm_Hess_times_vec < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_encountered)
    {
        hess_times_vector_->copy(*vector_);
    }
}

void DOTk_LSR1Hessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                             const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                             const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(mng_->getNewPrimal(), mng_->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(mng_->getNewGradient(), mng_->getOldGradient(), m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal, m_DeltaGradient, m_DeltaPrimalStorage, m_DeltaGradientStorage);
    this->getHessian(vector_, matrix_times_vector_);
}

}
