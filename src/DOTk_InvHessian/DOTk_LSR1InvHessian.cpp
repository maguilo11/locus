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

DOTk_LSR1InvHessian::DOTk_LSR1InvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                         size_t aSecantStorageSize) :
        dotk::DOTk_SecondOrderOperator(aSecantStorageSize),
        m_RhoStorage(std::make_shared<std::vector<Real>>(aSecantStorageSize)),
        m_DeltaPrimal(aVector->clone()),
        m_DeltaGradient(aVector->clone()),
        m_MatrixA(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*aVector, aSecantStorageSize)),
        m_DeltaPrimalStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*aVector, aSecantStorageSize)),
        m_DeltaGradientStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*aVector, aSecantStorageSize))
{
    std::fill(m_RhoStorage->begin(), m_RhoStorage->end(), 0.);
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::LSR1_INV_HESS);
}

DOTk_LSR1InvHessian::~DOTk_LSR1InvHessian()
{
}

const std::shared_ptr<std::vector<Real> > & DOTk_LSR1InvHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LSR1InvHessian::getDeltaGradStorage(size_t aIndex) const
{
    return (m_DeltaGradientStorage->basis(aIndex));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LSR1InvHessian::getDeltaPrimalStorage(size_t aIndex) const
{
    return (m_DeltaPrimalStorage->basis(aIndex));
}

void DOTk_LSR1InvHessian::unrollingSR1(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                       const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    /// Memory efficient SR1 formula, use for limited memory SOL_Hessian approximations \n
    /// \n
    /// In:  \n
    ///      aVector = Trial step at the current iteration, unchanged on exist. \n
    ///      (std::aVector<Real>) \n
    /// In/Out: \n
    ///      aOutput = SOL_Hessian-aVector product, i.e. application of \n
    ///                                 the SOL_Hessian operator to the trial step. \n
    ///      (std::aVector<Real>) \n
    ///
    int updates = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored();
    for(int index = 0; index < updates; ++index)
    {
        m_MatrixA->basis(index)->update(1., *m_DeltaGradientStorage->basis(index), 0.);
    }

    for(int index_i = 0; index_i < updates; ++index_i)
    {
        Real dgrad_dot_vec = m_DeltaPrimalStorage->basis(index_i)->dot(*aVector);
        Real rowA_dot_vec = m_MatrixA->basis(index_i)->dot(*aVector);
        Real dprimal_dot_dgrad_outer =
                m_DeltaPrimalStorage->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_i));
        Real rowA_dot_dgrad_outer = m_MatrixA->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_i));
        Real alpha = (dgrad_dot_vec - rowA_dot_vec) / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
        aOutput->update(alpha, *m_DeltaPrimalStorage->basis(index_i), 1.);
        aOutput->update(-alpha, *m_MatrixA->basis(index_i), 1.);

        for(int index_j = updates - 1; index_j > index_i; --index_j)
        {
            Real dgrad_dot_dprimal_inner =
                    m_DeltaPrimalStorage->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_j));
            Real rowA_dot_dgrad_inner = m_MatrixA->basis(index_i)->dot(*m_DeltaGradientStorage->basis(index_j));
            Real beta = (dgrad_dot_dprimal_inner - rowA_dot_dgrad_inner)
                    / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
            m_MatrixA->basis(index_j)->update(beta, *m_DeltaPrimalStorage->basis(index_i), 1.);
            m_MatrixA->basis(index_j)->update(-beta, *m_MatrixA->basis(index_i), 1.);
        }
    }
}

void DOTk_LSR1InvHessian::getInvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                        const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    /// Limited memory Symmetric Rank-1 (LSR1) DOTk Hessian approximation method. \n
    /// The symmetric rank-1 updating formula satisfies the following secant equation: \n
    /// \n
    ///               (dX_k - M_k*dG_k)*(dX_k - M_k*dG_k)^T \n
    /// M_k+1 = M_k + ------------------------------------  \n
    ///                     (dX_k - H_k*dG_k)^T * dG_k      \n
    /// \n
    /// In:  \n
    ///      aVector = direction at the current iteration, unchanged on exist. \n
    ///      (std::shared_ptr<dotk::Vector<Real> >) \n
    /// Out: \n
    ///      aOutput = inverse DOTk Hessian-vector product, i.e. application of the inverse DOTk Hessian \n
    ///      operator to the trial step. \n
    ///      (std::shared_ptr<dotk::Vector<Real> >) \n
    ///
    aOutput->update(1., *aVector, 0.);
    bool is_secant_information_stored = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() <= 0 ? true: false;
    if(is_secant_information_stored == true)
    {
        return;
    }

    this->unrollingSR1(aVector, aOutput);

    Real norm_invHess_times_vec = aOutput->norm();
    bool negative_curvature_detected = norm_invHess_times_vec < std::numeric_limits<Real>::min() ? true: false;
    if(negative_curvature_detected == true)
    {
        aOutput->update(1., *aVector, 0.);
    }
}

void DOTk_LSR1InvHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(aMng->getNewPrimal(), aMng->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(aMng->getNewGradient(),
                                                         aMng->getOldGradient(),
                                                         m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal,
                                                        m_DeltaGradient,
                                                        *m_RhoStorage,
                                                        m_DeltaPrimalStorage,
                                                        m_DeltaGradientStorage);
    this->getInvHessian(aVector, aOutput);
}

}
