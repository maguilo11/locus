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

DOTk_LSR1Hessian::DOTk_LSR1Hessian(const dotk::Vector<Real> & aVector, size_t aSecantStorageSize) :
        dotk::DOTk_SecondOrderOperator(aSecantStorageSize),
        m_DeltaPrimal(aVector.clone()),
        m_DeltaGradient(aVector.clone()),
        m_MatrixA(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize)),
        m_DeltaPrimalStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize)),
        m_DeltaGradientStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize))
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::LSR1_HESS);
}

DOTk_LSR1Hessian::~DOTk_LSR1Hessian()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LSR1Hessian::getDeltaGradStorage(size_t aIndex) const
{
    return (m_DeltaGradientStorage->basis(aIndex));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LSR1Hessian::getDeltaPrimalStorage(size_t aIndex) const
{
    return (m_DeltaPrimalStorage->basis(aIndex));
}

void DOTk_LSR1Hessian::unrollingSR1(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                    const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    /// Memory efficient SR1 formula, use for limited memory DOTk Hessian approximations \n
    /// \n
    /// In:  \n
    ///      aVector = Trial step at the current iteration, unchanged on exist. \n
    ///      (std::aVector<Real>) \n
    /// In/Out: \n
    ///      aOutput = DOTk Hessian-aVector product, i.e. application of \n
    ///                              the DOTk Hessian operator to the trial step. \n
    ///      (std::aVector<Real>) \n
    ///
    Int updates = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored();
    for(Int index = 0; index < updates; index++)
    {
        m_MatrixA->basis(index)->update(1., *m_DeltaPrimalStorage->basis(index), 0.);
    }

    for(Int index_i = 0; index_i < updates; index_i++)
    {
        Real dgrad_dot_vector = m_DeltaGradientStorage->basis(index_i)->dot(*aVector);
        Real rowA_times_vector = m_MatrixA->basis(index_i)->dot(*aVector);
        Real dprimal_dot_dgrad_outer = m_DeltaGradientStorage->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_i));
        Real rowA_dot_dgrad_outer = m_MatrixA->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_i));
        Real alpha = (dgrad_dot_vector - rowA_times_vector) / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
        aOutput->update(alpha, *m_DeltaGradientStorage->basis(index_i), 1.);
        aOutput->update(-alpha, *m_MatrixA->basis(index_i), 1.);

        for(Int index_j = updates - 1; index_j > index_i; index_j--)
        {
            Real dgrad_dot_dprimal_inner =
                    m_DeltaGradientStorage->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_j));
            Real rowA_dot_dgrad_inner = m_MatrixA->basis(index_i)->dot(*m_DeltaPrimalStorage->basis(index_j));
            Real beta = (dgrad_dot_dprimal_inner - rowA_dot_dgrad_inner)
                    / (dprimal_dot_dgrad_outer - rowA_dot_dgrad_outer);
            m_MatrixA->basis(index_j)->update(beta, *m_DeltaGradientStorage->basis(index_i), 1.);
            m_MatrixA->basis(index_j)->update(-beta, *m_MatrixA->basis(index_i), 1.);
        }
    }
}

void DOTk_LSR1Hessian::getHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                  const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    aOutput->update(1., *aVector, 0.);
    if(dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() == 0)
    {
        return;
    }
    this->unrollingSR1(aVector, aOutput);
    Real norm_Hess_times_vec = aOutput->norm();
    bool negative_curvature_encountered = norm_Hess_times_vec < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_encountered)
    {
        aOutput->update(1., *aVector, 0.);
    }
}

void DOTk_LSR1Hessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                             const std::shared_ptr<dotk::Vector<Real> > & aVector,
                             const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(aMng->getNewPrimal(), aMng->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(aMng->getNewGradient(), aMng->getOldGradient(), m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal, m_DeltaGradient, m_DeltaPrimalStorage, m_DeltaGradientStorage);
    this->getHessian(aVector, aOutput);
}

}
