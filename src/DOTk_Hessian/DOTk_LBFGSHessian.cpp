/*
 * DOTk_LBFGSHessian.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>
#include <vector>

#include "vector.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_LBFGSHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LBFGSHessian::DOTk_LBFGSHessian(const dotk::Vector<Real> & aVector, size_t aSecantStorageSize) :
        dotk::DOTk_SecondOrderOperator(aSecantStorageSize),
        m_RhoStorage(std::make_shared<std::vector<Real>>(aSecantStorageSize)),
        m_DeltaPrimal(aVector.clone()),
        m_DeltaGradient(aVector.clone()),
        m_MatrixA(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize)),
        m_MatrixB(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize)),
        m_DeltaPrimalStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize)),
        m_DeltaGradientStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize))
{
    std::fill(m_RhoStorage->begin(), m_RhoStorage->end(), 0.);
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::LBFGS_HESS);
}

DOTk_LBFGSHessian::~DOTk_LBFGSHessian()
{
}

const std::shared_ptr<std::vector<Real> > & DOTk_LBFGSHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LBFGSHessian::getDeltaGradStorage(size_t at_) const
{
    return (m_DeltaGradientStorage->basis(at_));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LBFGSHessian::getDeltaPrimalStorage(size_t at_) const
{
    return (m_DeltaPrimalStorage->basis(at_));
}

void DOTk_LBFGSHessian::getHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                   const std::shared_ptr<dotk::Vector<Real> > & aHessianTimesVector)
{
    /// Limited memory Broyden–Fletcher–Goldfarb–Shanno (BFGS) method \n
    /// In:  \n
    ///      aVector = Trial step at the current iteration, unchanged on exist. \n
    ///      (std::vector<Real>) \n
    /// Out: \n
    ///      aHessianTimesVector = DOTk_SecondOrderOperator-vector product, i.e. application of the DOTk_SecondOrderOperator operator to the trial step. \n
    ///      (std::vector<Real>) \n
    ///
    Int index = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    aHessianTimesVector->update(1., *aVector, 0.);
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
    aHessianTimesVector->scale(inv_barzilai_borwein_step);

    for(Int index_i = 0; index_i < dotk::DOTk_SecondOrderOperator::getNumUpdatesStored(); ++index_i)
    {
        Real alpha = static_cast<Real>(1.) / std::sqrt(m_RhoStorage.operator *().operator [](index_i));
        m_MatrixB->basis(index_i)->update(1., *m_DeltaGradientStorage->basis(index_i), 0.);
        m_MatrixB->basis(index_i)->scale(alpha);
        m_MatrixA->basis(index_i)->update(1., *m_DeltaPrimalStorage->basis(index_i), 0.);
        m_MatrixA->basis(index_i)->scale(inv_barzilai_borwein_step);

        for(Int index_j = 0; index_j <= index_i - 1; ++index_j)
        {
            alpha = m_MatrixB->basis(index_j)->dot(*m_DeltaPrimalStorage->basis(index_i));
            m_MatrixA->basis(index_i)->update(alpha, *m_MatrixB->basis(index_j), 1.);
            alpha = static_cast<Real>(-1.0) * m_MatrixA->basis(index_j)->dot(*m_DeltaPrimalStorage->basis(index_i));
            m_MatrixA->basis(index_i)->update(alpha, *m_MatrixA->basis(index_j), 1.);
        }

        Real tValue = m_DeltaPrimalStorage->basis(index_i)->dot(*m_MatrixA->basis(index_i));
        alpha = static_cast<Real>(1.) / std::sqrt(tValue);
        m_MatrixA->basis(index_i)->scale(alpha);
        // compute application of direction to BFGS DOTk Hessian
        alpha = m_MatrixB->basis(index_i)->dot(*aVector);
        aHessianTimesVector->update(alpha, *m_MatrixB->basis(index_i), 1.);
        alpha = m_MatrixA->basis(index_i)->dot(*aVector);
        aHessianTimesVector->update(-alpha, *m_MatrixA->basis(index_i), 1.);
    }
    Real norm_Hv = aHessianTimesVector->norm();
    bool negative_curvature_detected = norm_Hv < std::numeric_limits<Real>::min() ? true: false;
    if(negative_curvature_detected)
    {
        aHessianTimesVector->update(1., *aVector, 0.);
    }
}

void DOTk_LBFGSHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                              const std::shared_ptr<dotk::Vector<Real> > & aVector,
                              const std::shared_ptr<dotk::Vector<Real> > & aHessianTimesVector)
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
    this->getHessian(aVector, aHessianTimesVector);
}

}
