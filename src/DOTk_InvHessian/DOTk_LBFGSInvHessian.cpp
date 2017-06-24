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

DOTk_LBFGSInvHessian::DOTk_LBFGSInvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                           size_t aSecantStorageSize) :
        dotk::DOTk_SecondOrderOperator(aSecantStorageSize),
        m_Alpha(aSecantStorageSize, 0.),
        m_RhoStorage(std::make_shared<std::vector<Real>>(aSecantStorageSize)),
        m_DeltaPrimal(aVector->clone()),
        m_DeltaGradient(aVector->clone()),
        m_DeltaPrimalStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*aVector, aSecantStorageSize)),
        m_DeltaGradientStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*aVector, aSecantStorageSize))
{
    std::fill(m_RhoStorage->begin(), m_RhoStorage->end(), 0.);
    dotk::DOTk_SecondOrderOperator::setInvHessianType(dotk::types::LBFGS_INV_HESS);
}

DOTk_LBFGSInvHessian::~DOTk_LBFGSInvHessian()
{
}

const std::shared_ptr<std::vector<Real> > & DOTk_LBFGSInvHessian::getDeltaGradPrimalInnerProductStorage() const
{
    /// Return limited memory storage of inner product between deltaGradient and deltaPrimal
    return (m_RhoStorage);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LBFGSInvHessian::getDeltaGradStorage(size_t aIndex) const
{
    return (m_DeltaGradientStorage->basis(aIndex));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LBFGSInvHessian::getDeltaPrimalStorage(size_t aIndex) const
{
    return (m_DeltaPrimalStorage->basis(aIndex));
}

void DOTk_LBFGSInvHessian::getInvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                         const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    Int storage_size = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    aOutput->update(1., *aVector, 0.);
    bool is_secant_information_stored = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() <= 0 ? true : false;
    if(is_secant_information_stored == true)
    {
        return;
    }

    m_Alpha.assign(m_Alpha.size(), 0.);
    for (Int index_i = storage_size; index_i >= 0; index_i--)
    {
        m_Alpha[index_i] = (*m_RhoStorage)[index_i] * m_DeltaPrimalStorage->basis(index_i)->dot(*aOutput);
        aOutput->update(-m_Alpha.operator [](index_i), *(m_DeltaGradientStorage->basis(index_i)), static_cast<Real>(1.));
    }

    for(Int index_j = 0; index_j <= storage_size; ++index_j)
    {
        Real beta = (*m_RhoStorage)[index_j] * m_DeltaGradientStorage->basis(index_j)->dot(*aOutput);
        Real kappa = m_Alpha[index_j] - beta;
        aOutput->update(kappa, *m_DeltaPrimalStorage->basis(index_j), 1.);
    }

    Real norm_invhess_times_vec = aOutput->norm();
    bool negative_curvature_detected = norm_invhess_times_vec < std::numeric_limits<Real>::min() ? true: false;
    if (negative_curvature_detected == true)
    {
        aOutput->update(1., *aVector, 0.);
    }
}

void DOTk_LBFGSInvHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                 const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                 const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(aMng->getNewPrimal(), aMng->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(aMng->getNewGradient(), aMng->getOldGradient(), m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal,
                                                        m_DeltaGradient,
                                                        *m_RhoStorage,
                                                        m_DeltaPrimalStorage,
                                                        m_DeltaGradientStorage);
    this->getInvHessian(aVector, aOutput);
}

}
