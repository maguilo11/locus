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

DOTk_LDFPHessian::DOTk_LDFPHessian(const dotk::Vector<Real> & aVector, size_t aSecantStorageSize) :
        dotk::DOTk_SecondOrderOperator(aSecantStorageSize),
        m_Alpha(aSecantStorageSize, 0.),
        m_RhoStorage(aSecantStorageSize, 0.),
        m_DeltaPrimal(aVector.clone()),
        m_DeltaGradient(aVector.clone()),
        m_DeltaPrimalStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize)),
        m_DeltaGradientStorage(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(aVector, aSecantStorageSize))
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::LDFP_HESS);
}

DOTk_LDFPHessian::~DOTk_LDFPHessian()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LDFPHessian::getDeltaGradStorage(size_t aIndex) const
{
    return (m_DeltaGradientStorage->basis(aIndex));
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LDFPHessian::getDeltaPrimalStorage(size_t aIndex) const
{
    return (m_DeltaPrimalStorage->basis(aIndex));
}

void DOTk_LDFPHessian::getHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                  const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    Int storage_size = dotk::DOTk_SecondOrderOperator::getNumUpdatesStored() - 1;
    m_Alpha.assign(m_Alpha.size(), 0.);
    aOutput->update(1., *aVector, 0.);
    for(Int index = storage_size; index >= 0; index--)
    {
        m_Alpha[index] = m_RhoStorage[index] * m_DeltaGradientStorage->basis(index)->dot(*aOutput);
        aOutput->update(-m_Alpha.operator [](index), m_DeltaPrimalStorage->basis(index).operator *(), 1.);
    }
    for(Int index = 0; index <= storage_size; ++index)
    {
        Real beta = m_RhoStorage[index] * m_DeltaPrimalStorage->basis(index)->dot(*aOutput);
        Real kappa = m_Alpha[index] - beta;
        aOutput->update(kappa, *m_DeltaGradientStorage->basis(index), 1.);
    }
    Real norm_Hv = aOutput->norm();
    bool negative_curvature_detected = norm_Hv < std::numeric_limits<Real>::min() ? true : false;
    if(negative_curvature_detected)
    {
        aOutput->update(1., *aVector, 0.);
    }
}

void DOTk_LDFPHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                             const std::shared_ptr<dotk::Vector<Real> > & aVector,
                             const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    dotk::DOTk_SecondOrderOperator::computeDeltaPrimal(aMng->getNewPrimal(), aMng->getOldPrimal(), m_DeltaPrimal);
    dotk::DOTk_SecondOrderOperator::computeDeltaGradient(aMng->getNewGradient(), aMng->getOldGradient(), m_DeltaGradient);
    dotk::DOTk_SecondOrderOperator::updateSecantStorage(m_DeltaPrimal,
                                                        m_DeltaGradient,
                                                        m_RhoStorage,
                                                        m_DeltaPrimalStorage,
                                                        m_DeltaGradientStorage);
    this->getHessian(aVector, aOutput);
}

}
