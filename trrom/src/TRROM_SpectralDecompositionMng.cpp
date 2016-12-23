/*
 * TRROM_SpectralDecompositionMng.cpp
 *
 *  Created on: Aug 17, 2016
 *      Author: maguilo
 */

#include <cassert>
#include <sstream>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_ReducedBasis.hpp"
#include "TRROM_BrandLowRankSVD.hpp"
#include "TRROM_LinearAlgebraFactory.hpp"
#include "TRROM_SpectralDecomposition.hpp"
#include "TRROM_OrthogonalFactorization.hpp"
#include "TRROM_SpectralDecompositionMng.hpp"
#include "TRROM_LowRankSpectralDecomposition.hpp"

namespace trrom
{

SpectralDecompositionMng::SpectralDecompositionMng() :
        m_Factory(),
        m_FullRankSVD(),
        m_LowRankSVD(),
        m_DualEnergyThreshold(0.99),
        m_StateEnergyThreshold(0.99),
        m_LeftHandSideEnergyThreshold(0.99),
        m_DualSnapshotsCollected(false),
        m_StateSnapshotsCollected(false),
        m_LeftHandSideSnapshotsCollected(false),
        m_DualLowRankSVD_Active(false),
        m_StateLowRankSVD_Active(false),
        m_LeftHandSideLowRankSVD_Active(false),
        m_DualSingularValues(),
        m_DualSnapshotEnsemble(),
        m_DualLeftSingularVectors(),
        m_DualRightSingularVectors(),
        m_StateSingularValues(),
        m_StateSnapshotEnsemble(),
        m_StateLeftSingularVectors(),
        m_StateRightSingularVectors(),
        m_LeftHandSideSingularValues(),
        m_LeftHandSideSnapshotEnsemble(),
        m_LeftHandSideLeftSingularVectors(),
        m_LeftHandSideRightSingularVectors()
{
}

SpectralDecompositionMng::SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & algebra_factory_,
                                                   const std::tr1::shared_ptr<trrom::SpectralDecomposition> & full_rank_svd_,
                                                   const std::tr1::shared_ptr<trrom::LowRankSpectralDecomposition> & low_rank_svd_) :
        m_Factory(algebra_factory_),
        m_FullRankSVD(full_rank_svd_),
        m_LowRankSVD(low_rank_svd_),
        m_DualEnergyThreshold(0.99),
        m_StateEnergyThreshold(0.99),
        m_LeftHandSideEnergyThreshold(0.99),
        m_DualSnapshotsCollected(false),
        m_StateSnapshotsCollected(false),
        m_LeftHandSideSnapshotsCollected(false),
        m_DualLowRankSVD_Active(false),
        m_StateLowRankSVD_Active(false),
        m_LeftHandSideLowRankSVD_Active(false),
        m_DualSingularValues(),
        m_DualSnapshotEnsemble(),
        m_DualLeftSingularVectors(),
        m_DualRightSingularVectors(),
        m_StateSingularValues(),
        m_StateSnapshotEnsemble(),
        m_StateLeftSingularVectors(),
        m_StateRightSingularVectors(),
        m_LeftHandSideSingularValues(),
        m_LeftHandSideSnapshotEnsemble(),
        m_LeftHandSideLeftSingularVectors(),
        m_LeftHandSideRightSingularVectors()
{
}

SpectralDecompositionMng::SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::BrandMatrixFactory> & brands_factory_,
                                                   const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & algebra_factory_,
                                                   const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                                                   const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_) :
        m_Factory(algebra_factory_),
        m_FullRankSVD(svd_),
        m_LowRankSVD(new trrom::BrandLowRankSVD(brands_factory_, algebra_factory_, svd_, ortho_)),
        m_DualEnergyThreshold(0.99),
        m_StateEnergyThreshold(0.99),
        m_LeftHandSideEnergyThreshold(0.99),
        m_DualSnapshotsCollected(false),
        m_StateSnapshotsCollected(false),
        m_LeftHandSideSnapshotsCollected(false),
        m_DualLowRankSVD_Active(false),
        m_StateLowRankSVD_Active(false),
        m_LeftHandSideLowRankSVD_Active(false),
        m_DualSingularValues(),
        m_DualSnapshotEnsemble(),
        m_DualLeftSingularVectors(),
        m_DualRightSingularVectors(),
        m_StateSingularValues(),
        m_StateSnapshotEnsemble(),
        m_StateLeftSingularVectors(),
        m_StateRightSingularVectors(),
        m_LeftHandSideSingularValues(),
        m_LeftHandSideSnapshotEnsemble(),
        m_LeftHandSideLeftSingularVectors(),
        m_LeftHandSideRightSingularVectors()
{
}

void SpectralDecompositionMng::allocateDualSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    try
    {
        if(input_.getNumCols() <= 0 || input_.getNumRows() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << " -> Input matrix has one of its dimension set to zero.\n";
            throw error.str().c_str();
        }
        m_DualSnapshotEnsemble = input_.create();
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void SpectralDecompositionMng::allocateStateSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    try
    {
        if(input_.getNumCols() <= 0 || input_.getNumRows() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << " -> Input matrix has one of its dimension set to zero.\n";
            throw error.str().c_str();
        }
        m_StateSnapshotEnsemble = input_.create();
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void SpectralDecompositionMng::allocateLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    try
    {
        if(input_.getNumCols() <= 0 || input_.getNumRows() <= 0)
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << " -> Input matrix has one of its dimension set to zero.\n";
            throw error.str().c_str();
        }
        m_LeftHandSideSnapshotEnsemble = input_.create();
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

SpectralDecompositionMng::~SpectralDecompositionMng()
{
}

int SpectralDecompositionMng::getNumDualSnapshots()
{
    return (m_DualSnapshotEnsemble->getNumCols());
}

int SpectralDecompositionMng::getNumStateSnapshots()
{
    return (m_StateSnapshotEnsemble->getNumCols());
}

int SpectralDecompositionMng::getNumLeftHandSideSnapshots()
{
    return (m_LeftHandSideSnapshotEnsemble->getNumCols());
}

double SpectralDecompositionMng::getDualBasisEnergyThreshold() const
{
    return (m_DualEnergyThreshold);
}

void SpectralDecompositionMng::setDualBasisEnergyThreshold(double input_)
{
    m_DualEnergyThreshold = input_;
}

double SpectralDecompositionMng::getStateBasisEnergyThreshold() const
{
    return (m_StateEnergyThreshold);
}

void SpectralDecompositionMng::setStateBasisEnergyThreshold(double input_)
{
    m_StateEnergyThreshold = input_;
}

double SpectralDecompositionMng::getLeftHandSideBasisEnergyThreshold() const
{
    return (m_LeftHandSideEnergyThreshold);
}

void SpectralDecompositionMng::setLeftHandSideBasisEnergyThreshold(double input_)
{
    m_LeftHandSideEnergyThreshold = input_;
}

bool SpectralDecompositionMng::areDualSnapshotsCollected() const
{
    return (m_DualSnapshotsCollected);
}

bool SpectralDecompositionMng::areStateSnapshotsCollected() const
{
    return (m_StateSnapshotsCollected);
}

bool SpectralDecompositionMng::areLeftHandSideSnapshotsCollected() const
{
    return (m_LeftHandSideSnapshotsCollected);
}

void SpectralDecompositionMng::storeDualSnapshot(const trrom::Vector<double> & input_)
{
    m_DualSnapshotEnsemble->insert(input_);
}

void SpectralDecompositionMng::storeStateSnapshot(const trrom::Vector<double> & input_)
{
    m_StateSnapshotEnsemble->insert(input_);
}

void SpectralDecompositionMng::storeLeftHandSideSnapshot(const trrom::Vector<double> & input_)
{
    m_LeftHandSideSnapshotEnsemble->insert(input_);
}

void SpectralDecompositionMng::computeDualOrthonormalBasis(std::tr1::shared_ptr<trrom::Matrix<double> > & basis_)
{
    int num_basis_vectors = trrom::energy(m_DualEnergyThreshold, *m_DualSingularValues);
    m_Factory->buildMultiVector(num_basis_vectors, m_DualSnapshotEnsemble->vector(0), basis_);
    trrom::properOrthogonalDecomposition(*m_DualSingularValues,
                                         *m_DualLeftSingularVectors,
                                         *m_DualSnapshotEnsemble,
                                         *basis_);
}

void SpectralDecompositionMng::computeStateOrthonormalBasis(std::tr1::shared_ptr<trrom::Matrix<double> > & basis_)
{
    int num_basis_vectors = trrom::energy(m_StateEnergyThreshold, *m_StateSingularValues);
    m_Factory->buildMultiVector(num_basis_vectors, m_StateSnapshotEnsemble->vector(0), basis_);
    trrom::properOrthogonalDecomposition(*m_StateSingularValues,
                                         *m_StateLeftSingularVectors,
                                         *m_StateSnapshotEnsemble,
                                         *basis_);
}

void SpectralDecompositionMng::computeLeftHandSideOrthonormalBasis(std::tr1::shared_ptr<trrom::Matrix<double> > & basis_)
{
    int num_basis_vectors = trrom::energy(m_LeftHandSideEnergyThreshold, *m_LeftHandSideSingularValues);
    m_Factory->buildMultiVector(num_basis_vectors, m_LeftHandSideSnapshotEnsemble->vector(0), basis_);
    trrom::properOrthogonalDecomposition(*m_LeftHandSideSingularValues,
                                         *m_LeftHandSideLeftSingularVectors,
                                         *m_LeftHandSideSnapshotEnsemble,
                                         *basis_);
}

void SpectralDecompositionMng::solveDualSingularValueDecomposition()
{
    if(m_DualLowRankSVD_Active == true)
    {
        m_LowRankSVD->solve(m_DualSnapshotEnsemble,
                            m_DualSingularValues,
                            m_DualLeftSingularVectors,
                            m_DualRightSingularVectors);
    }
    else
    {
        m_FullRankSVD->solve(m_DualSnapshotEnsemble,
                             m_DualSingularValues,
                             m_DualLeftSingularVectors,
                             m_DualRightSingularVectors);
        m_DualLowRankSVD_Active = true;
    }
}
void SpectralDecompositionMng::solveStateSingularValueDecomposition()
{
    if(m_StateLowRankSVD_Active == true)
    {
        m_LowRankSVD->solve(m_StateSnapshotEnsemble,
                            m_StateSingularValues,
                            m_StateLeftSingularVectors,
                            m_StateRightSingularVectors);
    }
    else
    {
        m_FullRankSVD->solve(m_StateSnapshotEnsemble,
                             m_StateSingularValues,
                             m_StateLeftSingularVectors,
                             m_StateRightSingularVectors);
        m_StateLowRankSVD_Active = true;
    }
}

void SpectralDecompositionMng::solveLeftHandSideSingularValueDecomposition()
{
    if(m_LeftHandSideLowRankSVD_Active == true)
    {
        m_LowRankSVD->solve(m_LeftHandSideSnapshotEnsemble,
                            m_LeftHandSideSingularValues,
                            m_LeftHandSideLeftSingularVectors,
                            m_LeftHandSideRightSingularVectors);
    }
    else
    {
        m_FullRankSVD->solve(m_LeftHandSideSnapshotEnsemble,
                             m_LeftHandSideSingularValues,
                             m_LeftHandSideLeftSingularVectors,
                             m_LeftHandSideRightSingularVectors);
        m_LeftHandSideLowRankSVD_Active = true;
    }
}

const std::tr1::shared_ptr<trrom::Vector<double> > & SpectralDecompositionMng::getDualSnapshot(int index_) const
{
    return (m_DualSnapshotEnsemble->vector(index_));
}

const std::tr1::shared_ptr<trrom::Vector<double> > & SpectralDecompositionMng::getStateSnapshot(int index_) const
{
    return (m_StateSnapshotEnsemble->vector(index_));
}

const std::tr1::shared_ptr<trrom::Vector<double> > & SpectralDecompositionMng::getLeftHandSideSnapshot(int index_) const
{
    return (m_LeftHandSideSnapshotEnsemble->vector(index_));
}

}
