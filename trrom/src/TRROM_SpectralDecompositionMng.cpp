/*
 * TRROM_SpectralDecompositionMng.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <cassert>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_LowRankSVD.hpp"
#include "TRROM_ReducedBasis.hpp"
#include "TRROM_SpectralDecomposition.hpp"
#include "TRROM_OrthogonalFactorization.hpp"
#include "TRROM_SpectralDecompositionMng.hpp"

namespace trrom
{

SpectralDecompositionMng::SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_) :
        m_LowRankSVD(new trrom::LowRankSVD(svd_)),
        m_FullRankSVD(svd_),
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

SpectralDecompositionMng::SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                                                   const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_) :
        m_LowRankSVD(new trrom::LowRankSVD(svd_, ortho_)),
        m_FullRankSVD(svd_),
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

SpectralDecompositionMng::~SpectralDecompositionMng()
{
}

int SpectralDecompositionMng::getNumDualSnapshots()
{
    return (m_DualSnapshotEnsemble->numCols());
}

int SpectralDecompositionMng::getNumStateSnapshots()
{
    return (m_StateSnapshotEnsemble->numCols());
}

int SpectralDecompositionMng::getNumLeftHandSideSnapshots()
{
    return (m_LeftHandSideSnapshotEnsemble->numCols());
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

void SpectralDecompositionMng::setDualSingularValues(const trrom::Vector<double> & input_)
{
    if(m_DualSingularValues.use_count() <= 0)
    {
        m_DualSingularValues = input_.create();
    }
    m_DualSingularValues->copy(input_);
}

void SpectralDecompositionMng::setDualSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    if(m_DualSnapshotEnsemble.use_count() <= 0)
    {
        m_DualSnapshotEnsemble = input_.create();
    }
    m_DualSnapshotEnsemble->copy(input_);
}

void SpectralDecompositionMng::setDualLeftSingularVectors(const trrom::Matrix<double> & input_)
{
    if(m_DualLeftSingularVectors.use_count() <= 0)
    {
        m_DualLeftSingularVectors = input_.create();
    }
    m_DualLeftSingularVectors->copy(input_);
}

void SpectralDecompositionMng::setStateSingularValues(const trrom::Vector<double> & input_)
{
    if(m_StateSingularValues.use_count() <= 0)
    {
        m_StateSingularValues = input_.create();
    }
    m_StateSingularValues->copy(input_);
}

void SpectralDecompositionMng::setStateSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    if(m_StateSnapshotEnsemble.use_count() <= 0)
    {
        m_StateSnapshotEnsemble = input_.create();
    }
    m_StateSnapshotEnsemble->copy(input_);
}

void SpectralDecompositionMng::setStateLeftSingularVectors(const trrom::Matrix<double> & input_)
{
    if(m_StateLeftSingularVectors.use_count() <= 0)
    {
        m_StateLeftSingularVectors = input_.create();
    }
    m_StateLeftSingularVectors->copy(input_);
}

void SpectralDecompositionMng::setLeftHandSideSingularValues(const trrom::Vector<double> & input_)
{
    if(m_LeftHandSideSingularValues.use_count() <= 0)
    {
        m_LeftHandSideSingularValues = input_.create();
    }
    m_LeftHandSideSingularValues->copy(input_);
}

void SpectralDecompositionMng::setLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    if(m_LeftHandSideSnapshotEnsemble.use_count() <= 0)
    {
        m_LeftHandSideSnapshotEnsemble = input_.create();
    }
    m_LeftHandSideSnapshotEnsemble->copy(input_);
}

void SpectralDecompositionMng::setLeftHandSideLeftSingularVectors(const trrom::Matrix<double> & input_)
{
    if(m_LeftHandSideLeftSingularVectors.use_count() <= 0)
    {
        m_LeftHandSideLeftSingularVectors = input_.create();
    }
    m_LeftHandSideLeftSingularVectors->copy(input_);
}

void SpectralDecompositionMng::computeDualOrthonormalBasis(trrom::Matrix<double> & basis_)
{
    trrom::properOrthogonalDecomposition(m_DualEnergyThreshold,
                                         *m_DualSingularValues,
                                         *m_DualLeftSingularVectors,
                                         *m_DualSnapshotEnsemble,
                                         basis_);
}
void SpectralDecompositionMng::computeStateOrthonormalBasis(trrom::Matrix<double> & basis_)
{
    trrom::properOrthogonalDecomposition(m_StateEnergyThreshold,
                                         *m_StateSingularValues,
                                         *m_StateLeftSingularVectors,
                                         *m_StateSnapshotEnsemble,
                                         basis_);
}
void SpectralDecompositionMng::computeLeftHandSideOrthonormalBasis(trrom::Matrix<double> & basis_)
{
    trrom::properOrthogonalDecomposition(m_LeftHandSideEnergyThreshold,
                                         *m_LeftHandSideSingularValues,
                                         *m_LeftHandSideLeftSingularVectors,
                                         *m_LeftHandSideSnapshotEnsemble,
                                         basis_);
}

void SpectralDecompositionMng::allocateDualSingularValues(const trrom::Vector<double> & input_)
{
    m_DualSingularValues = input_.create();
}

void SpectralDecompositionMng::allocateDualSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    m_DualSnapshotEnsemble = input_.create();
}

void SpectralDecompositionMng::allocateDualLeftSingularVectors(const trrom::Matrix<double> & input_)
{
    m_DualLeftSingularVectors = input_.create();
}

void SpectralDecompositionMng::allocateDualRightSingularVectors(const trrom::Matrix<double> & input_)
{
    m_DualRightSingularVectors = input_.create();
}

void SpectralDecompositionMng::allocateStateSingularValues(const trrom::Vector<double> & input_)
{
    m_StateSingularValues = input_.create();
}

void SpectralDecompositionMng::allocateStateSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    m_StateSnapshotEnsemble = input_.create();
}

void SpectralDecompositionMng::allocateStateLeftSingularVectors(const trrom::Matrix<double> & input_)
{
    m_StateLeftSingularVectors = input_.create();
}

void SpectralDecompositionMng::allocateStateRightSingularVectors(const trrom::Matrix<double> & input_)
{
    m_StateRightSingularVectors = input_.create();
}

void SpectralDecompositionMng::allocateLeftHandSideSingularValues(const trrom::Vector<double> & input_)
{
    m_LeftHandSideSingularValues = input_.create();
}

void SpectralDecompositionMng::allocateLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_)
{
    m_LeftHandSideSnapshotEnsemble = input_.create();
}

void SpectralDecompositionMng::allocateLeftHandSideLeftSingularVectors(const trrom::Matrix<double> & input_)
{
    m_LeftHandSideLeftSingularVectors = input_.create();
}

void SpectralDecompositionMng::allocateLeftHandSideRightSingularVectors(const trrom::Matrix<double> & input_)
{
    m_LeftHandSideRightSingularVectors = input_.create();
}

void SpectralDecompositionMng::solveDualSVD()
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
void SpectralDecompositionMng::solveStateSVD()
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

void SpectralDecompositionMng::solveLeftHandSideSVD()
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

}
