/*
 * TRROM_SpectralDecompositionMng.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_SPECTRALDECOMPOSITIONMNG_HPP_
#define TRROM_SPECTRALDECOMPOSITIONMNG_HPP_

#include <tr1/memory>

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class LowRankSVD;
class SpectralDecomposition;
class OrthogonalFactorization;

class SpectralDecompositionMng
{
public:
    explicit SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_);
    SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                             const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_);
    ~SpectralDecompositionMng();

    int getNumDualSnapshots();
    int getNumStateSnapshots();
    int getNumLeftHandSideSnapshots();

    double getDualBasisEnergyThreshold() const;
    void setDualBasisEnergyThreshold(double input_);
    double getStateBasisEnergyThreshold() const;
    void setStateBasisEnergyThreshold(double input_);
    double getLeftHandSideBasisEnergyThreshold() const;
    void setLeftHandSideBasisEnergyThreshold(double input_);

    bool areDualSnapshotsCollected() const;
    bool areStateSnapshotsCollected() const;
    bool areLeftHandSideSnapshotsCollected() const;

    void storeDualSnapshot(const trrom::Vector<double> & input_);
    void storeStateSnapshot(const trrom::Vector<double> & input_);
    void storeLeftHandSideSnapshot(const trrom::Vector<double> & input_);

    void setDualSingularValues(const trrom::Vector<double> & input_);
    void setDualSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void setDualLeftSingularVectors(const trrom::Matrix<double> & input_);

    void setStateSingularValues(const trrom::Vector<double> & input_);
    void setStateSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void setStateLeftSingularVectors(const trrom::Matrix<double> & input_);

    void setLeftHandSideSingularValues(const trrom::Vector<double> & input_);
    void setLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void setLeftHandSideLeftSingularVectors(const trrom::Matrix<double> & input_);

    void computeDualOrthonormalBasis(trrom::Matrix<double> & basis_);
    void computeStateOrthonormalBasis(trrom::Matrix<double> & basis_);
    void computeLeftHandSideOrthonormalBasis(trrom::Matrix<double> & basis_);

    void allocateDualSingularValues(const trrom::Vector<double> & input_);
    void allocateDualSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateDualLeftSingularVectors(const trrom::Matrix<double> & input_);
    void allocateDualRightSingularVectors(const trrom::Matrix<double> & input_);

    void allocateStateSingularValues(const trrom::Vector<double> & input_);
    void allocateStateSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateStateLeftSingularVectors(const trrom::Matrix<double> & input_);
    void allocateStateRightSingularVectors(const trrom::Matrix<double> & input_);

    void allocateLeftHandSideSingularValues(const trrom::Vector<double> & input_);
    void allocateLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateLeftHandSideLeftSingularVectors(const trrom::Matrix<double> & input_);
    void allocateLeftHandSideRightSingularVectors(const trrom::Matrix<double> & input_);

    void solveDualSVD();
    void solveStateSVD();
    void solveLeftHandSideSVD();

private:
    std::tr1::shared_ptr<trrom::LowRankSVD> m_LowRankSVD;
    std::tr1::shared_ptr<trrom::SpectralDecomposition> m_FullRankSVD;

    double m_DualEnergyThreshold;
    double m_StateEnergyThreshold;
    double m_LeftHandSideEnergyThreshold;

    bool m_DualSnapshotsCollected;
    bool m_StateSnapshotsCollected;
    bool m_LeftHandSideSnapshotsCollected;

    bool m_DualLowRankSVD_Active;
    bool m_StateLowRankSVD_Active;
    bool m_LeftHandSideLowRankSVD_Active;

    std::tr1::shared_ptr<trrom::Vector<double> > m_DualSingularValues;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_DualSnapshotEnsemble;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_DualLeftSingularVectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_DualRightSingularVectors;

    std::tr1::shared_ptr<trrom::Vector<double> > m_StateSingularValues;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_StateSnapshotEnsemble;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_StateLeftSingularVectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_StateRightSingularVectors;

    std::tr1::shared_ptr<trrom::Vector<double> > m_LeftHandSideSingularValues;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideSnapshotEnsemble;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideLeftSingularVectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > m_LeftHandSideRightSingularVectors;

private:
    SpectralDecompositionMng(const trrom::SpectralDecompositionMng &);
    trrom::SpectralDecompositionMng & operator=(const trrom::SpectralDecompositionMng &);
};

}

#endif /* TRROM_SPECTRALDECOMPOSITIONMNG_HPP_ */
