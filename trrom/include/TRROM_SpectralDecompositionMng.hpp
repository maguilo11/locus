/*
 * TRROM_SpectralDecompositionMng.hpp
 *
 *  Created on: Aug 17, 2016
 *      Author: maguilo
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

class BrandMatrixFactory;
class LinearAlgebraFactory;
class SpectralDecomposition;
class OrthogonalFactorization;
class LowRankSpectralDecomposition;

class SpectralDecompositionMng
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a SpectralDecompositionMng object. This constructor is mostly used
     * for unit testing purposes.
     * return Reference to SpectralDecompositionMng.
     **/
    SpectralDecompositionMng();
    /*!
     * Creates a SpectralDecompositionMng object
     *    \param In
     *          full_rank_svd_: instance to a derived class from trrom::SpectralDecomposition
     *    \param In
     *          low_rank_svd_: instance to a derived class from trrom::LowRankSpectralDecomposition
     * \return Reference to SpectralDecompositionMng.
     **/
    SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & algebra_factory_,
                             const std::tr1::shared_ptr<trrom::SpectralDecomposition> & full_rank_svd_,
                             const std::tr1::shared_ptr<trrom::LowRankSpectralDecomposition> & low_rank_svd_);
    /*!
     * Creates a SpectralDecompositionMng object
     *    \param In
     *          factory_: instance to a derived class from trrom::BrandMatrixFactory
     *    \param In
     *          svd_: instance to a derived class from trrom::SpectralDecomposition
     *    \param In
     *          ortho_: instance to a derived class from trrom::OrthogonalFactorization
     * \return Reference to SpectralDecompositionMng.
     **/
    SpectralDecompositionMng(const std::tr1::shared_ptr<trrom::BrandMatrixFactory> & brands_factory_,
                             const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & algebra_factory_,
                             const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                             const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_);
    //! SpectralDecompositionMng destructor.
    ~SpectralDecompositionMng();
    //@}

    void allocateDualSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateStateSnapshotEnsemble(const trrom::Matrix<double> & input_);
    void allocateLeftHandSideSnapshotEnsemble(const trrom::Matrix<double> & input_);

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

    void computeDualOrthonormalBasis(std::tr1::shared_ptr<trrom::Matrix<double> > & basis_);
    void computeStateOrthonormalBasis(std::tr1::shared_ptr<trrom::Matrix<double> > & basis_);
    void computeLeftHandSideOrthonormalBasis(std::tr1::shared_ptr<trrom::Matrix<double> > & basis_);

    void solveDualSingularValueDecomposition();
    void solveStateSingularValueDecomposition();
    void solveLeftHandSideSingularValueDecomposition();

    const std::tr1::shared_ptr<trrom::Vector<double> > & getDualSnapshot(int index_) const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & getStateSnapshot(int index_) const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & getLeftHandSideSnapshot(int index_) const;

private:
    std::tr1::shared_ptr<trrom::LinearAlgebraFactory> m_Factory;
    std::tr1::shared_ptr<trrom::SpectralDecomposition> m_FullRankSVD;
    std::tr1::shared_ptr<trrom::LowRankSpectralDecomposition> m_LowRankSVD;

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
