/*
 * TRROM_BrandLowRankSVD.hpp
 *
 *  Created on: Dec 7, 2016
 *      Author: maguilo
 */

#ifndef TRROM_BRANDLOWRANKSVD_HPP_
#define TRROM_BRANDLOWRANKSVD_HPP_

#include "TRROM_LowRankSpectralDecomposition.hpp"

namespace trrom
{

class BrandMatrixFactory;
class SpectralDecomposition;
class OrthogonalFactorization;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class BrandLowRankSVD : public trrom::LowRankSpectralDecomposition
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a BrandLowRankSVD object
     *    \param In
     *          factory_: instance to a derived class from trrom::BrandMatrixFactory
     *    \param In
     *          svd_: instance to a derived class from trrom::SpectralDecomposition
     *    \param In
     *          ortho_: instance to a derived class from trrom::OrthogonalFactorization
     * \return Reference to BrandLowRankSVD.
     **/
    BrandLowRankSVD(const std::tr1::shared_ptr<trrom::BrandMatrixFactory> & factory_,
                    const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                    const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_);
    //! BrandLowRankSVD destructor.
    virtual ~BrandLowRankSVD();
    //@}

    /*! Solves low rank singular value decomposition problem based on Brand's algorithm,
     * M. Brand, "Fast low-rank modifications of the thin singular value decomposition"
     * Parameters:
     *    \param In
     *          data_set_: m-by-n matrix.
     *    \param Out
     *          singular_values_: r-by-1 vector of singular values.
     *    \param Out
     *          left_singular_vectors_: m-by-r matrix of left singular vectors.
     *    \param Out
     *          right_singular_vectors_: n-by-r matrix of right singular vectors.
     **/
    void solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_set_,
               std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_);

private:
    std::tr1::shared_ptr<trrom::BrandMatrixFactory> m_MatrixFactory;
    std::tr1::shared_ptr<trrom::SpectralDecomposition> m_SpectralMethod;
    std::tr1::shared_ptr<trrom::OrthogonalFactorization> m_OrthoFactorization;

private:
    BrandLowRankSVD(const trrom::BrandLowRankSVD &);
    trrom::BrandLowRankSVD & operator=(const trrom::BrandLowRankSVD &);
};

}

#endif /* TRROM_BRANDLOWRANKSVD_HPP_ */
