/*
 * TRROM_MxBrandLowRankSVD.hpp
 *
 *  Created on: Dec 7, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXBRANDLOWRANKSVD_HPP_
#define TRROM_MXBRANDLOWRANKSVD_HPP_

#include "TRROM_LowRankSpectralDecomposition.hpp"

namespace trrom
{

class BrandLowRankSVD;
class MxBrandMatrixFactory;
class MxLinearAlgebraFactory;
class MxOrthogonalDecomposition;
class MxSingularValueDecomposition;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class MxBrandLowRankSVD : public trrom::LowRankSpectralDecomposition
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxBrandLowRankSVD object
     * \return Reference to MxBrandLowRankSVD.
     **/
    MxBrandLowRankSVD();
    //! MxBrandLowRankSVD destructor.
    virtual ~MxBrandLowRankSVD();
    //@}

    /*! MEX interface for Brand's low rank singular value decomposition, M. Brand,
     * "Fast low-rank modifications of the thin singular value decomposition"
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
    std::tr1::shared_ptr<trrom::BrandLowRankSVD> m_Algorithm;
    std::tr1::shared_ptr<trrom::MxBrandMatrixFactory> m_BrandFactory;
    std::tr1::shared_ptr<trrom::MxLinearAlgebraFactory> m_AlgebraFactory;
    std::tr1::shared_ptr<trrom::MxSingularValueDecomposition> m_SpectralMethod;
    std::tr1::shared_ptr<trrom::MxOrthogonalDecomposition> m_OrthoFactorization;

private:
    MxBrandLowRankSVD(const trrom::MxBrandLowRankSVD &);
    trrom::MxBrandLowRankSVD & operator=(const trrom::MxBrandLowRankSVD &);
};

}

#endif /* TRROM_MXBRANDLOWRANKSVD_HPP_ */
