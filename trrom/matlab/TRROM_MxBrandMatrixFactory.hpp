/*
 * TRROM_MxBrandMatrixFactory.hpp
 *
 *  Created on: Dec 6, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXBRANDMATRIXFACTORY_HPP_
#define TRROM_MXBRANDMATRIXFACTORY_HPP_

#include "TRROM_BrandMatrixFactory.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class MxBrandMatrixFactory : public trrom::BrandMatrixFactory
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxBrandMatrixFactory object
     * \return Reference to MxBrandMatrixFactory.
     *
     **/
    MxBrandMatrixFactory();
    //! MxVector destructor.
    virtual ~MxBrandMatrixFactory();
    //@}

    /*! MEX interface that enables construction of (r+k)-by-(r+k) matrix K=\[\Sigma M; 0 R\], where
     * K\in\mathbb{R}^{(r+k)\times(r+k)} from input matrices \Sigma, M, and R. Here, r denotes the
     * length of input vector of singular values and k denotes the number of new snapshots stored
     * since the last low-rank singular value decomposition update was performed.
     * Parameters:
     *    \param In
     *          sigma_: r-by-1 vector of singular values, i.e. \Sigma\in\mathbb{R}^{r}.
     *    \param In
     *          M_: r-by-k matrix M=U^{\intercal}Y\in\mathbb{R}^{{r}\times{k}}, where U denotes the
     *          current set of left singular values and Y\in\mathbb{R}^{m \times k} denotes the new
     *          set of snapshots.
     *    \param In
     *          R_: k-by-k matrix R\in\mathbb{k \times k} denotes an upper triangular matrix obtained
     *          by performing a orthogonal decomposition (i.e. QR decomposition) of matrix \bar{P}=
     *          Y-UM, where \bar{P}\in\mathbb{R}^{m \times k}.
     *    \param Out
     *          K_: (r+k)-by-(r+k) matrix.
     **/
    void buildMatrixK(const std::tr1::shared_ptr<trrom::Vector<double> > & sigma_,
                      const std::tr1::shared_ptr<trrom::Matrix<double> > & M_,
                      const std::tr1::shared_ptr<trrom::Matrix<double> > & R_,
                      std::tr1::shared_ptr<trrom::Matrix<double> > & K_);
    /*! MEX interface that enables construction of m-by-k matrix \bar{U} = \[Uo P\]Ur from input
     * matrices Uo\in\mathbb{R}^{m \times r}, P\in\mathbb{R}^{m \times k}, and C\in\mathbb{R}^{(r+k)
     * \times(r+k)}. Here, r denotes the length of input vector of singular values, k denotes the number
     * of new snapshots stored since the last low-rank singular value decomposition update was performed,
     * and m denotes the number of degrees of freedom.
     * Parameters:
     *    \param In
     *          Uc_: m-by-r matrix containing the current left singular vectors.
     *    \param In
     *          Q_: m-by-k unitary matrix obtained from the orthogonal decomposition of matrix \bar{P}.
     *    \param In
     *          Ur_: (r+k)-by-(r+k) matrix containing the left singular vectors of matrix K=\[\Sigma M; 0 R\].
     *    \param Out
     *          Un_: m-by-(r+k) matrix containing the updated left singular vectors.
     **/
    void buildMatrixUbar(const std::tr1::shared_ptr<trrom::Matrix<double> > & Uc_,
                         const std::tr1::shared_ptr<trrom::Matrix<double> > & Q_,
                         const std::tr1::shared_ptr<trrom::Matrix<double> > & Ur_,
                         std::tr1::shared_ptr<trrom::Matrix<double> > & Un_);
    /*! MEX interface that enables construction of m-by-k matrix \bar{V} = \[Vo 0; 0 I\]Vr from input
     * matrices I\in\mathbb{R}^{k \times k}, Vo\in\mathbb{R}^{n \times r}, Vr\in\mathbb{R}^{(r+k)
     * \times(r+k)}. Here, n denotes the current number of snapshots, k denotes the new number of
     * snapshots stored since the last low-rank singular value decomposition update was performed,
     * and r denotes the current number of singular values.
     * Parameters:
     *    \param In
     *          Vc_: n-by-r matrix containing the current right singular vectors.
     *    \param In
     *          Vr_: (r+k)-by-(r+k) matrix containing the right singular vectors of matrix K=\[\Sigma M; 0 R\].
     *    \param Out
     *          Vn_: n-by-(r+k) matrix containing the updated right singular vectors.
     **/
    void buildMatrixVbar(const std::tr1::shared_ptr<trrom::Matrix<double> > & Vc_,
                         const std::tr1::shared_ptr<trrom::Matrix<double> > & Vr_,
                         std::tr1::shared_ptr<trrom::Matrix<double> > & Vn_);

private:
    MxBrandMatrixFactory(const trrom::MxBrandMatrixFactory &);
    trrom::MxBrandMatrixFactory & operator=(const trrom::MxBrandMatrixFactory & rhs_);
};

}

#endif /* TRROM_MXBRANDMATRIXFACTORY_HPP_ */
