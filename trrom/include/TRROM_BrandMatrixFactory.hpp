/*
 * TRROM_BrandMatrixFactory.hpp
 *
 *  Created on: Dec 6, 2016
 *      Author: maguilo
 */

#ifndef TRROM_BRANDMATRIXFACTORY_HPP_
#define TRROM_BRANDMATRIXFACTORY_HPP_

#include <memory>

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class BrandMatrixFactory
{
public:
    //! @name Constructors/destructors
    //@{
    virtual ~BrandMatrixFactory()
    {
    }
    //@}

    /*! Build (r+k)-by-(r+k) matrix K=\[\Sigma M; 0 R\], where K\in\mathbb{R}^{(r+k)\times(r+k)} from
     * input matrices \Sigma, M, and R. Here, r denotes the length of input vector of singular values
     * and k denotes the number of new snapshots stored since the last low-rank singular value
     * decomposition update was performed.
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
    virtual void buildMatrixK(const std::shared_ptr<trrom::Vector<double> > & sigma_,
                              const std::shared_ptr<trrom::Matrix<double> > & M_,
                              const std::shared_ptr<trrom::Matrix<double> > & R_,
                              std::shared_ptr<trrom::Matrix<double> > & K_) = 0;
    /*! Build m-by-k matrix \bar{U} = \[Uo P\]Ur from input matrices Uo\in\mathbb{R}^{m \times r},
     * P\in\mathbb{R}^{m \times k}, and C\in\mathbb{R}^{(r+k)\times(r+k)}. Here, r denotes the length
     * of input vector of singular values, k denotes the number of new snapshots stored since the last
     * low-rank singular value decomposition update was performed, and m denotes the number of degrees
     * of freedom.
     * Parameters:
     *    \param In
     *          Uo_: m-by-r matrix containing the current left singular vectors.
     *    \param In
     *          Q_: m-by-k unitary matrix obtained from the orthogonal decomposition of matrix \bar{P}.
     *    \param In
     *          Ur_: (r+k)-by-(r+k) matrix containing the left singular vectors of matrix K=\[\Sigma M; 0 R\].
     *    \param Out
     *          Un_: m-by-(r+k) matrix containing the updated left singular vectors.
     **/
    virtual void buildMatrixUbar(const std::shared_ptr<trrom::Matrix<double> > & Uo_,
                                 const std::shared_ptr<trrom::Matrix<double> > & Q_,
                                 const std::shared_ptr<trrom::Matrix<double> > & Ur_,
                                 std::shared_ptr<trrom::Matrix<double> > & Un_) = 0;
    /*! Build m-by-k matrix \bar{V} = \[Vo 0; 0 I\]Vr from input matrices I\in\mathbb{R}^{k \times k},
     * Vo\in\mathbb{R}^{n \times r}, Vr\in\mathbb{R}^{(r+k)\times(r+k)}. Here, n denotes the current
     * number of snapshots, k denotes the new number of snapshots stored since the last low-rank
     * singular value decomposition update was performed, and r denotes the current number of singular
     * values.
     * Parameters:
     *    \param In
     *          Vo_: n-by-r matrix containing the current right singular vectors.
     *    \param In
     *          Vr_: (r+k)-by-(r+k) matrix containing the right singular vectors of matrix K=\[\Sigma M; 0 R\].
     *    \param Out
     *          Vn_: n-by-(r+k) matrix containing the updated left singular vectors.
     **/
    virtual void buildMatrixVbar(const std::shared_ptr<trrom::Matrix<double> > & Vo_,
                                 const std::shared_ptr<trrom::Matrix<double> > & Vr_,
                                 std::shared_ptr<trrom::Matrix<double> > & Vn_) = 0;
};

}

#endif /* TRROM_BRANDMATRIXFACTORY_HPP_ */
