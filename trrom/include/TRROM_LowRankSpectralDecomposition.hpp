/*
 * TRROM_LowRankSpectralDecomposition.hpp
 *
 *  Created on: Dec 7, 2016
 *      Author: maguilo
 */

#ifndef TRROM_LOWRANKSPECTRALDECOMPOSITION_HPP_
#define TRROM_LOWRANKSPECTRALDECOMPOSITION_HPP_

#include <memory>

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class LowRankSpectralDecomposition
{
public:
    //! @name Constructors/destructors
    //@{
    //! LowRankSpectralDecomposition destructor
    virtual ~LowRankSpectralDecomposition()
    {
    }
    //@}

    /*! Pure abstract interface for low-rank spectral decomposition algorithms
     * Parameters:
     *    \param In
     *          data_: m-by-n matrix.
     *    \param Out
     *          singular_values_: r-by-1 vector of singular values.
     *    \param Out
     *          left_singular_vectors_: m-by-r matrix of left singular vectors.
     *    \param Out
     *          right_singular_vectors_: n-by-r matrix of right singular vectors.
     **/
    virtual void solve(const std::shared_ptr<trrom::Matrix<double> > & data_,
                       std::shared_ptr<trrom::Vector<double> > & singular_values_,
                       std::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                       std::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_) = 0;
};

}

#endif /* TRROM_LOWRANKSPECTRALDECOMPOSITION_HPP_ */
