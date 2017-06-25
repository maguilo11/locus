/*
 * TRROM_MxSingularValueDecomposition.hpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXSINGULARVALUEDECOMPOSITION_HPP_
#define TRROM_MXSINGULARVALUEDECOMPOSITION_HPP_

#include "TRROM_SpectralDecomposition.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class MxSingularValueDecomposition : public trrom::SpectralDecomposition
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxSingularValueDecomposition object
     * \return Reference to MxSingularValueDecomposition.
     *
     **/
    MxSingularValueDecomposition();
    //! MxSingularValueDecomposition destructor.
    virtual ~MxSingularValueDecomposition();
    //@}

    /*! Perform singular value decomposition (SVD)
     *  Parameters:
     *    \param In
     *          data_: const m-by-n matrix
     *    \param Out
     *          singular_values_: singular values
     *    \param Out
     *          left_singular_vectors_: left singular vectors
     *    \param Out
     *          right_singular_vectors_: right singular vectors
     **/
    void solve(const std::shared_ptr<trrom::Matrix<double> > & data_,
               std::shared_ptr<trrom::Vector<double> > & singular_values_,
               std::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
               std::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_);

private:
    MxSingularValueDecomposition(const trrom::MxSingularValueDecomposition &);
    trrom::MxSingularValueDecomposition & operator=(const trrom::MxSingularValueDecomposition & rhs_);
};

}

#endif /* TRROM_MXSINGULARVALUEDECOMPOSITION_HPP_ */
