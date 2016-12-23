/*
 * TRROM_MxOrthogonalDecomposition.hpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXORTHOGONALDECOMPOSITION_HPP_
#define TRROM_MXORTHOGONALDECOMPOSITION_HPP_

#include "TRROM_OrthogonalFactorization.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;

class MxMatrix;

class MxOrthogonalDecomposition : public trrom::OrthogonalFactorization
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxOrthogonalDecomposition object
     * \return Reference to MxOrthogonalDecomposition.
     *
     **/
    MxOrthogonalDecomposition();
    //! MxOrthogonalDecomposition destructor.
    virtual ~MxOrthogonalDecomposition();
    //@}

    //! Returns the type of the orthogonal factorization method.
    trrom::types::ortho_factorization_t type() const;
    /*! Performs orthogonal-triangular decomposition
     *  Parameters:
     *    \param In
     *          input_: const m-by-n matrix
     *    \param Out
     *          Q_: m-by-n unitary matrix
     *    \param Out
     *          R_: n-by-n upper triangular matrix
     **/
    void factorize(const std::tr1::shared_ptr<trrom::Matrix<double> > & input_,
                   std::tr1::shared_ptr<trrom::Matrix<double> > & Q_,
                   std::tr1::shared_ptr<trrom::Matrix<double> > & R_);

    //! Returns permutation matrix
    const trrom::MxMatrix & getPermutationData() const;

private:
    std::tr1::shared_ptr<MxMatrix> m_PermutationData;

private:
    MxOrthogonalDecomposition(const trrom::MxOrthogonalDecomposition &);
    trrom::MxOrthogonalDecomposition & operator=(const trrom::MxOrthogonalDecomposition & rhs_);
};

}

#endif /* TRROM_MXORTHOGONALDECOMPOSITION_HPP_ */
