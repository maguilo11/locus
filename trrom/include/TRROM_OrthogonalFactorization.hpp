/*
 * TRROM_OrthogonalFactorization.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_ORTHOGONALFACTORIZATION_HPP_
#define TRROM_ORTHOGONALFACTORIZATION_HPP_

namespace trrom
{

namespace types
{

/*!
 * ENUM use to denote the multiple type of orthogonal factorization available to users.
 * Users have the flexibility to define a custom orthognal factorization routine, In such
 * case, users are required to use the USER_DEFINED_QR flag.
 **/
enum ortho_factorization_t
{
    CLASSICAL_GRAM_SCHMIDT_QR = 1, MODIFIED_GRAM_SCHMIDT_QR = 2, HOUSEHOLDER_QR = 3, MATLAB_QR = 4, USER_DEFINED_QR = 5
};

}

template<typename ScalarType>
class Matrix;

class OrthogonalFactorization
{
public:
    //! OrthogonalFactorization destructor.
    virtual ~OrthogonalFactorization()
    {
    }
    //! Returns the type of the orthogonal factorization method.
    virtual trrom::types::ortho_factorization_t type() const = 0;
    /*! Performs orthogonal-triangular decomposition
     *  Parameters:
     *    \param In
     *          input_: const m-by-n matrix
     *    \param Out
     *          Q_: m-by-n unitary matrix
     *    \param Out
     *          R_: n-by-n upper triangular matrix
     **/
    virtual void factorize(const trrom::Matrix<double> & input_,
                           trrom::Matrix<double> & Q_,
                           trrom::Matrix<double> & R_) = 0;
};

}

#endif /* TRROM_ORTHOGONALFACTORIZATION_HPP_ */
