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

enum ortho_factorization_t
{
    CLASSICAL_GRAM_SCHMIDT_QR = 1, MODIFIED_GRAM_SCHMIDT_QR = 2, HOUSEHOLDER_QR = 3, USER_DEFINED_QR = 4
};

}

template<typename ScalarType>
class Matrix;

class OrthogonalFactorization
{
public:
    virtual ~OrthogonalFactorization()
    {
    }

    virtual trrom::types::ortho_factorization_t type() const = 0;
    virtual void factorize(const trrom::Matrix<double> & input_,
                           trrom::Matrix<double> & Q_,
                           trrom::Matrix<double> & R_) = 0;
};

}

#endif /* TRROM_ORTHOGONALFACTORIZATION_HPP_ */
