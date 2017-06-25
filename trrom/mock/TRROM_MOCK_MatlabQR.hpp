/*
 * TRROM_MOCK_MatlabQR.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_MOCK_MATLABQR_HPP_
#define TRROM_MOCK_MATLABQR_HPP_

#include <memory>

#include "TRROM_OrthogonalFactorization.hpp"

namespace trrom
{
template<typename ScalarType>
class Matrix;

namespace mock
{

class MatlabQR : public trrom::OrthogonalFactorization
{
public:
    MatlabQR();
    virtual ~MatlabQR();

    trrom::types::ortho_factorization_t type() const;
    void factorize(const std::shared_ptr<trrom::Matrix<double> > & input_,
                   std::shared_ptr<trrom::Matrix<double> > & Q_,
                   std::shared_ptr<trrom::Matrix<double> > & R_);

private:
    MatlabQR(const mock::MatlabQR &);
    mock::MatlabQR & operator=(const mock::MatlabQR &);
};

}

}

#endif /* TRROM_MOCK_MATLABQR_HPP_ */
