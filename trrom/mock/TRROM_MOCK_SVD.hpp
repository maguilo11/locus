/*
 * TRROM_MOCK_SVD.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_SVD_HPP_
#define TRROM_SVD_HPP_

#include "TRROM_SpectralDecomposition.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

namespace mock
{

class SVD : public trrom::SpectralDecomposition
{
public:
    SVD();
    virtual ~SVD();

    void solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_,
               std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_);

private:
    std::tr1::shared_ptr< trrom::Vector<double> > m_Vector;

private:
    SVD(const mock::SVD &);
    mock::SVD & operator=(const mock::SVD &);
};

}

}

#endif /* TRROM_SVD_HPP_ */
