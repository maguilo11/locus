/*
 * TRROM_SpectralDecomposition.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_SPECTRALDECOMPOSITION_HPP_
#define TRROM_SPECTRALDECOMPOSITION_HPP_

#include <tr1/memory>

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class SpectralDecomposition
{
public:
    virtual ~SpectralDecomposition()
    {
    }
    virtual void solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_,
                       std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_) = 0;
};

}

#endif /* TRROM_SPECTRALDECOMPOSITION_HPP_ */
