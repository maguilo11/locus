/*
 * TRROM_ReducedBasis.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASIS_HPP_
#define TRROM_REDUCEDBASIS_HPP_

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

void properOrthogonalDecomposition(const double & threshold_,
                                   const trrom::Vector<double> & singular_values_,
                                   const trrom::Matrix<double> & left_singular_vectors_,
                                   const trrom::Matrix<double> & snapshots_,
                                   trrom::Matrix<double> & basis_);

}

#endif /* TRROM_REDUCEDBASIS_HPP_ */
