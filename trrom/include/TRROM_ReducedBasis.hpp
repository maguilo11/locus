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

/*!
 * Returns the number of basis vectors to be computed based on the user-defined energy
 * threshold, i.e. index = \sum_{index=1}^{N}\Sigma_index > threshold, where N denotes
 * the number of singular values.
 * Parameters:
 *      \param In
 *          threshold_: energy threshold
 *      \param In
 *          singular_values_: current set of singular values
 **/

int energy(const double & threshold_, const trrom::Vector<double> & singular_values_);

/*!
 * Proper Orthogonal Decomposition (POD) Method
 * Parameters:
 *      \param In
 *          singular_values_ (\Sigma): current set of singular values
 *      \param In
 *          singular_vectors_ (\Psi): current set of singular vectors
 *      \param In
 *          snapshots_ (\mathbf{U}), current snapshot ensemble
 *      \param In
 *          basis_ (\Phi), orthonormal basis
 **/
void properOrthogonalDecomposition(const trrom::Vector<double> & singular_values_,
                                   const trrom::Matrix<double> & left_singular_vectors_,
                                   const trrom::Matrix<double> & snapshots_,
                                   trrom::Matrix<double> & basis_);

}

#endif /* TRROM_REDUCEDBASIS_HPP_ */
