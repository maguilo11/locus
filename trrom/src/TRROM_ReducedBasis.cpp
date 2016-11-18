/*
 * TRROM_ReducedBasis.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <cassert>
#include <tr1/memory>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_ReducedBasis.hpp"

namespace trrom
{

void properOrthogonalDecomposition(const double & threshold_,
                                   const trrom::Vector<double> & singular_values_,
                                   const trrom::Matrix<double> & left_singular_vectors_,
                                   const trrom::Matrix<double> & snapshots_,
                                   trrom::Matrix<double> & basis_)
{
    /// Proper Orthogonal Decomposition (POD) Method routine
    ///   Parameters:
    ///     threshold_ (\epsilon)     - (In) tolerance on cumulative energy
    ///     singular_values_ (\Sigma) - (In) current set of singular values
    ///     singular_vectors_ (\Psi)  - (In) current set of singular vectors
    ///     snapshots_ (\mathbf{U})   - (In) current data storage
    ///     basis_ (\Phi)             - (Out) orthonormal basis
    int num_singular_values = singular_values_.size();
    assert(basis_.getNumCols() > 0);
    assert(num_singular_values == basis_.getNumCols());
    assert(num_singular_values == left_singular_vectors_.getNumRows());
    assert(num_singular_values == left_singular_vectors_.getNumCols());

    double cumulative_energy = 0;
    double total_energy = singular_values_.sum();
    for(int index = 0; index < num_singular_values; ++index)
    {
        // Compute i-th basis vector: \Phi(:,i) = \alpha\mathbf{U}\Psi(:,i), where \alpha = \frac{1}{\sqrt(\Sigma_i)}
        basis_.insert(snapshots_.vector(index));
        snapshots_.gemv(false, 1., left_singular_vectors_.vector(index), 0., basis_.vector(index));
        double alpha = static_cast<double>(1.) / std::pow(singular_values_[index], 0.5);
        basis_.vector(index).scale(alpha);
        // Check if energy threshold is violated, i.e. \sum_{i=1}^{index}\Sigma_i > \epsilon
        cumulative_energy += singular_values_[index];
        double energy = cumulative_energy / total_energy;
        if(energy > threshold_)
        {
            break;
        }
    }
}

}
