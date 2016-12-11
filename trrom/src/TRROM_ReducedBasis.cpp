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

int energy(const double & threshold_, const trrom::Vector<double> & singular_values_)
{
    assert(threshold_ > 0.);
    assert(threshold_ <= 1.);
    assert(singular_values_.size() > 0);

    int index;
    double cumulative_energy = 0;
    double total_energy = singular_values_.sum();
    int num_singular_values = singular_values_.size();
    for(index = 0; index < num_singular_values; ++ index)
    {
        // Check if energy threshold is violated, i.e. \sum_{i=1}^{index}\Sigma_i > \epsilon
        cumulative_energy += singular_values_[index];
        double energy = cumulative_energy / total_energy;
        if(energy > threshold_)
        {
            break;
        }
    }
    return (index);
}

void properOrthogonalDecomposition(const trrom::Vector<double> & singular_values_,
                                   const trrom::Matrix<double> & left_singular_vectors_,
                                   const trrom::Matrix<double> & snapshots_,
                                   trrom::Matrix<double> & basis_)
{
    assert(basis_.getNumCols() > 0);
    assert(singular_values_.size() == left_singular_vectors_.getNumRows());
    assert(singular_values_.size() == left_singular_vectors_.getNumCols());

    int num_basis_vectors = basis_.getNumCols();
    for(int index = 0; index < num_basis_vectors; ++ index)
    {
        // Compute i-th basis vector: \Phi(:,i) = \alpha\mathbf{U}\Psi(:,i), where \alpha = \frac{1}{\sqrt(\Sigma_i)}
        basis_.insert(*snapshots_.vector(index), index);
        snapshots_.gemv(false, 1., *left_singular_vectors_.vector(index), 0., *basis_.vector(index));
        double alpha = static_cast<double>(1.) / std::pow(singular_values_[index], 0.5);
        basis_.vector(index)->scale(alpha);
    }
}

}
