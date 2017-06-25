/*
 * TRROM_DiscreteEmpiricalInterpolation.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: maguilo
 */

#include <cassert>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_SolverInterface.hpp"
#include "TRROM_LinearAlgebraFactory.hpp"
#include "TRROM_DiscreteEmpiricalInterpolation.hpp"

namespace trrom
{

DiscreteEmpiricalInterpolation::DiscreteEmpiricalInterpolation(const std::shared_ptr<trrom::SolverInterface> & solver_,
                                                               const std::shared_ptr<trrom::LinearAlgebraFactory> & factory_) :
        m_Solver(solver_),
        m_Factory(factory_)
{
}

DiscreteEmpiricalInterpolation::~DiscreteEmpiricalInterpolation()
{
}

void DiscreteEmpiricalInterpolation::apply(const std::shared_ptr<trrom::Matrix<double> > & data_,
                                           const std::shared_ptr<trrom::Matrix<double> > & binary_matrix_,
                                           std::shared_ptr<trrom::Vector<double> > & active_indices_)
{
    assert(data_->getNumRows() == binary_matrix_->getNumRows());
    assert(data_->getNumCols() == binary_matrix_->getNumCols());

    std::shared_ptr<trrom::Vector<double> > residual = data_->vector(0)->create();
    residual->update(1., *data_->vector(0), 0.);
    residual->modulus();

    int max_index = 0;
    residual->max(max_index);
    int num_basis_vectors = data_->getNumCols();
    m_Factory->buildLocalVector(num_basis_vectors, active_indices_);

    binary_matrix_->fill(0);
    (*active_indices_)[0] = max_index;
    (*binary_matrix_)(max_index, 0) = 1.;

    std::shared_ptr<trrom::Vector<double> > reduced_snapshot;
    m_Factory->buildLocalVector(num_basis_vectors, reduced_snapshot);

    std::shared_ptr<trrom::Matrix<double> > A;
    std::shared_ptr<trrom::Matrix<double> > P;
    std::shared_ptr<trrom::Vector<double> > rhs;
    for(int basis_vector_index = 1; basis_vector_index < num_basis_vectors; ++basis_vector_index)
    {
        m_Factory->buildMultiVector(basis_vector_index, residual, P);
        for(int index = 0; index < basis_vector_index; ++index)
        {
            P->insert(*binary_matrix_->vector(index), index);
        }
        // Compute \mathbf{A} = \mathbf{P}(:,1:index-1)^{T}\mathbf{U}
        m_Factory->buildLocalMatrix(basis_vector_index, num_basis_vectors, A);
        P->gemm(true, false, 1., *data_, 0., *A);

        // Compute \vec{rhs} = \mathbf{P}(:,1:basis_vector-1)^{T}\mathbf{U}(:,basis_vector)
        m_Factory->buildLocalVector(basis_vector_index, rhs);
        P->gemv(true, 1., *data_->vector(basis_vector_index), 0., *rhs);

        // Solve \mathbf{A}\vec{lhs}=\vec{rhs}
        reduced_snapshot->fill(0);
        m_Solver->solve(*A, *rhs, *reduced_snapshot);

        // Compute residual, \vec{r} = \mathbf{U}(:,basis_vector) - \mathbf{U}\vec{lhs}
        data_->gemv(false, -1., *reduced_snapshot, 0., *residual);
        residual->update(1., *data_->vector(basis_vector_index), 1.);

        // Compute DEIM max_index = \max(\lvert \vec{res} \rvert) and set P(max_index,basis_vector) = 1
        residual->modulus();
        residual->max(max_index);
        (*active_indices_)[basis_vector_index] = max_index;
        (*binary_matrix_)(max_index, basis_vector_index) = 1.;
    }
}

}
