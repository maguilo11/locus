/*
 * DOTk_GramSchmidt.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_GramSchmidt.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_LinearOperator.hpp"

namespace dotk
{

DOTk_GramSchmidt::DOTk_GramSchmidt(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t krylov_subspace_dim_) :
        dotk::DOTk_OrthogonalProjection(dotk::types::GRAM_SCHMIDT, krylov_subspace_dim_),
        m_OrthogonalBasis(krylov_subspace_dim_),
        m_LinearOperatorTimesOrthoVector(krylov_subspace_dim_)
{
    this->initialize(primal_);
}

DOTk_GramSchmidt::~DOTk_GramSchmidt()
{
}

void DOTk_GramSchmidt::gramSchmidt(size_t ortho_vector_index_,
                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    m_OrthogonalBasis[ortho_vector_index_]->axpy(static_cast<Real>(-1.), *kernel_vector_);
    for(size_t index = 0; index < ortho_vector_index_; ++index)
    {
        Real kernel_vector_dot_linear_operator_times_ortho_vector =
                kernel_vector_->dot(*m_LinearOperatorTimesOrthoVector[index]);
        Real projected_prec_residual_dot_linear_operator_times_projected_prec_residual =
                m_OrthogonalBasis[index]->dot(*m_LinearOperatorTimesOrthoVector[index]);

        Real rayleigh_quotient = kernel_vector_dot_linear_operator_times_ortho_vector
                / projected_prec_residual_dot_linear_operator_times_projected_prec_residual;

        m_OrthogonalBasis[ortho_vector_index_]->axpy(-rayleigh_quotient, *m_OrthogonalBasis[index]);
    }
}

void DOTk_GramSchmidt::setOrthogonalVector(size_t index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_)
{
    m_OrthogonalBasis[index_]->copy(*vec_);
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_GramSchmidt::getOrthogonalVector(size_t index_) const
{
    return (m_OrthogonalBasis[index_]);
}

void DOTk_GramSchmidt::clear()
{
    size_t krylov_subspace_dim = m_OrthogonalBasis.size();
    for(size_t dim = 0; dim < krylov_subspace_dim; ++dim)
    {
        m_OrthogonalBasis[dim]->fill(0.);
        m_LinearOperatorTimesOrthoVector[dim]->fill(0.);
    }
}

void DOTk_GramSchmidt::apply(const dotk::DOTk_KrylovSolver * const solver_,
                             const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    size_t current_solver_itr = solver_->getNumSolverItrDone();
    this->gramSchmidt(current_solver_itr, kernel_vector_);
}

void DOTk_GramSchmidt::setLinearOperatorTimesOrthoVector(size_t index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_)
{
    m_LinearOperatorTimesOrthoVector[index_]->copy(*vec_);
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_GramSchmidt::getLinearOperatorTimesOrthoVector(size_t index_) const
{
    return (m_LinearOperatorTimesOrthoVector[index_]);
}

void DOTk_GramSchmidt::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    bool is_dual_allocated = primal_->dual().use_count() > 0;
    bool is_state_allocated = primal_->state().use_count() > 0;
    bool is_control_allocated = primal_->control().use_count() > 0;

    if((is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == true))
    {
        for(size_t row = 0; row < dotk::DOTk_OrthogonalProjection::getKrylovSubspaceDim(); ++row)
        {
            m_OrthogonalBasis[row] = primal_->control()->clone();
            m_LinearOperatorTimesOrthoVector[row] = primal_->control()->clone();
        }
    }
    else
    {
        for(size_t row = 0; row < dotk::DOTk_OrthogonalProjection::getKrylovSubspaceDim(); ++row)
        {
            m_OrthogonalBasis[row].reset(new dotk::DOTk_MultiVector<Real>(*primal_));
            m_OrthogonalBasis[row]->fill(0);
            m_LinearOperatorTimesOrthoVector[row].reset(new dotk::DOTk_MultiVector<Real>(*primal_));
            m_LinearOperatorTimesOrthoVector[row]->fill(0);
        }
    }

    if((is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == false))
    {
        std::perror("\n**** DOTk ERROR in DOTk_GramSchmidt::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}

