/*
 * DOTk_ArnoldiProjection.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_MathUtils.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_ArnoldiProjection.hpp"
#include "DOTk_UpperTriangularMatrix.hpp"
#include "DOTk_UpperTriangularMatrix.cpp"
#include "DOTk_UpperTriangularDirectSolver.hpp"

namespace dotk
{

DOTk_ArnoldiProjection::DOTk_ArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                               size_t krylov_subspace_dim_) :
        dotk::DOTk_OrthogonalProjection::DOTk_OrthogonalProjection(dotk::types::ARNOLDI, krylov_subspace_dim_),
        m_Sine(krylov_subspace_dim_, 0.),
        m_Cosine(krylov_subspace_dim_, 0.),
        m_ScaleFactorsStorage(krylov_subspace_dim_, 0.),
        m_NormAppxProjectedResidualStorage(krylov_subspace_dim_, 0.),
        m_DirectSolver(new dotk::DOTk_UpperTriangularDirectSolver(krylov_subspace_dim_)),
        m_HessenbergMatrix(new dotk::serial::DOTk_UpperTriangularMatrix<Real>(krylov_subspace_dim_)),
        m_OrthogonalBasis(krylov_subspace_dim_ + 1)
{
    this->initialize(primal_);
}

DOTk_ArnoldiProjection::~DOTk_ArnoldiProjection()
{
}

void DOTk_ArnoldiProjection::setInitialResidual(Real value_)
{
    m_NormAppxProjectedResidualStorage[0] = value_;
}

Real DOTk_ArnoldiProjection::getInitialResidual() const
{
    return (m_NormAppxProjectedResidualStorage[0]);
}

void DOTk_ArnoldiProjection::setOrthogonalVector(size_t index_,
                                                 const std::shared_ptr<dotk::Vector<Real> > & vector_)
{
    m_OrthogonalBasis[index_]->update(1., *vector_, 0.);
}

const std::shared_ptr<dotk::Vector<Real> > &
DOTk_ArnoldiProjection::getOrthogonalVector(size_t index_) const
{
    return (m_OrthogonalBasis[index_]);
}

void DOTk_ArnoldiProjection::clear()
{
    m_HessenbergMatrix->fill(0.);
    m_Sine.assign(m_Sine.size(), 0.);
    m_Cosine.assign(m_Sine.size(), 0.);
    m_ScaleFactorsStorage.assign(m_Sine.size(), 0.);
    m_NormAppxProjectedResidualStorage.assign(m_Sine.size(), 0.);
}

void DOTk_ArnoldiProjection::applyGivensRotationsToHessenbergMatrix(int current_itr_)
{
    for(int index = 0; index <= current_itr_ - 1; ++ index)
    {
        Real alpha = m_Cosine[index] * (*m_HessenbergMatrix)(index, current_itr_)
                + m_Sine[index] * (*m_HessenbergMatrix)(index + 1, current_itr_);
        Real beta = -m_Sine[index] * (*m_HessenbergMatrix)(index, current_itr_)
                + m_Cosine[index] * (*m_HessenbergMatrix)(index + 1, current_itr_);
        (*m_HessenbergMatrix)(index + 1, current_itr_) = beta;
        (*m_HessenbergMatrix)(index, current_itr_) = alpha;
    }
}
void DOTk_ArnoldiProjection::updateHessenbergMatrix(size_t current_itr_,
                                                    const std::shared_ptr<dotk::Vector<Real> > & left_prec_times_vec_)
{
    for(size_t index = 0; index <= current_itr_; ++ index)
    {
        Real alpha = left_prec_times_vec_->dot(*m_OrthogonalBasis[index]);
        left_prec_times_vec_->update(-alpha, *m_OrthogonalBasis[index], 1.);
        (*m_HessenbergMatrix)(index, current_itr_) = alpha;
    }
}

void DOTk_ArnoldiProjection::apply(const dotk::DOTk_KrylovSolver * const solver_,
                                   const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    size_t current_solver_itr = solver_->getNumSolverItrDone();
    size_t current_orthogonal_vector_index = current_solver_itr - 1;
    this->arnoldi(current_orthogonal_vector_index, kernel_vector_);
}

void DOTk_ArnoldiProjection::arnoldi(size_t ortho_vector_index_,
                                     const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    this->updateHessenbergMatrix(ortho_vector_index_, kernel_vector_);
    Real scaling = kernel_vector_->norm();

    Real beta = static_cast<Real>(1.) / scaling;
    m_OrthogonalBasis[ortho_vector_index_ + 1]->update(beta, *kernel_vector_, 0.);

    this->applyGivensRotationsToHessenbergMatrix(ortho_vector_index_);
    Real hessenberg_matrix_entry = (*m_HessenbergMatrix)(ortho_vector_index_, ortho_vector_index_);
    dotk::givens(hessenberg_matrix_entry, scaling, m_Cosine[ortho_vector_index_], m_Sine[ortho_vector_index_]);

    Real sine_times_norm_projected_residual = static_cast<Real>(-1.) * m_Sine[ortho_vector_index_]
            * m_NormAppxProjectedResidualStorage[ortho_vector_index_];
    m_NormAppxProjectedResidualStorage[ortho_vector_index_] = m_Cosine[ortho_vector_index_]
            * m_NormAppxProjectedResidualStorage[ortho_vector_index_];
    Real value = m_Cosine[ortho_vector_index_] * hessenberg_matrix_entry + m_Sine[ortho_vector_index_] * scaling;
    (*m_HessenbergMatrix)(ortho_vector_index_, ortho_vector_index_) = value;

    m_DirectSolver->setNumUnknowns(ortho_vector_index_ + 1);
    m_DirectSolver->solve(m_HessenbergMatrix, m_NormAppxProjectedResidualStorage, m_ScaleFactorsStorage);
    m_NormAppxProjectedResidualStorage[ortho_vector_index_ + 1] = sine_times_norm_projected_residual;

    kernel_vector_->fill(0.);
    for(size_t index = 0; index <= ortho_vector_index_; ++ index)
    {
        kernel_vector_->update(m_ScaleFactorsStorage[index], *m_OrthogonalBasis[index], 1.);
    }
}

void DOTk_ArnoldiProjection::initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    bool is_dual_allocated = primal_->dual().use_count() > 0;
    bool is_state_allocated = primal_->state().use_count() > 0;
    bool is_control_allocated = primal_->control().use_count() > 0;
    size_t dimensions = dotk::DOTk_OrthogonalProjection::getKrylovSubspaceDim() + 1;

    if((is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == true))
    {
        for(size_t row = 0; row < dimensions; ++ row)
        {
            m_OrthogonalBasis[row] = primal_->control()->clone();
        }
    }
    else
    {
        for(size_t row = 0; row < dimensions; ++ row)
        {
            m_OrthogonalBasis[row].reset(new dotk::DOTk_MultiVector<Real>(*primal_));
            m_OrthogonalBasis[row]->fill(0);
        }
    }

    if((is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == false))
    {
        std::perror("\n**** DOTk ERROR in DOTk_ArnoldiProjection::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
