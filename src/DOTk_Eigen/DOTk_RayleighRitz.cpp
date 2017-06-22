/*
 * DOTk_RayleighRitz.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <omp.h>
#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_EigenQR.hpp"
#include "DOTk_RayleighRitz.hpp"
#include "DOTk_OrthogonalFactorization.hpp"

namespace dotk
{

DOTk_RayleighRitz::DOTk_RayleighRitz(const std::shared_ptr<dotk::DOTk_OrthogonalFactorization> & qr_method_) :
        dotk::DOTk_EigenMethod(dotk::types::RAYLEIGH_RITZ_METHOD),
        m_WorkMatrix(),
        m_ReducedMatrix(),
        m_OrthonormalBasis(),
        m_ReducedEigenBasis(),
        m_Eigen(new dotk::DOTk_EigenQR(qr_method_)),
        m_QR(qr_method_)
{
}

DOTk_RayleighRitz::DOTk_RayleighRitz(const std::shared_ptr<dotk::DOTk_OrthogonalFactorization> & qr_method_,
                                     const std::shared_ptr<dotk::DOTk_EigenMethod> & eigen_solver_) :
        dotk::DOTk_EigenMethod(dotk::types::RAYLEIGH_RITZ_METHOD),
        m_WorkMatrix(),
        m_ReducedMatrix(),
        m_OrthonormalBasis(),
        m_ReducedEigenBasis(),
        m_Eigen(eigen_solver_),
        m_QR(qr_method_)
{
}

DOTk_RayleighRitz::~DOTk_RayleighRitz()
{
}

void DOTk_RayleighRitz::solve(const std::shared_ptr<dotk::matrix<Real> > & matrix_,
                              std::shared_ptr<dotk::Vector<Real> > & eigenvalues_,
                              std::shared_ptr<dotk::matrix<Real> > & eigenvectors_)
{
    this->initialize(eigenvectors_);

    // Q = m_OrthonormalBasis; R = m_WorkMatrix
    m_QR->factorization(matrix_, m_OrthonormalBasis, m_WorkMatrix);

    // Mr  = Q^t * M * Q
    matrix_->gemm(false, false, 1., *m_OrthonormalBasis, 0., *m_WorkMatrix);
    m_OrthonormalBasis->gemm(true, false, 1., *m_WorkMatrix, 0., *m_ReducedMatrix);

    // Compute reduced matrix (Mo) eigenpairs (lambda,V)
    m_Eigen->solve(m_ReducedMatrix, eigenvalues_, m_ReducedEigenBasis);

    // Approximate full eigenvectors from reduced eigenvectors map (U_i = Q * v_i)
    size_t index;
    size_t basis_dim = matrix_->basisDimension();
    int thread_count = omp_get_max_threads();

# pragma omp parallel num_threads(thread_count) \
default( none ) \
shared ( basis_dim, eigenvectors_ ) \
private ( index )

# pragma omp for
    for(index = 0; index < basis_dim; ++ index)
    {
        m_OrthonormalBasis->matVec(*m_ReducedEigenBasis->basis(index), *eigenvectors_->basis(index));
    }
}

void DOTk_RayleighRitz::initialize(const std::shared_ptr<dotk::matrix<Real> > & eigenvectors_)
{
    if(m_OrthonormalBasis.use_count() <= 0)
    {
        m_WorkMatrix = eigenvectors_->clone();
        m_OrthonormalBasis = eigenvectors_->clone();
        // TODO: PROPERLY INITIALIZE REDUCED MATRICES
        m_ReducedMatrix = eigenvectors_->clone();
        m_ReducedEigenBasis = eigenvectors_->clone();
    }
    eigenvectors_->fill(0.);
}

}
