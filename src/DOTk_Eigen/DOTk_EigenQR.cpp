/*
 * DOTk_EigenQR.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_EigenQR.hpp"
#include "DOTk_OrthogonalFactorization.hpp"

namespace dotk
{

DOTk_EigenQR::DOTk_EigenQR(std::tr1::shared_ptr<dotk::DOTk_OrthogonalFactorization> qr_method_,
                           size_t max_num_qr_iterations_) :
        dotk::DOTk_EigenMethod(dotk::types::QR_EIGEN_METHOD),
        m_MaxNumItr(max_num_qr_iterations_),
        m_R(),
        m_Q(),
        m_Matrix(),
        m_CurrentEigenVectors(),
        m_QR(qr_method_)
{
}

DOTk_EigenQR::~DOTk_EigenQR()
{
}

void DOTk_EigenQR::setMaxNumItr(size_t itr_)
{
    m_MaxNumItr = itr_;
}

size_t DOTk_EigenQR::getMaxNumItr() const
{
    return (m_MaxNumItr);
}

void DOTk_EigenQR::solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & matrix_,
                         std::tr1::shared_ptr<dotk::vector<Real> > & eigenvalues_,
                         std::tr1::shared_ptr<dotk::matrix<Real> > & eigenvectors_)
{
    this->initialize(matrix_);

    eigenvalues_->fill(0.);
    eigenvectors_->identity();
    m_Matrix->copy(*matrix_);
    size_t max_num_itr = this->getMaxNumItr();

    for(size_t itr = 0; itr < max_num_itr; ++itr)
    {
        m_QR->factorization(m_Matrix, m_Q, m_R);
        m_R->gemm(false, false, 1., *m_Q, 0., *m_Matrix);
        eigenvectors_->gemm(false, false, 1., *m_Q, 0., *m_CurrentEigenVectors);
        eigenvectors_->copy(*m_CurrentEigenVectors);
    }
    m_Matrix->diag(*eigenvalues_);
}

void DOTk_EigenQR::initialize(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_)
{
    if(m_Q.use_count() <= 0)
    {
        m_R = input_->clone();
        m_Q = input_->clone();
        m_Matrix = input_->clone();
        m_CurrentEigenVectors = input_->clone();
    }
}

}
