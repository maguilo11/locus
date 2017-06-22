/*
 * DOTk_EigenQR.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EIGENQR_HPP_
#define DOTK_EIGENQR_HPP_

#include "DOTk_EigenMethod.hpp"

namespace dotk
{

class DOTk_OrthogonalFactorization;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_EigenQR : public dotk::DOTk_EigenMethod
{
public:
    DOTk_EigenQR(std::shared_ptr<dotk::DOTk_OrthogonalFactorization> qr_method_,
                 size_t max_num_qr_iterations_ = 25);
    virtual ~DOTk_EigenQR();

    void setMaxNumItr(size_t itr_);
    size_t getMaxNumItr() const;
    virtual void solve(const std::shared_ptr<dotk::matrix<Real> > & matrix_,
                       std::shared_ptr<dotk::Vector<Real> > & eigenvalues_,
                       std::shared_ptr<dotk::matrix<Real> > & eigenvectors_);

private:
    void initialize(const std::shared_ptr<dotk::matrix<Real> > & input_);

private:
    size_t m_MaxNumItr;
    std::shared_ptr<dotk::matrix<Real> > m_R;
    std::shared_ptr<dotk::matrix<Real> > m_Q;
    std::shared_ptr<dotk::matrix<Real> > m_Matrix;
    std::shared_ptr<dotk::matrix<Real> > m_CurrentEigenVectors;
    std::shared_ptr<dotk::DOTk_OrthogonalFactorization> m_QR;

private:
    DOTk_EigenQR(const dotk::DOTk_EigenQR &);
    dotk::DOTk_EigenQR & operator=(const dotk::DOTk_EigenQR & rhs_);
};

}

#endif /* DOTK_EIGENQR_HPP_ */
