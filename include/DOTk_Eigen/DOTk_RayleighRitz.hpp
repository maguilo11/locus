/*
 * DOTk_RayleighRitz.hpp
 *
 *  Created on: Jul 17, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_RAYLEIGHRITZ_HPP_
#define DOTK_RAYLEIGHRITZ_HPP_

#include <tr1/memory>
#include "DOTk_EigenMethod.hpp"

namespace dotk
{

class DOTk_EigenMethod;
class DOTk_OrthogonalFactorization;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_RayleighRitz : public dotk::DOTk_EigenMethod
{
public:
    explicit DOTk_RayleighRitz(const std::tr1::shared_ptr<dotk::DOTk_OrthogonalFactorization> & qr_method_);
    DOTk_RayleighRitz(const std::tr1::shared_ptr<dotk::DOTk_OrthogonalFactorization> & qr_method_,
                      const std::tr1::shared_ptr<dotk::DOTk_EigenMethod> & eigen_solver_);
    virtual ~DOTk_RayleighRitz();

    virtual void solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & matrix_,
                       std::tr1::shared_ptr<dotk::Vector<Real> > & eigenvalues_,
                       std::tr1::shared_ptr<dotk::matrix<Real> > & eigenvectors_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::matrix<Real> > & eigenvectors_);

private:
    std::tr1::shared_ptr<dotk::matrix<Real> > m_WorkMatrix;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_ReducedMatrix;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_OrthonormalBasis;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_ReducedEigenBasis;

    std::tr1::shared_ptr<dotk::DOTk_EigenMethod> m_Eigen;
    std::tr1::shared_ptr<dotk::DOTk_OrthogonalFactorization> m_QR;

private:
    DOTk_RayleighRitz(const dotk::DOTk_RayleighRitz &);
    dotk::DOTk_RayleighRitz & operator=(const dotk::DOTk_RayleighRitz & rhs_);
};

}

#endif /* DOTK_RAYLEIGHRITZ_HPP_ */
