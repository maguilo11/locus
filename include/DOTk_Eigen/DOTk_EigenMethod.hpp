/*
 * DOTk_EigenMethod.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EIGENMETHOD_HPP_
#define DOTK_EIGENMETHOD_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_EigenMethod
{
public:
    DOTk_EigenMethod(dotk::types::eigen_t type_);
    virtual ~DOTk_EigenMethod();

    dotk::types::eigen_t type() const;
    virtual void solve(const std::shared_ptr<dotk::matrix<Real> > & input_,
                       std::shared_ptr<dotk::Vector<Real> > & eigenvalues_,
                       std::shared_ptr<dotk::matrix<Real> > & eigenvectors_);
    virtual void solve(const std::shared_ptr<dotk::matrix<Real> > & input_,
                       Real & eigenvalues_,
                       std::shared_ptr<dotk::Vector<Real> > & eigenvectors_);

private:
    dotk::types::eigen_t m_EigenType;

private:
    DOTk_EigenMethod(const dotk::DOTk_EigenMethod &);
    dotk::DOTk_EigenMethod & operator=(const dotk::DOTk_EigenMethod & rhs_);
};

}

#endif /* DOTK_EIGENMETHOD_HPP_ */
