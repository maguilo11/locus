/*
 * DOTk_EigenMethod.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EIGENMETHOD_HPP_
#define DOTK_EIGENMETHOD_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename Type>
class vector;
template<typename Type>
class matrix;

class DOTk_EigenMethod
{
public:
    DOTk_EigenMethod(dotk::types::eigen_t type_);
    virtual ~DOTk_EigenMethod();

    dotk::types::eigen_t type() const;
    virtual void solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                       std::tr1::shared_ptr<dotk::vector<Real> > & eigenvalues_,
                       std::tr1::shared_ptr<dotk::matrix<Real> > & eigenvectors_);
    virtual void solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                       Real & eigenvalues_,
                       std::tr1::shared_ptr<dotk::vector<Real> > & eigenvectors_);

private:
    dotk::types::eigen_t m_EigenType;

private:
    DOTk_EigenMethod(const dotk::DOTk_EigenMethod &);
    dotk::DOTk_EigenMethod & operator=(const dotk::DOTk_EigenMethod & rhs_);
};

}

#endif /* DOTK_EIGENMETHOD_HPP_ */
