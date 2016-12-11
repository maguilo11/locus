/*
 * DOTk_EigenMethod.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>
#include <ostream>
#include <iostream>

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_EigenMethod.hpp"

namespace dotk
{

DOTk_EigenMethod::DOTk_EigenMethod(dotk::types::eigen_t type_) :
        m_EigenType(type_)
{
}

DOTk_EigenMethod::~DOTk_EigenMethod()
{
}

dotk::types::eigen_t DOTk_EigenMethod::type() const
{
    return (m_EigenType);
}

void DOTk_EigenMethod::solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                             std::tr1::shared_ptr<dotk::Vector<Real> > & eigenvalues_,
                             std::tr1::shared_ptr<dotk::matrix<Real> > & eigenvectors_)
{
    std::string msg(" CALLING UNIMPLEMENTED dotk::DOTk_EigenMethod::solve **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
}

void DOTk_EigenMethod::solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                             Real & eigenvalues_,
                             std::tr1::shared_ptr<dotk::Vector<Real> > & eigenvectors_)
{
    std::string msg(" CALLING UNIMPLEMENTED dotk::DOTk_EigenMethod::solve **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
}

}
