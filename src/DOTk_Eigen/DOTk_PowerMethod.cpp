/*
 * DOTk_PowerMethod.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_EigenUtils.hpp"
#include "DOTk_PowerMethod.hpp"

namespace dotk
{

DOTk_PowerMethod::DOTk_PowerMethod(size_t max_num_itr_, Real relative_difference_tolerance_) :
        dotk::DOTk_EigenMethod(dotk::types::POWER_METHOD),
        m_MaxNumItr(max_num_itr_),
        m_RelativeDifferenceTolerance(relative_difference_tolerance_)
{
}

DOTk_PowerMethod::~DOTk_PowerMethod()
{
}

size_t DOTk_PowerMethod::getMaxNumItr() const
{
    return (m_MaxNumItr);
}

void DOTk_PowerMethod::setMaxNumItr(size_t itr_)
{
    m_MaxNumItr = itr_;
}

Real DOTk_PowerMethod::getRelativeDifferenceTolerance() const
{
    return (m_RelativeDifferenceTolerance);
}

void DOTk_PowerMethod::setRelativeDifferenceTolerance(Real tol_)
{
    m_RelativeDifferenceTolerance = tol_;
}

void DOTk_PowerMethod::solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & matrix_,
                             Real & eigenvalues_,
                             std::tr1::shared_ptr<dotk::Vector<Real> > & eigenvectors_)
{
    size_t max_num_itr = this->getMaxNumItr();
    Real relative_difference_tolerance = this->getRelativeDifferenceTolerance();
    eigenvalues_ = dotk::eigen::powerMethod(*matrix_, *eigenvectors_, max_num_itr, relative_difference_tolerance);
}

}
