/*
 * DOTk_RayleighQuotient.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_EigenUtils.hpp"
#include "DOTk_RayleighQuotient.hpp"

namespace dotk
{

DOTk_RayleighQuotient::DOTk_RayleighQuotient(size_t max_num_itr_, Real relative_difference_tolerance_) :
        dotk::DOTk_EigenMethod(dotk::types::RAYLEIGH_QUOTIENT_METHOD),
        m_MaxNumItr(max_num_itr_),
        m_RelativeDifferenceTolerance(relative_difference_tolerance_)
{
}

DOTk_RayleighQuotient::~DOTk_RayleighQuotient()
{
}

size_t DOTk_RayleighQuotient::getMaxNumItr() const
{
    return (m_MaxNumItr);
}

void DOTk_RayleighQuotient::setMaxNumItr(size_t itr_)
{
    m_MaxNumItr = itr_;
}

Real DOTk_RayleighQuotient::getRelativeDifferenceTolerance() const
{
    return (m_RelativeDifferenceTolerance);
}

void DOTk_RayleighQuotient::setRelativeDifferenceTolerance(Real tol_)
{
    m_RelativeDifferenceTolerance = tol_;
}

void DOTk_RayleighQuotient::solve(const std::shared_ptr<dotk::matrix<Real> > & matrix_,
                                  Real & eigenvalues_,
                                  std::shared_ptr<dotk::Vector<Real> > & eigenvectors_)
{
    size_t max_num_itr = this->getMaxNumItr();
    Real relative_difference_tolerance = this->getRelativeDifferenceTolerance();
    eigenvalues_ = dotk::eigen::rayleighQuotientMethod(*matrix_,
                                                       *eigenvectors_,
                                                       max_num_itr,
                                                       relative_difference_tolerance);
}

}
