/*
 * DOTk_PowerMethod.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_POWERMETHOD_HPP_
#define DOTK_POWERMETHOD_HPP_

#include <tr1/memory>
#include "DOTk_EigenMethod.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_PowerMethod : public dotk::DOTk_EigenMethod
{
public:
    DOTk_PowerMethod(size_t max_num_itr_ = 10, Real relative_difference_tolerance_ = 1e-6);
    virtual ~DOTk_PowerMethod();

    size_t getMaxNumItr() const;
    void setMaxNumItr(size_t itr_);
    Real getRelativeDifferenceTolerance() const;
    void setRelativeDifferenceTolerance(Real tol_);
    virtual void solve(const std::tr1::shared_ptr<dotk::matrix<Real> > & matrix_,
                       Real & eigenvalues_,
                       std::tr1::shared_ptr<dotk::Vector<Real> > & eigenvectors_);

private:
    size_t m_MaxNumItr;
    Real m_RelativeDifferenceTolerance;

private:
    DOTk_PowerMethod(const dotk::DOTk_PowerMethod &);
    dotk::DOTk_PowerMethod & operator=(const dotk::DOTk_PowerMethod & rhs_);
};

}

#endif /* DOTK_POWERMETHOD_HPP_ */
