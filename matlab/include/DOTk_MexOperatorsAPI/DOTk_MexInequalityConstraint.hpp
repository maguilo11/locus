/*
 * DOTk_MexInequalityConstraint.hpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXINEQUALITYCONSTRAINT_HPP_
#define DOTK_MEXINEQUALITYCONSTRAINT_HPP_

#include <mex.h>
#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_MexInequalityConstraint : public DOTk_InequalityConstraint<double>
{
public:
    DOTk_MexInequalityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexInequalityConstraint();

    // Linear Programming API
    double bound();
    double value(const dotk::Vector<double> & control_);
    void gradient(const dotk::Vector<double> & control_, dotk::Vector<double> & derivative_);
    void hessian(const dotk::Vector<double> & control_,
                 const dotk::Vector<double> & vector_,
                 dotk::Vector<double> & output_);

    // Nonlinear Programming API
    double value(const dotk::Vector<double> & state_, const dotk::Vector<double> & control_);
    void partialDerivativeState(const dotk::Vector<double> & state_,
                                const dotk::Vector<double> & control_,
                                dotk::Vector<double> & output_);
    void partialDerivativeControl(const dotk::Vector<double> & state_,
                                  const dotk::Vector<double> & control_,
                                  dotk::Vector<double> & output_);

private:
    void clear();
    void initialize(const mxArray* operators_, const dotk::types::problem_t & type_);

private:
    mxArray* m_Value;
    mxArray* m_Bound;
    mxArray* m_Gradient;
    mxArray* m_Hessian;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;

private:
    DOTk_MexInequalityConstraint(const dotk::DOTk_MexInequalityConstraint & rhs_);
    dotk::DOTk_MexInequalityConstraint & operator=(const dotk::DOTk_MexInequalityConstraint & rhs_);
};

}

#endif /* DOTK_MEXINEQUALITYCONSTRAINT_HPP_ */
