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

class DOTk_MexArrayPtr;

template<typename ScalarType>
class DOTk_MexInequalityConstraint : public DOTk_InequalityConstraint<ScalarType>
{
public:
    DOTk_MexInequalityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexInequalityConstraint();

    // Linear Programming API
    ScalarType bound(const size_t & index_);
    ScalarType value(const dotk::Vector<ScalarType> & primal_, const size_t & index_);
    void gradient(const dotk::Vector<ScalarType> & primal_, const size_t & index_, dotk::Vector<ScalarType> & derivative_);
    void hessian(const dotk::Vector<ScalarType> & primal_,
                 const dotk::Vector<ScalarType> & delta_primal_,
                 const size_t & index_,
                 dotk::Vector<ScalarType> & derivative_);

    // Nonlinear Programming API
    ScalarType value(const dotk::Vector<ScalarType> & state_, const dotk::Vector<ScalarType> & control_, const size_t & index_);
    void partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                const dotk::Vector<ScalarType> & control_,
                                const size_t & index_,
                                dotk::Vector<ScalarType> & derivative_);
    void partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                  const dotk::Vector<ScalarType> & control_,
                                  const size_t & index_,
                                  dotk::Vector<ScalarType> & derivative_);

private:
    void clear();
    void initialize(const mxArray* operators_, const dotk::types::problem_t & type_);

private:
    dotk::DOTk_MexArrayPtr m_Value;
    dotk::DOTk_MexArrayPtr m_Evaluate;
    dotk::DOTk_MexArrayPtr m_FirstDerivative;
    dotk::DOTk_MexArrayPtr m_SecondDerivative;
    dotk::DOTk_MexArrayPtr m_FirstDerivativeWrtState;
    dotk::DOTk_MexArrayPtr m_FirstDerivativeWrtControl;

private:
    DOTk_MexInequalityConstraint(const dotk::DOTk_MexInequalityConstraint<ScalarType> & rhs_);
    dotk::DOTk_MexInequalityConstraint<ScalarType> & operator=(const dotk::DOTk_MexInequalityConstraint<ScalarType> & rhs_);
};

}

#endif /* DOTK_MEXINEQUALITYCONSTRAINT_HPP_ */
