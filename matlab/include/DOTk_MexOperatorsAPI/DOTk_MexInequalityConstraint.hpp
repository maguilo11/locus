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

template<typename Type>
class vector;

class DOTk_MexArrayPtr;

template<typename Type>
class DOTk_MexInequalityConstraint : public DOTk_InequalityConstraint<Type>
{
public:
    DOTk_MexInequalityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexInequalityConstraint();

    // Linear Programming API
    Type bound(const size_t & index_);
    Type value(const dotk::vector<Type> & primal_, const size_t & index_);
    void gradient(const dotk::vector<Type> & primal_, const size_t & index_, dotk::vector<Type> & derivative_);
    void hessian(const dotk::vector<Type> & primal_,
                 const dotk::vector<Type> & delta_primal_,
                 const size_t & index_,
                 dotk::vector<Type> & derivative_);

    // Nonlinear Programming API
    Type value(const dotk::vector<Type> & state_, const dotk::vector<Type> & control_, const size_t & index_);
    void partialDerivativeState(const dotk::vector<Type> & state_,
                                const dotk::vector<Type> & control_,
                                const size_t & index_,
                                dotk::vector<Type> & derivative_);
    void partialDerivativeControl(const dotk::vector<Type> & state_,
                                  const dotk::vector<Type> & control_,
                                  const size_t & index_,
                                  dotk::vector<Type> & derivative_);

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
    DOTk_MexInequalityConstraint(const dotk::DOTk_MexInequalityConstraint<Type> & rhs_);
    dotk::DOTk_MexInequalityConstraint<Type> & operator=(const dotk::DOTk_MexInequalityConstraint<Type> & rhs_);
};

}

#endif /* DOTK_MEXINEQUALITYCONSTRAINT_HPP_ */
