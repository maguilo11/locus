/*
 * DOTk_MexEqualityConstraint.hpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXEQUALITYCONSTRAINT_HPP_
#define DOTK_MEXEQUALITYCONSTRAINT_HPP_

#include <mex.h>

#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;

template<typename Type>
class DOTk_MexEqualityConstraint : public DOTk_EqualityConstraint<Type>
{
public:
    DOTk_MexEqualityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexEqualityConstraint();

    virtual void residual(const dotk::vector<Type> & primal_, dotk::vector<Type> & output_);
    virtual void jacobian(const dotk::vector<Type> & primal_,
                          const dotk::vector<Type> & vector_,
                          dotk::vector<Type> & jacobian_times_vector_);
    virtual void adjointJacobian(const dotk::vector<Type> & primal_,
                                 const dotk::vector<Type> & dual_,
                                 dotk::vector<Type> & output_);
    virtual void hessian(const dotk::vector<Type> & primal_,
                         const dotk::vector<Type> & dual_,
                         const dotk::vector<Type> & vector_,
                         dotk::vector<Type> & output_);


    virtual void solve(const dotk::vector<Type> & control_, dotk::vector<Type> & output_);
    virtual void applyInverseJacobianState(const dotk::vector<Type> & state_,
                                              const dotk::vector<Type> & control_,
                                              const dotk::vector<Type> & rhs_,
                                              dotk::vector<Type> & output_);
    virtual void applyAdjointInverseJacobianState(const dotk::vector<Type> & state_,
                                                  const dotk::vector<Type> & control_,
                                                  const dotk::vector<Type> & rhs_,
                                                  dotk::vector<Type> & output_);
    virtual void residual(const dotk::vector<Type> & state_,
                          const dotk::vector<Type> & control_,
                          dotk::vector<Type> & output_);
    virtual void partialDerivativeState(const dotk::vector<Type> & state_,
                                        const dotk::vector<Type> & control_,
                                        const dotk::vector<Type> & vector_,
                                        dotk::vector<Type> & output_);
    virtual void partialDerivativeControl(const dotk::vector<Type> & state_,
                                          const dotk::vector<Type> & control_,
                                          const dotk::vector<Type> & vector_,
                                          dotk::vector<Type> & output_);
    virtual void adjointPartialDerivativeState(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & dual_,
                                               dotk::vector<Type> & output_);
    virtual void adjointPartialDerivativeControl(const dotk::vector<Type> & state_,
                                                 const dotk::vector<Type> & control_,
                                                 const dotk::vector<Type> & dual_,
                                                 dotk::vector<Type> & output_);
    virtual void partialDerivativeStateState(const dotk::vector<Type> & state_,
                                             const dotk::vector<Type> & control_,
                                             const dotk::vector<Type> & dual_,
                                             const dotk::vector<Type> & vector_,
                                             dotk::vector<Type> & output_);
    virtual void partialDerivativeStateControl(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & dual_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_);
    virtual void partialDerivativeControlControl(const dotk::vector<Type> & state_,
                                                 const dotk::vector<Type> & control_,
                                                 const dotk::vector<Type> & dual_,
                                                 const dotk::vector<Type> & vector_,
                                                 dotk::vector<Type> & output_);
    virtual void partialDerivativeControlState(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & dual_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_);

private:
    void clear();
    void initialize(const mxArray* operators_, const dotk::types::problem_t & type_);

private:
    dotk::DOTk_MexArrayPtr m_Solve;
    dotk::DOTk_MexArrayPtr m_Hessian;
    dotk::DOTk_MexArrayPtr m_Residual;
    dotk::DOTk_MexArrayPtr m_Jacobian;
    dotk::DOTk_MexArrayPtr m_PartialDerivativeState;
    dotk::DOTk_MexArrayPtr m_PartialDerivativeControl;
    dotk::DOTk_MexArrayPtr m_AdjointPartialDerivative;
    dotk::DOTk_MexArrayPtr m_ApplyInverseJacobianState;
    dotk::DOTk_MexArrayPtr m_PartialDerivativeStateState;
    dotk::DOTk_MexArrayPtr m_AdjointPartialDerivativeState;
    dotk::DOTk_MexArrayPtr m_PartialDerivativeStateControl;
    dotk::DOTk_MexArrayPtr m_PartialDerivativeControlState;
    dotk::DOTk_MexArrayPtr m_AdjointPartialDerivativeControl;
    dotk::DOTk_MexArrayPtr m_PartialDerivativeControlControl;
    dotk::DOTk_MexArrayPtr m_ApplyAdjointInverseJacobianState;

private:
    DOTk_MexEqualityConstraint(const dotk::DOTk_MexEqualityConstraint<Type> &);
    dotk::DOTk_MexEqualityConstraint<Type> & operator=(const dotk::DOTk_MexEqualityConstraint<Type> &);
};

}

#endif /* DOTK_MEXEQUALITYCONSTRAINT_HPP_ */
