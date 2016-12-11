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

template<typename ScalarType>
class Vector;

template<typename ScalarType>
class DOTk_MexEqualityConstraint : public DOTk_EqualityConstraint<ScalarType>
{
public:
    DOTk_MexEqualityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexEqualityConstraint();

    virtual void residual(const dotk::Vector<ScalarType> & primal_, dotk::Vector<ScalarType> & output_);
    virtual void jacobian(const dotk::Vector<ScalarType> & primal_,
                          const dotk::Vector<ScalarType> & vector_,
                          dotk::Vector<ScalarType> & jacobian_times_vector_);
    virtual void adjointJacobian(const dotk::Vector<ScalarType> & primal_,
                                 const dotk::Vector<ScalarType> & dual_,
                                 dotk::Vector<ScalarType> & output_);
    virtual void hessian(const dotk::Vector<ScalarType> & primal_,
                         const dotk::Vector<ScalarType> & dual_,
                         const dotk::Vector<ScalarType> & vector_,
                         dotk::Vector<ScalarType> & output_);


    virtual void solve(const dotk::Vector<ScalarType> & control_, dotk::Vector<ScalarType> & output_);
    virtual void applyInverseJacobianState(const dotk::Vector<ScalarType> & state_,
                                              const dotk::Vector<ScalarType> & control_,
                                              const dotk::Vector<ScalarType> & rhs_,
                                              dotk::Vector<ScalarType> & output_);
    virtual void applyAdjointInverseJacobianState(const dotk::Vector<ScalarType> & state_,
                                                  const dotk::Vector<ScalarType> & control_,
                                                  const dotk::Vector<ScalarType> & rhs_,
                                                  dotk::Vector<ScalarType> & output_);
    virtual void residual(const dotk::Vector<ScalarType> & state_,
                          const dotk::Vector<ScalarType> & control_,
                          dotk::Vector<ScalarType> & output_);
    virtual void partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                        const dotk::Vector<ScalarType> & control_,
                                        const dotk::Vector<ScalarType> & vector_,
                                        dotk::Vector<ScalarType> & output_);
    virtual void partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                          const dotk::Vector<ScalarType> & control_,
                                          const dotk::Vector<ScalarType> & vector_,
                                          dotk::Vector<ScalarType> & output_);
    virtual void adjointPartialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               dotk::Vector<ScalarType> & output_);
    virtual void adjointPartialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                                 const dotk::Vector<ScalarType> & control_,
                                                 const dotk::Vector<ScalarType> & dual_,
                                                 dotk::Vector<ScalarType> & output_);
    virtual void partialDerivativeStateState(const dotk::Vector<ScalarType> & state_,
                                             const dotk::Vector<ScalarType> & control_,
                                             const dotk::Vector<ScalarType> & dual_,
                                             const dotk::Vector<ScalarType> & vector_,
                                             dotk::Vector<ScalarType> & output_);
    virtual void partialDerivativeStateControl(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_);
    virtual void partialDerivativeControlControl(const dotk::Vector<ScalarType> & state_,
                                                 const dotk::Vector<ScalarType> & control_,
                                                 const dotk::Vector<ScalarType> & dual_,
                                                 const dotk::Vector<ScalarType> & vector_,
                                                 dotk::Vector<ScalarType> & output_);
    virtual void partialDerivativeControlState(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_);

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
    DOTk_MexEqualityConstraint(const dotk::DOTk_MexEqualityConstraint<ScalarType> &);
    dotk::DOTk_MexEqualityConstraint<ScalarType> & operator=(const dotk::DOTk_MexEqualityConstraint<ScalarType> &);
};

}

#endif /* DOTK_MEXEQUALITYCONSTRAINT_HPP_ */
