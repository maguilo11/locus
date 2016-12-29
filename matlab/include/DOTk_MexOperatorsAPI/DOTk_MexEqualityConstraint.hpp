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

template<typename ScalarType>
class Vector;

class DOTk_MexEqualityConstraint : public DOTk_EqualityConstraint<double>
{
public:
    DOTk_MexEqualityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexEqualityConstraint();

    virtual void residual(const dotk::Vector<double> & primal_, dotk::Vector<double> & output_);
    virtual void jacobian(const dotk::Vector<double> & primal_,
                          const dotk::Vector<double> & vector_,
                          dotk::Vector<double> & jacobian_times_vector_);
    virtual void adjointJacobian(const dotk::Vector<double> & primal_,
                                 const dotk::Vector<double> & dual_,
                                 dotk::Vector<double> & output_);
    virtual void hessian(const dotk::Vector<double> & primal_,
                         const dotk::Vector<double> & dual_,
                         const dotk::Vector<double> & vector_,
                         dotk::Vector<double> & output_);


    virtual void solve(const dotk::Vector<double> & control_, dotk::Vector<double> & output_);
    virtual void applyInverseJacobianState(const dotk::Vector<double> & state_,
                                              const dotk::Vector<double> & control_,
                                              const dotk::Vector<double> & rhs_,
                                              dotk::Vector<double> & output_);
    virtual void applyAdjointInverseJacobianState(const dotk::Vector<double> & state_,
                                                  const dotk::Vector<double> & control_,
                                                  const dotk::Vector<double> & rhs_,
                                                  dotk::Vector<double> & output_);
    virtual void residual(const dotk::Vector<double> & state_,
                          const dotk::Vector<double> & control_,
                          dotk::Vector<double> & output_);
    virtual void partialDerivativeState(const dotk::Vector<double> & state_,
                                        const dotk::Vector<double> & control_,
                                        const dotk::Vector<double> & vector_,
                                        dotk::Vector<double> & output_);
    virtual void partialDerivativeControl(const dotk::Vector<double> & state_,
                                          const dotk::Vector<double> & control_,
                                          const dotk::Vector<double> & vector_,
                                          dotk::Vector<double> & output_);
    virtual void adjointPartialDerivativeState(const dotk::Vector<double> & state_,
                                               const dotk::Vector<double> & control_,
                                               const dotk::Vector<double> & dual_,
                                               dotk::Vector<double> & output_);
    virtual void adjointPartialDerivativeControl(const dotk::Vector<double> & state_,
                                                 const dotk::Vector<double> & control_,
                                                 const dotk::Vector<double> & dual_,
                                                 dotk::Vector<double> & output_);
    virtual void partialDerivativeStateState(const dotk::Vector<double> & state_,
                                             const dotk::Vector<double> & control_,
                                             const dotk::Vector<double> & dual_,
                                             const dotk::Vector<double> & vector_,
                                             dotk::Vector<double> & output_);
    virtual void partialDerivativeStateControl(const dotk::Vector<double> & state_,
                                               const dotk::Vector<double> & control_,
                                               const dotk::Vector<double> & dual_,
                                               const dotk::Vector<double> & vector_,
                                               dotk::Vector<double> & output_);
    virtual void partialDerivativeControlControl(const dotk::Vector<double> & state_,
                                                 const dotk::Vector<double> & control_,
                                                 const dotk::Vector<double> & dual_,
                                                 const dotk::Vector<double> & vector_,
                                                 dotk::Vector<double> & output_);
    virtual void partialDerivativeControlState(const dotk::Vector<double> & state_,
                                               const dotk::Vector<double> & control_,
                                               const dotk::Vector<double> & dual_,
                                               const dotk::Vector<double> & vector_,
                                               dotk::Vector<double> & output_);

private:
    void clear();
    void initialize(const mxArray* operators_, const dotk::types::problem_t & type_);

private:
    mxArray* m_Solve;
    mxArray* m_Residual;
    mxArray* m_ApplyInverseJacobianState;
    mxArray* m_ApplyAdjointInverseJacobianState;

    mxArray* m_Jacobian;
    mxArray* m_AdjointJacobian;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_AdjointPartialDerivativeState;
    mxArray* m_AdjointPartialDerivativeControl;

    mxArray* m_Hessian;
    mxArray* m_PartialDerivativeStateState;
    mxArray* m_PartialDerivativeStateControl;
    mxArray* m_PartialDerivativeControlState;
    mxArray* m_PartialDerivativeControlControl;

private:
    DOTk_MexEqualityConstraint(const dotk::DOTk_MexEqualityConstraint &);
    dotk::DOTk_MexEqualityConstraint & operator=(const dotk::DOTk_MexEqualityConstraint &);
};

}

#endif /* DOTK_MEXEQUALITYCONSTRAINT_HPP_ */
