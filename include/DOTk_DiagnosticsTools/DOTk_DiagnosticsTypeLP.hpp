/*
 * DOTk_DiagnosticsTypeLP.hpp
 *
 *  Created on: Mar 29, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DIAGNOSTICSTYPELP_HPP_
#define DOTK_DIAGNOSTICSTYPELP_HPP_

#include "DOTk_EqualityTypeLP.hpp"
#include "DOTk_ObjectiveTypeLP.hpp"
#include "DOTk_InequalityTypeLP.hpp"
#include "DOTk_DerivativeDiagnosticsTool.hpp"

namespace dotk
{

class DOTk_Dual;
class DOTk_Variable;

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_EqualityConstraint;
template<class Type>
class DOTk_InequalityConstraint;

class DOTk_DiagnosticsTypeLP : public dotk::DOTk_DerivativeDiagnosticsTool
{
public:
    explicit DOTk_DiagnosticsTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    DOTk_DiagnosticsTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                           const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    DOTk_DiagnosticsTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                           const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                           const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    ~DOTk_DiagnosticsTypeLP();

    void checkObjectiveGradient(const dotk::DOTk_Variable & primal_, std::ostringstream & msg_);
    void checkObjectiveHessian(const dotk::DOTk_Variable & primal_, std::ostringstream & msg_);
    void checkEqualityConstraintJacobian(const dotk::DOTk_Variable & primal_, std::ostringstream & msg_);
    Real checkEqualityConstraintAdjointJacobian(const dotk::DOTk_Variable & primal_,
                                                const dotk::DOTk_Dual & dual_,
                                                std::ostringstream & msg_);
    void checkEqualityConstraintJacobianDerivative(const dotk::DOTk_Variable & primal_,
                                                   const dotk::DOTk_Dual & dual_,
                                                   std::ostringstream & msg_);
    void checkInequalityConstraintJacobian(const dotk::DOTk_Variable & primal_, std::ostringstream & msg_);

private:
    template<typename Functor, typename DerivativeOperators>
    void checkScalarValuedFunctionFirstDerivative(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                  const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                  const Functor & functor_,
                                                  const DerivativeOperators & operators_);
    template<typename Functor, typename DerivativeOperators>
    void checkScalarValuedFunctionSecondDerivative(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                   const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                   const Functor & first_derivative_,
                                                   const DerivativeOperators & second_derivative_);
    void checkVectorValuedFunctionFirstDerivative(const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                  const dotk::lp::EqualityConstraintResidual & function_,
                                                  const dotk::lp::EqualityConstraintFirstDerivative & first_derivative_,
                                                  const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);
    Real checkAdjointFirstDerivativeVectorValuedFunction(const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                         const dotk::lp::EqualityConstraintFirstDerivative & first_derivative_,
                                                         const dotk::lp::EqualityConstraintAdjointFirstDerivative & adjoint_first_derivative_,
                                                         const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                         const std::tr1::shared_ptr<dotk::vector<Real> > & dual_);
    void checkSecondDerivativeVectorValuedFunction(const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                   const dotk::lp::EqualityConstraintAdjointFirstDerivative & adjoint_first_derivative_,
                                                   const dotk::lp::EqualityConstraintSecondDerivative & adjoint_second_derivative_,
                                                   const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                   const std::tr1::shared_ptr<dotk::vector<Real> > & dual_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_TrueDerivative;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OriginalPrimal;

    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityConstraint;
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > m_InequalityConstraint;

private:
    DOTk_DiagnosticsTypeLP(const dotk::DOTk_DiagnosticsTypeLP &);
    dotk::DOTk_DiagnosticsTypeLP & operator=(const dotk::DOTk_DiagnosticsTypeLP &);
};

}

#endif /* DOTK_DIAGNOSTICSTYPELP_HPP_ */
