/*
 * DOTk_DiagnosticsObjectiveTypeNP.hpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DIAGNOSTICSOBJECTIVETYPENP_HPP_
#define DOTK_DIAGNOSTICSOBJECTIVETYPENP_HPP_

#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_DerivativeDiagnosticsTool.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"

namespace dotk
{

class DOTk_State;
class DOTk_Control;

template<typename ScalarType>
class Vector;

class DOTk_DiagnosticsObjectiveTypeNP : public dotk::DOTk_DerivativeDiagnosticsTool
{
public:
    DOTk_DiagnosticsObjectiveTypeNP(const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    ~DOTk_DiagnosticsObjectiveTypeNP();

    void checkPartialDerivativeControl(const dotk::DOTk_State & state_,
                                       const dotk::DOTk_Control & control_,
                                       std::ostringstream & msg_);
    void checkPartialDerivativeState(const dotk::DOTk_State & state_,
                                     const dotk::DOTk_Control & control_,
                                     std::ostringstream & msg_);
    void checkPartialDerivativeStateState(const dotk::DOTk_State & state_,
                                          const dotk::DOTk_Control & control_,
                                          std::ostringstream & msg_);
    void checkPartialDerivativeStateControl(const dotk::DOTk_State & state_,
                                            const dotk::DOTk_Control & control_,
                                            std::ostringstream & msg_);
    void checkPartialDerivativeControlControl(const dotk::DOTk_State & state_,
                                              const dotk::DOTk_Control & control_,
                                              std::ostringstream & msg_);
    void checkPartialDerivativeControlState(const dotk::DOTk_State & state_,
                                            const dotk::DOTk_Control & control_,
                                            std::ostringstream & msg_);

private:
    template<typename ScalarValuedFunction, typename ScalarValuedFunctionFirstDerivative>
    void checkScalarValuedFunctionFirstDerivative(const std::shared_ptr<dotk::Vector<Real> > & perturbation_vec_,
                                                  const ScalarValuedFunction & function_,
                                                  const ScalarValuedFunctionFirstDerivative & first_derivative_,
                                                  dotk::nlp::variables & variables_);
    template<typename ScalarValuedFunctionFirstDerivative, typename ScalarValuedFunctionSecondDerivative>
    void checkScalarValuedFunctionSecondDerivative(const std::shared_ptr<dotk::Vector<Real> > & perturbation_vec_,
                                                   const ScalarValuedFunctionFirstDerivative & first_derivative_,
                                                   const ScalarValuedFunctionSecondDerivative & second_derivative_,
                                                   dotk::nlp::variables & variables_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_OriginalField;
    std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;

private:
    DOTk_DiagnosticsObjectiveTypeNP(const dotk::DOTk_DiagnosticsObjectiveTypeNP &);
    dotk::DOTk_DiagnosticsObjectiveTypeNP operator=(const dotk::DOTk_DiagnosticsObjectiveTypeNP &);
};

}

#endif /* DOTK_DIAGNOSTICSOBJECTIVETYPENP_HPP_ */
