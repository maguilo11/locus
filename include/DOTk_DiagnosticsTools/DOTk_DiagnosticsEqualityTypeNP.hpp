/*
 * DOTk_DiagnosticsEqualityTypeNP.hpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DIAGNOSTICSEQUALITYTYPENP_HPP_
#define DOTK_DIAGNOSTICSEQUALITYTYPENP_HPP_

#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_DerivativeDiagnosticsTool.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"

namespace dotk
{

class DOTk_Dual;
class DOTk_State;
class DOTk_Control;

template<typename ScalarType>
class Vector;

class DOTk_DiagnosticsEqualityTypeNP : public dotk::DOTk_DerivativeDiagnosticsTool
{
public:
    explicit DOTk_DiagnosticsEqualityTypeNP(const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    ~DOTk_DiagnosticsEqualityTypeNP();

    void checkPartialDerivativeState(const dotk::DOTk_State & state_,
                                     const dotk::DOTk_Control & control_,
                                     std::ostringstream & msg_);
    void checkPartialDerivativeControl(const dotk::DOTk_State & state_,
                                       const dotk::DOTk_Control & control_,
                                       std::ostringstream & msg_);

    Real checkAdjointPartialDerivativeState(const dotk::DOTk_State & state_,
                                            const dotk::DOTk_Control & control_,
                                            const dotk::DOTk_Dual & dual_,
                                            std::ostringstream & msg_);
    Real checkAdjointPartialDerivativeControl(const dotk::DOTk_State & state_,
                                              const dotk::DOTk_Control & control_,
                                              const dotk::DOTk_Dual & dual_,
                                              std::ostringstream & msg_);

    void checkAdjointPartialDerivativeControlControl(const dotk::DOTk_State & state_,
                                                     const dotk::DOTk_Control & control_,
                                                     const dotk::DOTk_Dual & dual_,
                                                     std::ostringstream & msg_);
    void checkAdjointPartialDerivativeControlState(const dotk::DOTk_State& state_,
                                                   const dotk::DOTk_Control& control_,
                                                   const dotk::DOTk_Dual& dual_,
                                                   std::ostringstream& msg_);
    void checkAdjointPartialDerivativeStateState(const dotk::DOTk_State & state_,
                                                 const dotk::DOTk_Control & control_,
                                                 const dotk::DOTk_Dual & dual_,
                                                 std::ostringstream & msg_);
    void checkAdjointPartialDerivativeStateControl(const dotk::DOTk_State& state_,
                                                   const dotk::DOTk_Control& control_,
                                                   const dotk::DOTk_Dual& dual_,
                                                   std::ostringstream& msg_);

private:
    template<typename VectorValuedFunction, typename VectorValuedFunctionFirstDerivative>
    void checkVectorValuedFunctionPartialDerivative(const std::shared_ptr<dotk::Vector<Real> > & perturbation_vec_,
                                                    const VectorValuedFunction & function_,
                                                    const VectorValuedFunctionFirstDerivative & first_derivative_,
                                                    dotk::nlp::variables & variables_);
    template<typename VectorValuedFunctionFirstDerivative, typename AdjointVectorValuedFunctionFirstDerivative>
    Real checkAdjointPartialDerivativeVectorValuedFunction(const std::shared_ptr<dotk::Vector<Real> > & perturbation_vec_,
                                                           const VectorValuedFunctionFirstDerivative& first_derivative_,
                                                           const AdjointVectorValuedFunctionFirstDerivative& adjoint_first_derivative_,
                                                           dotk::nlp::variables & variables_);
    template<typename AdjointVectorValuedFunctionFirstDerivative, typename AdjointVectorValuedFunctionSecondDerivative>
    void checkAdjointSecondPartialDerivativeVectorValuedFunction(const std::shared_ptr<dotk::Vector<Real> > & direction_,
                                                                 const AdjointVectorValuedFunctionFirstDerivative& adjoint_first_derivative_,
                                                                 const AdjointVectorValuedFunctionSecondDerivative& adjoint_second_derivative_,
                                                                 dotk::nlp::variables & variables_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_OriginalField;
    std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_ContinuousOperators;

private:
    DOTk_DiagnosticsEqualityTypeNP(const dotk::DOTk_DiagnosticsEqualityTypeNP &);
    dotk::DOTk_DiagnosticsEqualityTypeNP operator=(const dotk::DOTk_DiagnosticsEqualityTypeNP &);
};

}

#endif /* DOTK_DIAGNOSTICSEQUALITYTYPENP_HPP_ */
