/*
 * DOTk_DiagnosticsInequalityTypeNP.hpp
 *
 *  Created on: Jun 28, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DIAGNOSTICSINEQUALITYTYPENP_HPP_
#define DOTK_DIAGNOSTICSINEQUALITYTYPENP_HPP_

#include <vector>

#include "DOTk_InequalityConstraint.hpp"
#include "DOTk_DerivativeDiagnosticsTool.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"

namespace dotk
{

class DOTk_State;
class DOTk_Control;

template<typename ScalarType>
class Vector;

class DOTk_DiagnosticsInequalityTypeNP : public dotk::DOTk_DerivativeDiagnosticsTool
{
public:
    explicit DOTk_DiagnosticsInequalityTypeNP(const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    ~DOTk_DiagnosticsInequalityTypeNP();

    void checkPartialDerivativeControl(const dotk::DOTk_State & state_,
                                       const dotk::DOTk_Control & control_,
                                       std::ostringstream & msg_);
    void checkPartialDerivativeState(const dotk::DOTk_State & state_,
                                     const dotk::DOTk_Control & control_,
                                     std::ostringstream & msg_);

private:
    template<typename ScalarValuedFunction>
    void checkScalarValuedFunctionFirstDerivative(const ScalarValuedFunction & function_,
                                                  const dotk::Vector<Real> & perturbation_,
                                                  const dotk::Vector<Real> & true_partial_derivative_,
                                                  dotk::nlp::variables & variables_,
                                                  std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > inequality_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OriginalField;
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > m_InequalityConstraint;

private:
    DOTk_DiagnosticsInequalityTypeNP(const dotk::DOTk_DiagnosticsInequalityTypeNP &);
    dotk::DOTk_DiagnosticsInequalityTypeNP operator=(const dotk::DOTk_DiagnosticsInequalityTypeNP &);
};

}

#endif /* DOTK_DIAGNOSTICSINEQUALITYTYPENP_HPP_ */
