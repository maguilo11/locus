/*
 * DOTk_DerivativeDiagnosticsTool.hpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DERIVATIVEDIAGNOSTICSTOOL_HPP_
#define DOTK_DERIVATIVEDIAGNOSTICSTOOL_HPP_

#include "vector.hpp"

namespace dotk
{

class DOTk_DerivativeDiagnosticsTool
{
public:
    DOTk_DerivativeDiagnosticsTool();
    ~DOTk_DerivativeDiagnosticsTool();

    void printFiniteDifferenceDiagnostics(bool print_);
    bool willFiniteDifferenceDiagnosticsBePrinted() const;
    void setCodomain(dotk::types::variable_t type_);
    dotk::types::variable_t getCodomain() const;
    void setDerivativeType(dotk::types::derivative_t type_);
    dotk::types::derivative_t getDerivativeType() const;
    void setFiniteDifferenceDiagnosticsSuperScripts(Int lower_limit_, Int upper_limit_);
    Int getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit() const;
    Int getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit() const;

    void addMessage(std::ostringstream & msg_);
    void getFiniteDifferenceDiagnosticsMsg(std::ostringstream & msg_);

    void saveFiniteDifferenceDiagnostics(const Real expected_value_,
                                         const Real finite_difference_approximation_,
                                         const Real relative_difference_,
                                         const Real epsilon_);
    bool checkDomainDimensions(const std::tr1::shared_ptr<dotk::vector<Real> > & field_,
                               const std::string & function_name_,
                               std::ostringstream & output_msg_);
    bool checkCodomainDimensions(const std::tr1::shared_ptr<dotk::vector<Real> > & field_,
                                 const std::string & function_name_,
                                 std::ostringstream & output_msg_);

private:
    Int mSuperScriptLowerLimit;
    Int mSuperScriptUpperLimit;

    bool mPrintFiniteDifferenceDiagnostics;

    dotk::types::variable_t mCodomain;
    dotk::types::derivative_t mDerivativeType;
    std::ostringstream mFiniteDifferenceDiagnosticsMsg;

private:
    DOTk_DerivativeDiagnosticsTool(const dotk::DOTk_DerivativeDiagnosticsTool &);
    dotk::DOTk_DerivativeDiagnosticsTool operator=(const dotk::DOTk_DerivativeDiagnosticsTool &);
};

}

#endif /* DOTK_DERIVATIVEDIAGNOSTICSTOOL_HPP_ */
