/*
 * DOTk_DerivativeDiagnosticsTool.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iomanip>
#include <sstream>

#include "vector.hpp"
#include "DOTk_DerivativeDiagnosticsTool.hpp"

namespace dotk
{

DOTk_DerivativeDiagnosticsTool::DOTk_DerivativeDiagnosticsTool() :
        mSuperScriptLowerLimit(-3),
        mSuperScriptUpperLimit(5),
        mPrintFiniteDifferenceDiagnostics(false),
        mCodomain(dotk::types::UNDEFINED_VARIABLE),
        mDerivativeType(dotk::types::ZERO_ORDER_DERIVATIVE),
        mFiniteDifferenceDiagnosticsMsg()
{
}

DOTk_DerivativeDiagnosticsTool::~DOTk_DerivativeDiagnosticsTool()
{
}

void DOTk_DerivativeDiagnosticsTool::printFiniteDifferenceDiagnostics(bool print_)
{
    mPrintFiniteDifferenceDiagnostics = print_;
}

bool DOTk_DerivativeDiagnosticsTool::willFiniteDifferenceDiagnosticsBePrinted() const
{
    return (mPrintFiniteDifferenceDiagnostics);
}

void DOTk_DerivativeDiagnosticsTool::setCodomain(dotk::types::variable_t type_)
{
    mCodomain = type_;
}

dotk::types::variable_t DOTk_DerivativeDiagnosticsTool::getCodomain() const
{
    return (mCodomain);
}

void DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::derivative_t type_)
{
    mDerivativeType = type_;
}

dotk::types::derivative_t DOTk_DerivativeDiagnosticsTool::getDerivativeType() const
{
    return (mDerivativeType);
}

void DOTk_DerivativeDiagnosticsTool::setFiniteDifferenceDiagnosticsSuperScripts(Int lower_limit_, Int upper_limit_)
{
    mSuperScriptLowerLimit = lower_limit_;
    mSuperScriptUpperLimit = upper_limit_;
}

Int DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit() const
{
    return (mSuperScriptLowerLimit);
}

Int DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit() const
{
    return (mSuperScriptUpperLimit);
}

void DOTk_DerivativeDiagnosticsTool::addMessage(std::ostringstream & msg_)
{
    mFiniteDifferenceDiagnosticsMsg << msg_.str().c_str();
}

void DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(std::ostringstream & msg_)
{
    msg_.str(mFiniteDifferenceDiagnosticsMsg.str());
    mFiniteDifferenceDiagnosticsMsg.str("");
}

void DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(const Real expected_value_,
                                                                     const Real finite_difference_approximation_,
                                                                     const Real relative_difference_,
                                                                     const Real epsilon_)
{
    mFiniteDifferenceDiagnosticsMsg << std::setw(10) << std::right << std::scientific << std::setprecision(4)
            << "Expected Value = " << expected_value_ << std::setw(10) << ", Finite Difference Appx. = "
            << finite_difference_approximation_ << std::setw(10) << ", Relative Difference = " << relative_difference_
            << std::setw(10) << ", Epsilon = " << epsilon_ << "\n";
}

bool DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(const std::shared_ptr<dotk::Vector<Real> > & field_,
                                                           const std::string & function_name_,
                                                           std::ostringstream & output_msg_)
{
    bool exit_function = false;
    if(field_->size() <= 0)
    {
        exit_function = true;
        output_msg_ << "DOTk ERROR: ZERO dimension domain in " << function_name_.c_str() << ", EXIT FUNCTION\n";
    }
    return (exit_function);
}

bool DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(const std::shared_ptr<dotk::Vector<Real> > & field_,
                                                             const std::string & function_name_,
                                                             std::ostringstream & output_msg_)
{
    bool exit_function = false;
    if(field_->size() <= 0)
    {
        exit_function = true;
        output_msg_ << "DOTk ERROR: ZERO dimension codomain in " << function_name_.c_str() << ", EXIT FUNCTION\n";
    }
    return (exit_function);
}

}
