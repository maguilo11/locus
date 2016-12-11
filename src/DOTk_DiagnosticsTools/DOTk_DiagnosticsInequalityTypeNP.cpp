/*
 * DOTk_DiagnosticsInequalityTypeNP.cpp
 *
 *  Created on: Jun 28, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <sstream>

#include "vector.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_InequalityTypeNP.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_DiagnosticsInequalityTypeNP.hpp"

namespace dotk
{

DOTk_DiagnosticsInequalityTypeNP::DOTk_DiagnosticsInequalityTypeNP
(const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_) :
        dotk::DOTk_DerivativeDiagnosticsTool(),
        m_OriginalField(),
        m_InequalityConstraint(inequality_.begin(), inequality_.end())
{
}

DOTk_DiagnosticsInequalityTypeNP::~DOTk_DiagnosticsInequalityTypeNP()
{
}

void DOTk_DiagnosticsInequalityTypeNP::checkPartialDerivativeControl(const dotk::DOTk_State & state_,
                                                                     const dotk::DOTk_Control & control_,
                                                                     std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(control_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::Z);
    bool invalid_codomain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(control_.data(),
                                                                                                     function_name,
                                                                                                     msg_);
    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mControl->clone();
    m_OriginalField->copy(*vars.mControl);

    std::tr1::shared_ptr<dotk::Vector<Real> > perturbtation = vars.mControl->clone();
    dotk::gtools::generateRandomVector(perturbtation);

    dotk::nlp::InequalityConstraintEvaluate evaluate;
    dotk::nlp::InequalityConstraintFirstDerivativeWrtControl partial_derivative;

    dotk::types::variable_t codomain = dotk::DOTk_DerivativeDiagnosticsTool::getCodomain();
    std::tr1::shared_ptr<dotk::Vector<Real> > true_partial_derivative = dotk::nlp::clone(vars, codomain);

    std::ostringstream message;
    const size_t num_inequality_constraints = m_InequalityConstraint.size();
    for(size_t index = 0; index < num_inequality_constraints; ++ index)
    {
        message << "\n CHECK INEQUALITY CONSTRAINT WITH ID = " << index << " PARTIAL DERIVATIVE WITH RESPECT TO CONTROLS\n";
        dotk::DOTk_DerivativeDiagnosticsTool::addMessage(message);

        partial_derivative(m_InequalityConstraint[index], vars.mState, vars.mControl, true_partial_derivative);
        this->checkScalarValuedFunctionFirstDerivative(evaluate, *perturbtation, *true_partial_derivative, vars, m_InequalityConstraint[index]);

        message.str("");
    }

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsInequalityTypeNP::checkPartialDerivativeState(const dotk::DOTk_State & state_,
                                                                   const dotk::DOTk_Control & control_,
                                                                   std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(state_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::U);

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(state_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mState->clone();
    m_OriginalField->copy(*vars.mState);

    std::tr1::shared_ptr<dotk::Vector<Real> > delta_state = vars.mState->clone();
    dotk::gtools::generateRandomVector(delta_state);

    dotk::nlp::InequalityConstraintEvaluate evaluate;
    dotk::nlp::InequalityConstraintFirstDerivativeWrtState partial_derivative;

    dotk::types::variable_t codomain = dotk::DOTk_DerivativeDiagnosticsTool::getCodomain();
    std::tr1::shared_ptr<dotk::Vector<Real> > true_partial_derivative = dotk::nlp::clone(vars, codomain);

    std::ostringstream message;
    const size_t num_inequality_constraints = m_InequalityConstraint.size();
    for(size_t index = 0; index < num_inequality_constraints; ++ index)
    {
        message << "\n CHECK INEQUALITY CONSTRAINT WITH ID = " << index << " PARTIAL DERIVATIVE WITH RESPECT TO STATES\n";
        dotk::DOTk_DerivativeDiagnosticsTool::addMessage(message);

        partial_derivative(m_InequalityConstraint[index], vars.mState, vars.mControl, true_partial_derivative);
        this->checkScalarValuedFunctionFirstDerivative(evaluate, *delta_state, *true_partial_derivative, vars, m_InequalityConstraint[index]);

        message.str("");
    }

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

template<typename ScalarValuedFunction>
void DOTk_DiagnosticsInequalityTypeNP::checkScalarValuedFunctionFirstDerivative(const ScalarValuedFunction & function_,
                                                                                const dotk::Vector<Real> & perturbation_,
                                                                                const dotk::Vector<Real> & true_partial_derivative_,
                                                                                dotk::nlp::variables & variables_,
                                                                                std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > inequality_)
{
    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();

    dotk::types::derivative_t derivative_type = dotk::DOTk_DerivativeDiagnosticsTool::getDerivativeType();
    Real gradient_dot_direction = true_partial_derivative_.dot(perturbation_);

    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);

        // four point finite difference approximation
        dotk::nlp::perturbField(epsilon, perturbation_, variables_, derivative_type);
        Real objective_function_value_one = function_(inequality_, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(-epsilon, perturbation_, variables_, derivative_type);
        Real objective_function_value_two = function_(inequality_, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(2.) * epsilon, perturbation_, variables_, derivative_type);
        Real objective_function_value_three = function_(inequality_, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(-2.) * epsilon, perturbation_, variables_, derivative_type);
        Real objective_function_value_four = function_(inequality_, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        Real numerator = -objective_function_value_three + static_cast<Real>(8.) * objective_function_value_one
                - static_cast<Real>(8.) * objective_function_value_two + objective_function_value_four;
        Real denominator = static_cast<Real>(12.) * epsilon;
        Real finite_difference_approximation = numerator / denominator;
        Real relative_error = std::abs(gradient_dot_direction - finite_difference_approximation);

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(gradient_dot_direction,
                                                                              finite_difference_approximation,
                                                                              relative_error,
                                                                              epsilon);
    }
}

}
