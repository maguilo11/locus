/*
 * DOTk_DiagnosticsObjectiveTypeNP.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "vector.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_ObjectiveTypeNP.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_DiagnosticsObjectiveTypeNP.hpp"

namespace dotk
{

DOTk_DiagnosticsObjectiveTypeNP::DOTk_DiagnosticsObjectiveTypeNP(const std::tr1::shared_ptr<
        dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_DerivativeDiagnosticsTool::DOTk_DerivativeDiagnosticsTool(),
        m_OriginalField(),
        m_ObjectiveFunction(objective_)
{
}

DOTk_DiagnosticsObjectiveTypeNP::~DOTk_DiagnosticsObjectiveTypeNP()
{
}

void DOTk_DiagnosticsObjectiveTypeNP::checkPartialDerivativeControl(const dotk::DOTk_State & state_,
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

    std::tr1::shared_ptr<dotk::vector<Real> > delta_control = vars.mControl->clone();
    dotk::gtools::generateRandomVector(delta_control);

    dotk::nlp::ObjectiveValue objective_function;
    dotk::nlp::PartialDerivativeObjectiveControl first_derivative;
    this->checkScalarValuedFunctionFirstDerivative(delta_control, objective_function, first_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsObjectiveTypeNP::checkPartialDerivativeState(const dotk::DOTk_State & state_,
                                                                   const dotk::DOTk_Control & control_,
                                                                   std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(state_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::U);
    bool invalid_codomain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(state_.data(),
                                                                                                     function_name,
                                                                                                     msg_);
    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mState->clone();
    m_OriginalField->copy(*vars.mState);

    std::tr1::shared_ptr<dotk::vector<Real> > delta_state = vars.mState->clone();
    dotk::gtools::generateRandomVector(delta_state);

    dotk::nlp::ObjectiveValue objective_function;
    dotk::nlp::PartialDerivativeObjectiveState first_derivative;
    this->checkScalarValuedFunctionFirstDerivative(delta_state, objective_function, first_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsObjectiveTypeNP::checkPartialDerivativeStateState(const dotk::DOTk_State & state_,
                                                                       const dotk::DOTk_Control & control_,
                                                                       std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeStateState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(state_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::UU);
    bool invalid_codomain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(state_.data(),
                                                                                                     function_name,
                                                                                                     msg_);
    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mState->clone();
    m_OriginalField->copy(*vars.mState);

    std::tr1::shared_ptr<dotk::vector<Real> > delta_state = vars.mState->clone();
    dotk::gtools::generateRandomVector(delta_state);

    dotk::nlp::PartialDerivativeObjectiveState first_derivative;
    dotk::nlp::PartialDerivativeObjectiveStateState second_derivative;
    this->checkScalarValuedFunctionSecondDerivative(delta_state, first_derivative, second_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsObjectiveTypeNP::checkPartialDerivativeStateControl(const dotk::DOTk_State & state_,
                                                                           const dotk::DOTk_Control & control_,
                                                                           std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeStateControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(state_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::UZ);
    bool invalid_codomain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(state_.data(),
                                                                                                     function_name,
                                                                                                     msg_);
    bool invalid_domain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(control_.data(),
                                                                                                 function_name,
                                                                                                 msg_);
    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mControl->clone();
    m_OriginalField->copy(*vars.mControl);

    std::tr1::shared_ptr<dotk::vector<Real> > delta_control = vars.mControl->clone();
    dotk::gtools::generateRandomVector(delta_control);

    dotk::nlp::PartialDerivativeObjectiveState first_derivative;
    dotk::nlp::PartialDerivativeObjectiveStateControl second_derivative;
    this->checkScalarValuedFunctionSecondDerivative(delta_control, first_derivative, second_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsObjectiveTypeNP::checkPartialDerivativeControlControl(const dotk::DOTk_State & state_,
                                                                           const dotk::DOTk_Control & control_,
                                                                           std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeControlControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(control_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::ZZ);
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

    std::tr1::shared_ptr<dotk::vector<Real> > delta_control = vars.mControl->clone();
    dotk::gtools::generateRandomVector(delta_control);

    dotk::nlp::PartialDerivativeObjectiveControl first_derivative;
    dotk::nlp::PartialDerivativeObjectiveControlControl second_derivative;
    this->checkScalarValuedFunctionSecondDerivative(delta_control, first_derivative, second_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsObjectiveTypeNP::checkPartialDerivativeControlState(const dotk::DOTk_State & state_,
                                                                           const dotk::DOTk_Control & control_,
                                                                           std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeControlState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(control_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::ZU);
    bool invalid_codomain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(control_.data(),
                                                                                                     function_name,
                                                                                                     msg_);
    bool invalid_domain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(state_.data(),
                                                                                                 function_name,
                                                                                                 msg_);
    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mState->clone();
    m_OriginalField->copy(*vars.mState);

    std::tr1::shared_ptr<dotk::vector<Real> > delta_state = vars.mState->clone();
    dotk::gtools::generateRandomVector(delta_state);

    dotk::nlp::PartialDerivativeObjectiveControl first_derivative;
    dotk::nlp::PartialDerivativeObjectiveControlState second_derivative;
    this->checkScalarValuedFunctionSecondDerivative(delta_state, first_derivative, second_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

template<typename ScalarValuedFunction, typename ScalarValuedFunctionFirstDerivative>
void DOTk_DiagnosticsObjectiveTypeNP::checkScalarValuedFunctionFirstDerivative(const std::tr1::shared_ptr<
                                                                                       dotk::vector<Real> > & perturbation_vec_,
                                                                               const ScalarValuedFunction & function_,
                                                                               const ScalarValuedFunctionFirstDerivative & first_derivative_,
                                                                               dotk::nlp::variables & variables_)
{
    dotk::types::variable_t codomain = dotk::DOTk_DerivativeDiagnosticsTool::getCodomain();
    std::tr1::shared_ptr<dotk::vector<Real> > true_gradient = dotk::nlp::clone(variables_, codomain);
    first_derivative_(m_ObjectiveFunction, variables_.mState, variables_.mControl, true_gradient);
    Real gradient_dot_direction = true_gradient->dot(*perturbation_vec_);

    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();
    dotk::types::derivative_t derivative_type = dotk::DOTk_DerivativeDiagnosticsTool::getDerivativeType();
    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);

        // four point finite difference approximation
        dotk::nlp::perturbField(epsilon, *perturbation_vec_, variables_, derivative_type);
        Real objective_function_value_one = function_(m_ObjectiveFunction, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(-epsilon, *perturbation_vec_, variables_, derivative_type);
        Real objective_function_value_two = function_(m_ObjectiveFunction, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(2.) * epsilon, *perturbation_vec_, variables_, derivative_type);
        Real objective_function_value_three = function_(m_ObjectiveFunction, variables_.mState, variables_.mControl);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(-2.) * epsilon, *perturbation_vec_, variables_, derivative_type);
        Real objective_function_value_four = function_(m_ObjectiveFunction, variables_.mState, variables_.mControl);
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

template<typename ScalarValuedFunctionFirstDerivative, typename ScalarValuedFunctionSecondDerivative>
void DOTk_DiagnosticsObjectiveTypeNP::checkScalarValuedFunctionSecondDerivative(const std::tr1::shared_ptr<
                                                                                        dotk::vector<Real> > & perturbation_vec_,
                                                                                const ScalarValuedFunctionFirstDerivative & first_derivative_,
                                                                                const ScalarValuedFunctionSecondDerivative & second_derivative_,
                                                                                dotk::nlp::variables & variables_)
{
    dotk::types::variable_t codomain = dotk::DOTk_DerivativeDiagnosticsTool::getCodomain();
    std::tr1::shared_ptr<dotk::vector<Real> > true_second_derivative_times_direction = dotk::nlp::clone(variables_,
                                                                                                        codomain);
    second_derivative_(m_ObjectiveFunction,
                       variables_.mState,
                       variables_.mControl,
                       perturbation_vec_,
                       true_second_derivative_times_direction);

    Real norm_true_second_derivative_times_direction = true_second_derivative_times_direction->norm();

    std::tr1::shared_ptr<dotk::vector<Real> > finite_diff_second_derivative_times_direction =
            dotk::nlp::clone(variables_, codomain);
    std::tr1::shared_ptr<dotk::vector<Real> > first_derivative = dotk::nlp::clone(variables_, codomain);

    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();
    dotk::types::derivative_t derivative_type = dotk::DOTk_DerivativeDiagnosticsTool::getDerivativeType();
    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);
        finite_diff_second_derivative_times_direction->fill(0.);

        // four point finite difference approximation
        dotk::nlp::perturbField(epsilon, *perturbation_vec_, variables_, derivative_type);
        first_derivative_(m_ObjectiveFunction, variables_.mState, variables_.mControl, first_derivative);
        finite_diff_second_derivative_times_direction->axpy(static_cast<Real>(8.), *first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(-epsilon, *perturbation_vec_, variables_, derivative_type);
        first_derivative_(m_ObjectiveFunction, variables_.mState, variables_.mControl, first_derivative);
        finite_diff_second_derivative_times_direction->axpy(static_cast<Real>(-8.), *first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(2.) * epsilon, *perturbation_vec_, variables_, derivative_type);
        first_derivative_(m_ObjectiveFunction, variables_.mState, variables_.mControl, first_derivative);
        finite_diff_second_derivative_times_direction->axpy(static_cast<Real>(-1.), *first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(-2.) * epsilon, *perturbation_vec_, variables_, derivative_type);
        first_derivative_(m_ObjectiveFunction, variables_.mState, variables_.mControl, first_derivative);
        finite_diff_second_derivative_times_direction->axpy(static_cast<Real>(1.), *first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        Real alpha = static_cast<Real>(1.) / (static_cast<Real>(12.) * epsilon);
        finite_diff_second_derivative_times_direction->scale(alpha);

        Real norm_finite_diff_second_derivative_times_direction = finite_diff_second_derivative_times_direction->norm();

        finite_diff_second_derivative_times_direction->axpy(static_cast<Real>(-1.),
                                                            *true_second_derivative_times_direction);
        Real numerator = finite_diff_second_derivative_times_direction->norm();
        Real denominator = std::numeric_limits<Real>::epsilon() + norm_true_second_derivative_times_direction;
        Real relative_error = numerator / denominator;

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(norm_true_second_derivative_times_direction,
                                                                              norm_finite_diff_second_derivative_times_direction,
                                                                              relative_error,
                                                                              epsilon);
    }
}

}
