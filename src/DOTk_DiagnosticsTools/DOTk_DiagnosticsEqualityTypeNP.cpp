/*
 * DOTk_DiagnosticsEqualityTypeNP.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <string>
#include <limits>
#include <sstream>
#include <iomanip>

#include "vector.hpp"
#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_EqualityTypeNP.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_DiagnosticsEqualityTypeNP.hpp"

namespace dotk
{

DOTk_DiagnosticsEqualityTypeNP::DOTk_DiagnosticsEqualityTypeNP
(const std::tr1::shared_ptr< dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_DerivativeDiagnosticsTool::DOTk_DerivativeDiagnosticsTool(),
        m_OriginalField(),
        m_ContinuousOperators(equality_)
{
}

DOTk_DiagnosticsEqualityTypeNP::~DOTk_DiagnosticsEqualityTypeNP()
{
}

void DOTk_DiagnosticsEqualityTypeNP::checkPartialDerivativeState(const dotk::DOTk_State & state_,
                                                                 const dotk::DOTk_Control & control_,
                                                                 std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkPartialDerivativeState");
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

    dotk::nlp::EqualityConstraintResidual equality_constraint;
    dotk::nlp::EqualityConstraintFirstDerivativeState first_derivative;
    this->checkVectorValuedFunctionPartialDerivative(delta_state, equality_constraint, first_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsEqualityTypeNP::checkPartialDerivativeControl(const dotk::DOTk_State & state_,
                                                                   const dotk::DOTk_Control & control_,
                                                                   std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkPartialDerivativeControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(control_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::Z);

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(control_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mControl->clone();
    m_OriginalField->copy(*vars.mControl);

    std::tr1::shared_ptr<dotk::Vector<Real> > delta_control = vars.mControl->clone();
    dotk::gtools::generateRandomVector(delta_control);

    dotk::nlp::EqualityConstraintResidual equality_constraint;
    dotk::nlp::EqualityConstraintFirstDerivativeControl first_derivative;
    this->checkVectorValuedFunctionPartialDerivative(delta_control, equality_constraint, first_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

Real DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeState(const dotk::DOTk_State & state_,
                                                                        const dotk::DOTk_Control & control_,
                                                                        const dotk::DOTk_Dual & dual_,
                                                                        std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(dual_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::U);
    bool invalid_codomain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(dual_.data(),
                                                                                                     function_name,
                                                                                                     msg_);
    bool invalid_domain_dimensions = dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(state_.data(),
                                                                                                 function_name,
                                                                                                 msg_);
    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return (-1.);
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data(), *dual_.data());
    std::tr1::shared_ptr<dotk::Vector<Real> > perturbation = vars.mState->clone();
    dotk::gtools::generateRandomVector(perturbation);

    dotk::nlp::EqualityConstraintFirstDerivativeState jacobian;
    dotk::nlp::EqualityConstraintAdjointFirstDerivativeState adjoint_jacobian;
    Real absolute_difference =
            this->checkAdjointPartialDerivativeVectorValuedFunction(perturbation, jacobian, adjoint_jacobian, vars);

    msg_ << "The absolute difference between (dual_dot_first_derivative_times_direction) and "
            << "(adjoint_first_derivative_times_dual_dot_direction) = " << std::scientific << std::setprecision(6)
            << absolute_difference << "\n";

    return (absolute_difference);
}

Real DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeControl(const dotk::DOTk_State & state_,
                                                                          const dotk::DOTk_Control & control_,
                                                                          const dotk::DOTk_Dual & dual_,
                                                                          std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(dual_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::Z);

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(dual_.data(), function_name, msg_);
    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(control_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return (-1.);
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data(), *dual_.data());
    std::tr1::shared_ptr<dotk::Vector<Real> > perturbtation = vars.mControl->clone();
    dotk::gtools::generateRandomVector(perturbtation);

    dotk::nlp::EqualityConstraintFirstDerivativeControl jacobian;
    dotk::nlp::EqualityConstraintAdjointFirstDerivativeControl adjoint_jacobian;
    Real absolute_difference =
            this->checkAdjointPartialDerivativeVectorValuedFunction(perturbtation, jacobian, adjoint_jacobian, vars);

    msg_ << "The absolute difference between (dual_dot_first_derivative_times_direction) and "
            << "(adjoint_first_derivative_times_dual_dot_direction) = " << std::scientific << std::setprecision(6)
            << absolute_difference << "\n";

    return (absolute_difference);
}

void DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeControlControl(const dotk::DOTk_State & state_,
                                                                                 const dotk::DOTk_Control & control_,
                                                                                 const dotk::DOTk_Dual & dual_,
                                                                                 std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControlControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(control_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::ZZ);

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(control_.data(), function_name, msg_);
    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(control_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data(), *dual_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mControl->clone();
    m_OriginalField->copy(*vars.mControl);

    std::tr1::shared_ptr<dotk::Vector<Real> > perturbation = vars.mControl->clone();
    dotk::gtools::generateRandomVector(perturbation);

    dotk::nlp::EqualityConstraintAdjointFirstDerivativeControl adjoint_jacobian;
    dotk::nlp::EqualityConstraintSecondDerivativeControlControl adjoint_second_partial_derivative;
    this->checkAdjointSecondPartialDerivativeVectorValuedFunction(perturbation, adjoint_jacobian, adjoint_second_partial_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeControlState(const dotk::DOTk_State& state_,
                                                                               const dotk::DOTk_Control& control_,
                                                                               const dotk::DOTk_Dual& dual_,
                                                                               std::ostringstream& msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControlState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(control_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::ZU);

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(control_.data(), function_name, msg_);
    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(state_.data(), function_name,
                                                                        msg_);
    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }
    dotk::nlp::variables vars(*state_.data(), *control_.data(), *dual_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mState->clone();
    m_OriginalField->copy(*vars.mState);
    std::tr1::shared_ptr<dotk::Vector<Real> > delta_state = vars.mState->clone();
    dotk::gtools::generateRandomVector(delta_state);
    dotk::nlp::EqualityConstraintAdjointFirstDerivativeControl adjoint_jacobian;
    dotk::nlp::EqualityConstraintSecondDerivativeControlState adjoint_second_partial_derivative;
    this->checkAdjointSecondPartialDerivativeVectorValuedFunction(delta_state, adjoint_jacobian, adjoint_second_partial_derivative, vars);
    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeStateState(const dotk::DOTk_State & state_,
                                                                             const dotk::DOTk_Control & control_,
                                                                             const dotk::DOTk_Dual & dual_,
                                                                             std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeStateState");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(state_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::UU);
    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(state_.data(), function_name, msg_);
    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(state_.data(), function_name, msg_);
    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }

    dotk::nlp::variables vars(*state_.data(), *control_.data(), *dual_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mState->clone();
    m_OriginalField->copy(*vars.mState);

    std::tr1::shared_ptr<dotk::Vector<Real> > perturbation = vars.mState->clone();
    dotk::gtools::generateRandomVector(perturbation);

    dotk::nlp::EqualityConstraintAdjointFirstDerivativeState adjoint_jacobian;
    dotk::nlp::EqualityConstraintSecondDerivativeStateState adjoint_second_partial_derivative;
    this->checkAdjointSecondPartialDerivativeVectorValuedFunction(perturbation, adjoint_jacobian, adjoint_second_partial_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeStateControl(const dotk::DOTk_State& state_,
                                                                               const dotk::DOTk_Control& control_,
                                                                               const dotk::DOTk_Dual& dual_,
                                                                               std::ostringstream& msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeStateControl");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(state_.type());
    dotk::DOTk_DerivativeDiagnosticsTool::setDerivativeType(dotk::types::UZ);

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(state_.data(), function_name, msg_);
    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(control_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }
    dotk::nlp::variables vars(*state_.data(), *control_.data(), *dual_.data());
    m_OriginalField.reset();
    m_OriginalField = vars.mControl->clone();
    m_OriginalField->copy(*vars.mControl);
    std::tr1::shared_ptr<dotk::Vector<Real> > perturbation = vars.mControl->clone();
    dotk::gtools::generateRandomVector(perturbation);
    dotk::nlp::EqualityConstraintAdjointFirstDerivativeState adjoint_jacobian;
    dotk::nlp::EqualityConstraintSecondDerivativeStateControl adjoint_second_derivative;
    this->checkAdjointSecondPartialDerivativeVectorValuedFunction(perturbation, adjoint_jacobian, adjoint_second_derivative, vars);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

template<typename VectorValuedFunction, typename VectorValuedFunctionFirstDerivative>
void DOTk_DiagnosticsEqualityTypeNP::checkVectorValuedFunctionPartialDerivative
(const std::tr1::shared_ptr< dotk::Vector<Real> > & perturbation_vec_,
 const VectorValuedFunction & function_,
 const VectorValuedFunctionFirstDerivative & first_derivative_,
 dotk::nlp::variables & variables_)
{
    dotk::types::variable_t codomain = dotk::DOTk_DerivativeDiagnosticsTool::getCodomain();
    std::tr1::shared_ptr<dotk::Vector<Real> > residual = dotk::nlp::clone(variables_, codomain);
    std::tr1::shared_ptr<dotk::Vector<Real> > finite_diff_first_derivative = dotk::nlp::clone(variables_, codomain);
    std::tr1::shared_ptr<dotk::Vector<Real> > true_first_derivative = dotk::nlp::clone(variables_, codomain);

    first_derivative_(m_ContinuousOperators,
                      variables_.mState,
                      variables_.mControl,
                      perturbation_vec_,
                      true_first_derivative);
    Real norm_true_first_derivative = true_first_derivative->norm();

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
        function_(m_ContinuousOperators, variables_.mState, variables_.mControl, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(8.), *residual);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(-epsilon, *perturbation_vec_, variables_, derivative_type);
        function_(m_ContinuousOperators, variables_.mState, variables_.mControl, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(-8.), *residual);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(2.) * epsilon, *perturbation_vec_, variables_, derivative_type);
        function_(m_ContinuousOperators, variables_.mState, variables_.mControl, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(-1.), *residual);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(-2.) * epsilon, *perturbation_vec_, variables_, derivative_type);
        function_(m_ContinuousOperators, variables_.mState, variables_.mControl, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(1.), *residual);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        Real alpha = static_cast<Real>(1.) / (static_cast<Real>(12.) * epsilon);
        finite_diff_first_derivative->scale(alpha);
        Real finite_difference_approximation = finite_diff_first_derivative->norm();

        finite_diff_first_derivative->axpy(static_cast<Real>(-1.), *true_first_derivative);
        Real numerator = finite_diff_first_derivative->norm();
        Real denominator = std::numeric_limits<Real>::epsilon() + norm_true_first_derivative;
        Real relative_error = numerator / denominator;

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(norm_true_first_derivative,
                                                                              finite_difference_approximation,
                                                                              relative_error,
                                                                              epsilon);
    }
}

template<typename VectorValuedFunctionFirstDerivative, typename AdjointVectorValuedFunctionFirstDerivative>
Real DOTk_DiagnosticsEqualityTypeNP::checkAdjointPartialDerivativeVectorValuedFunction
(const std::tr1::shared_ptr< dotk::Vector<Real> > & perturbation_vec_,
 const VectorValuedFunctionFirstDerivative& first_derivative_,
 const AdjointVectorValuedFunctionFirstDerivative& adjoint_first_derivative_,
 dotk::nlp::variables & variables_)
{
    std::tr1::shared_ptr<dotk::Vector<Real> > first_derivative_times_direction = variables_.mDual->clone();
    first_derivative_(m_ContinuousOperators,
                      variables_.mState,
                      variables_.mControl,
                      perturbation_vec_,
                      first_derivative_times_direction);
    std::tr1::shared_ptr<dotk::Vector<Real> > adjoint_first_derivative_times_dual = perturbation_vec_->clone();
    adjoint_first_derivative_(m_ContinuousOperators,
                              variables_.mState,
                              variables_.mControl,
                              variables_.mDual,
                              adjoint_first_derivative_times_dual);

    Real dual_dot_first_derivative_times_direction = variables_.mDual->dot(*first_derivative_times_direction);
    Real perturbation_adjoint_first_derivative_times_dual = adjoint_first_derivative_times_dual->dot(*perturbation_vec_);

    Real absolute_difference = std::fabs(dual_dot_first_derivative_times_direction
            - perturbation_adjoint_first_derivative_times_dual);

    return (absolute_difference);
}

template<typename AdjointVectorValuedFunctionFirstDerivative, typename AdjointVectorValuedFunctionSecondDerivative>
void DOTk_DiagnosticsEqualityTypeNP::checkAdjointSecondPartialDerivativeVectorValuedFunction
(const std::tr1::shared_ptr< dotk::Vector<Real> > & perturbation_,
 const AdjointVectorValuedFunctionFirstDerivative& adjoint_first_derivative_,
 const AdjointVectorValuedFunctionSecondDerivative& adjoint_second_derivative_,
 dotk::nlp::variables & variables_)
{
    dotk::types::variable_t codomain = dotk::DOTk_DerivativeDiagnosticsTool::getCodomain();
    std::tr1::shared_ptr<dotk::Vector<Real> > true_adjoint_second_derivative_times_direction =
            dotk::nlp::clone(variables_, codomain);
    adjoint_second_derivative_(m_ContinuousOperators,
                               variables_.mState,
                               variables_.mControl,
                               variables_.mDual,
                               perturbation_,
                               true_adjoint_second_derivative_times_direction);

    Real true_norm_adjoint_second_derivative_times_direction = true_adjoint_second_derivative_times_direction->norm();

    std::tr1::shared_ptr<dotk::Vector<Real> > adjoint_first_derivative = dotk::nlp::clone(variables_, codomain);
    std::tr1::shared_ptr<dotk::Vector<Real> > finite_diff_adjoint_second_derivative_times_direction =
            dotk::nlp::clone(variables_, codomain);

    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();
    dotk::types::derivative_t derivative_type = dotk::DOTk_DerivativeDiagnosticsTool::getDerivativeType();
    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);

        // four point finite difference approximation
        dotk::nlp::perturbField(epsilon, *perturbation_, variables_, derivative_type);
        adjoint_first_derivative_(m_ContinuousOperators,
                                  variables_.mState,
                                  variables_.mControl,
                                  variables_.mDual,
                                  adjoint_first_derivative);
        finite_diff_adjoint_second_derivative_times_direction->axpy(static_cast<Real>(8.), *adjoint_first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(-epsilon, *perturbation_, variables_, derivative_type);
        adjoint_first_derivative_(m_ContinuousOperators,
                                  variables_.mState,
                                  variables_.mControl,
                                  variables_.mDual,
                                  adjoint_first_derivative);
        finite_diff_adjoint_second_derivative_times_direction->axpy(static_cast<Real>(-8.), *adjoint_first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(2.) * epsilon, *perturbation_, variables_, derivative_type);
        adjoint_first_derivative_(m_ContinuousOperators,
                                  variables_.mState,
                                  variables_.mControl,
                                  variables_.mDual,
                                  adjoint_first_derivative);
        finite_diff_adjoint_second_derivative_times_direction->axpy(static_cast<Real>(-1.), *adjoint_first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        dotk::nlp::perturbField(static_cast<Real>(-2.) * epsilon, *perturbation_, variables_, derivative_type);
        adjoint_first_derivative_(m_ContinuousOperators,
                                  variables_.mState,
                                  variables_.mControl,
                                  variables_.mDual,
                                  adjoint_first_derivative);
        finite_diff_adjoint_second_derivative_times_direction->axpy(static_cast<Real>(1.), *adjoint_first_derivative);
        dotk::nlp::resetField(*m_OriginalField, variables_, derivative_type);

        Real alpha = static_cast<Real>(1.) / (static_cast<Real>(12.) * epsilon);
        finite_diff_adjoint_second_derivative_times_direction->scale(alpha);
        Real finite_difference_approximation = finite_diff_adjoint_second_derivative_times_direction->norm();

        finite_diff_adjoint_second_derivative_times_direction->axpy(static_cast<Real>(-1.),
                                                                    *true_adjoint_second_derivative_times_direction);
        Real numerator = finite_diff_adjoint_second_derivative_times_direction->norm();
        Real denominator = std::numeric_limits<Real>::epsilon() + true_norm_adjoint_second_derivative_times_direction;
        Real relative_error = numerator / denominator;

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(true_norm_adjoint_second_derivative_times_direction,
                                                                              finite_difference_approximation,
                                                                              relative_error,
                                                                              epsilon);
    }
}

}
