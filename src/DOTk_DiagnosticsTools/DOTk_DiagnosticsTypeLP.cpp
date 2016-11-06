/*
 * DOTk_DiagnosticsTypeLP.cpp
 *
 *  Created on: Mar 29, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

#include "vector.hpp"
#include "DOTk_Dual.hpp"
#include "DOTk_DiagnosticsTypeLP.hpp"
#include "DOTk_DescentDirectionTools.hpp"

namespace dotk
{

DOTk_DiagnosticsTypeLP::DOTk_DiagnosticsTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_DerivativeDiagnosticsTool(),
        m_TrueDerivative(),
        m_OriginalPrimal(),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(),
        m_InequalityConstraint()
{
}

DOTk_DiagnosticsTypeLP::DOTk_DiagnosticsTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                               const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_DerivativeDiagnosticsTool(),
        m_TrueDerivative(),
        m_OriginalPrimal(),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(equality_),
        m_InequalityConstraint()
{
}

DOTk_DiagnosticsTypeLP::DOTk_DiagnosticsTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                               const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                                               const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_) :
        dotk::DOTk_DerivativeDiagnosticsTool(),
        m_TrueDerivative(),
        m_OriginalPrimal(),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(equality_),
        m_InequalityConstraint(inequality_.begin(), inequality_.end())
{
}

DOTk_DiagnosticsTypeLP::~DOTk_DiagnosticsTypeLP()
{
}

void DOTk_DiagnosticsTypeLP::checkObjectiveGradient(const dotk::DOTk_Variable & primal_, std::ostringstream & msg_)
{
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(primal_.type());
    std::string function_name("dotk::DOTk_DiagnosticsTypeLP.checkObjectiveGradient");

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(primal_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    m_OriginalPrimal.reset();
    m_OriginalPrimal = primal_.data()->clone();
    m_OriginalPrimal->copy(*primal_.data());

    std::tr1::shared_ptr<dotk::vector<Real> > delta_primal = primal_.data()->clone();
    dotk::gtools::generateRandomVector(delta_primal);

    m_TrueDerivative.reset();
    m_TrueDerivative = primal_.data()->clone();
    m_TrueDerivative->fill(0.);
    dotk::lp::ObjectiveFunctionFirstDerivative first_derivative(primal_.type());
    first_derivative(m_ObjectiveFunction, m_OriginalPrimal, m_TrueDerivative);

    dotk::lp::ObjectiveFunctionEvaluate objective;
    this->checkScalarValuedFunctionFirstDerivative(primal_.data(), delta_primal, objective, m_ObjectiveFunction);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsTypeLP::checkObjectiveHessian(const dotk::DOTk_Variable & primal_, std::ostringstream & msg_)
{
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(primal_.type());
    std::string function_name("dotk::DOTk_DiagnosticsTypeLP.checkObjectiveHessian");

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(primal_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    m_OriginalPrimal.reset();
    m_OriginalPrimal = primal_.data()->clone();
    m_OriginalPrimal->copy(*primal_.data());

    std::tr1::shared_ptr<dotk::vector<Real> > delta_primal = primal_.data()->clone();
    dotk::gtools::generateRandomVector(delta_primal);

    m_TrueDerivative.reset();
    m_TrueDerivative = primal_.data()->clone();
    m_TrueDerivative->fill(0.);
    dotk::lp::ObjectiveFunctionSecondDerivative second_derivative(primal_.type());
    second_derivative(m_ObjectiveFunction, m_OriginalPrimal, delta_primal, m_TrueDerivative);

    dotk::lp::ObjectiveFunctionFirstDerivative first_derivative(primal_.type());
    this->checkScalarValuedFunctionSecondDerivative(primal_.data(),
                                                    delta_primal,
                                                    first_derivative,
                                                    m_ObjectiveFunction);

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsTypeLP::checkEqualityConstraintJacobian(const dotk::DOTk_Variable & primal_,
                                                             std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsTypeLP.checkEqualityConstraintJacobian");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(primal_.type());

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(primal_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    m_OriginalPrimal.reset();
    m_OriginalPrimal = primal_.data()->clone();
    m_OriginalPrimal->copy(*primal_.data());

    std::tr1::shared_ptr<dotk::vector<Real> > delta_primal = m_OriginalPrimal->clone();
    dotk::gtools::generateRandomVector(delta_primal);

    dotk::lp::EqualityConstraintResidual equality_constraint;
    dotk::lp::EqualityConstraintFirstDerivative first_derivative;
    this->checkVectorValuedFunctionFirstDerivative(delta_primal, equality_constraint, first_derivative, primal_.data());

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

Real DOTk_DiagnosticsTypeLP::checkEqualityConstraintAdjointJacobian(const dotk::DOTk_Variable & primal_,
                                                                    const dotk::DOTk_Dual & dual_,
                                                                    std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsTypeLP.checkEqualityConstraintAdjointJacobian");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(dual_.type());

    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(primal_.data(), function_name, msg_);
    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(dual_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return (-1.);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > delta_primal = primal_.data()->clone();
    dotk::gtools::generateRandomVector(delta_primal);

    dotk::lp::EqualityConstraintFirstDerivative first_derivative;
    dotk::lp::EqualityConstraintAdjointFirstDerivative adjoint_first_derivative;
    Real absolute_difference = this->checkAdjointFirstDerivativeVectorValuedFunction(delta_primal,
                                                                                     first_derivative,
                                                                                     adjoint_first_derivative,
                                                                                     primal_.data(),
                                                                                     dual_.data());

    msg_ << "The absolute difference between (dual_dot_first_derivative_times_direction) and "
            << "(adjoint_first_derivative_times_dual_dot_direction) = " << std::scientific << std::setprecision(6)
            << absolute_difference << "\n";

    return (absolute_difference);
}

void DOTk_DiagnosticsTypeLP::checkEqualityConstraintJacobianDerivative(const dotk::DOTk_Variable & primal_,
                                                                       const dotk::DOTk_Dual & dual_,
                                                                       std::ostringstream & msg_)
{
    std::string function_name("dotk::DOTk_DiagnosticsTypeLP.checkEqualityConstraintJacobianDerivative");
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(primal_.type());

    bool invalid_domain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkDomainDimensions(primal_.data(), function_name, msg_);
    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(primal_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true || invalid_domain_dimensions == true)
    {
        return;
    }

    m_OriginalPrimal.reset();
    m_OriginalPrimal = primal_.data()->clone();
    m_OriginalPrimal->copy(*primal_.data());

    std::tr1::shared_ptr<dotk::vector<Real> > delta_primal = primal_.data()->clone();
    dotk::gtools::generateRandomVector(delta_primal);

    dotk::lp::EqualityConstraintAdjointFirstDerivative adjoint_first_derivative;
    dotk::lp::EqualityConstraintSecondDerivative adjoint_second_derivative;
    this->checkSecondDerivativeVectorValuedFunction(delta_primal,
                                                    adjoint_first_derivative,
                                                    adjoint_second_derivative,
                                                    primal_.data(),
                                                    dual_.data());

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

void DOTk_DiagnosticsTypeLP::checkInequalityConstraintJacobian(const dotk::DOTk_Variable & primal_,
                                                               std::ostringstream & msg_)
{
    dotk::DOTk_DerivativeDiagnosticsTool::setCodomain(primal_.type());
    std::string function_name("dotk::DOTk_DiagnosticsTypeLP.checkInequalityConstraintJacobian");

    bool invalid_codomain_dimensions =
            dotk::DOTk_DerivativeDiagnosticsTool::checkCodomainDimensions(primal_.data(), function_name, msg_);

    if(invalid_codomain_dimensions == true)
    {
        return;
    }

    m_OriginalPrimal.reset();
    m_OriginalPrimal = primal_.data()->clone();
    m_OriginalPrimal->copy(*primal_.data());

    std::tr1::shared_ptr<dotk::vector<Real> > perturbation = primal_.data()->clone();
    dotk::gtools::generateRandomVector(perturbation);

    m_TrueDerivative.reset();
    m_TrueDerivative = primal_.data()->clone();
    m_TrueDerivative->fill(0.);
    dotk::lp::InequalityConstraintEvaluate value;
    dotk::lp::InequalityConstraintFirstDerivative gradient;

    std::ostringstream message;
    const size_t num_inequality_constraints = m_InequalityConstraint.size();
    for(size_t index = 0; index < num_inequality_constraints; ++ index)
    {
        message << "\n CHECK GRADIENT OF INEQUALITY CONSTRAINT WITH ID = " << index << "\n";
        dotk::DOTk_DerivativeDiagnosticsTool::addMessage(message);

        gradient(m_InequalityConstraint[index], m_OriginalPrimal, m_TrueDerivative);
        this->checkScalarValuedFunctionFirstDerivative(primal_.data(), perturbation, value, m_InequalityConstraint[index]);

        message.str("");
    }

    dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsMsg(msg_);
}

template<typename Functor, typename DerivativeOperators>
void DOTk_DiagnosticsTypeLP::checkScalarValuedFunctionFirstDerivative(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                                      const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                                      const Functor & functor_,
                                                                      const DerivativeOperators & operators_)
{
    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();

    Real gradient_dot_dprimal = m_TrueDerivative->dot(*delta_primal_);

    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);
        // four point finite difference approximation
        primal_->axpy(epsilon, *delta_primal_);
        Real objective_function_value_one = functor_(operators_, primal_);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(-epsilon, *delta_primal_);
        Real objective_function_value_two = functor_(operators_, primal_);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(2.) * epsilon, *delta_primal_);
        Real objective_function_value_three = functor_(operators_, primal_);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(-2.) * epsilon, *delta_primal_);
        Real objective_function_value_four = functor_(operators_, primal_);
        primal_->copy(*m_OriginalPrimal);

        Real numerator = -objective_function_value_three + static_cast<Real>(8.) * objective_function_value_one
                - static_cast<Real>(8.) * objective_function_value_two + objective_function_value_four;
        Real denominator = static_cast<Real>(12.) * epsilon;
        Real finite_difference_appx = numerator / denominator;
        Real relative_error = std::abs(gradient_dot_dprimal - finite_difference_appx);

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(gradient_dot_dprimal,
                                                                              finite_difference_appx,
                                                                              relative_error,
                                                                              epsilon);
    }
}

template<typename Functor, typename DerivativeOperators>
void DOTk_DiagnosticsTypeLP::checkScalarValuedFunctionSecondDerivative(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                                       const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                                       const Functor & first_derivative_,
                                                                       const DerivativeOperators & second_derivative_)
{
    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();

    Real norm_true_derivative_times_dprimal = m_TrueDerivative->norm();
    std::tr1::shared_ptr<dotk::vector<Real> > gradient = primal_->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > finite_difference_derivative_times_dprimal = primal_->clone();

    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        finite_difference_derivative_times_dprimal->fill(0.);
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);

        // four point finite difference approximation
        primal_->axpy(epsilon, *delta_primal_);
        first_derivative_(m_ObjectiveFunction, primal_, gradient);
        finite_difference_derivative_times_dprimal->axpy(static_cast<Real>(8.), *gradient);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(-epsilon, *delta_primal_);
        first_derivative_(m_ObjectiveFunction, primal_, gradient);
        finite_difference_derivative_times_dprimal->axpy(static_cast<Real>(-8.), *gradient);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(2.) * epsilon, *delta_primal_);
        first_derivative_(m_ObjectiveFunction, primal_, gradient);
        finite_difference_derivative_times_dprimal->axpy(static_cast<Real>(-1.), *gradient);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(-2.) * epsilon, *delta_primal_);
        first_derivative_(m_ObjectiveFunction, primal_, gradient);
        finite_difference_derivative_times_dprimal->axpy(static_cast<Real>(1.), *gradient);
        primal_->copy(*m_OriginalPrimal);
        Real alpha = static_cast<Real>(1.) / (static_cast<Real>(12.) * epsilon);
        finite_difference_derivative_times_dprimal->scale(alpha);

        Real norm_finite_difference_derivative = finite_difference_derivative_times_dprimal->norm();

        finite_difference_derivative_times_dprimal->axpy(static_cast<Real>(-1.), *m_TrueDerivative);
        Real numerator = finite_difference_derivative_times_dprimal->norm();
        Real denominator = std::numeric_limits<Real>::epsilon() + norm_true_derivative_times_dprimal;
        Real relative_error = numerator / denominator;

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(norm_true_derivative_times_dprimal,
                                                                              norm_finite_difference_derivative,
                                                                              relative_error,
                                                                              epsilon);
    }

}
void DOTk_DiagnosticsTypeLP::checkVectorValuedFunctionFirstDerivative(const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                                      const dotk::lp::EqualityConstraintResidual & function_,
                                                                      const dotk::lp::EqualityConstraintFirstDerivative & first_derivative_,
                                                                      const std::tr1::shared_ptr<dotk::vector<Real> > & primal_)
{
    std::tr1::shared_ptr<dotk::vector<Real> > residual = primal_->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > finite_diff_first_derivative = primal_->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > true_first_derivative = primal_->clone();

    first_derivative_(m_EqualityConstraint, primal_, delta_primal_, true_first_derivative);
    Real norm_true_first_derivative = true_first_derivative->norm();

    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();

    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);

        // four point finite difference approximation
        primal_->axpy(epsilon, *delta_primal_);
        function_(m_EqualityConstraint, primal_, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(8.), *residual);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(-epsilon, *delta_primal_);
        function_(m_EqualityConstraint, primal_, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(-8.), *residual);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(2.) * epsilon, *delta_primal_);
        function_(m_EqualityConstraint, primal_, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(-1.), *residual);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(-2.) * epsilon, *delta_primal_);
        function_(m_EqualityConstraint, primal_, residual);
        finite_diff_first_derivative->axpy(static_cast<Real>(1.), *residual);
        primal_->copy(*m_OriginalPrimal);

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

Real DOTk_DiagnosticsTypeLP::checkAdjointFirstDerivativeVectorValuedFunction(const std::tr1::shared_ptr<
                                                                                     dotk::vector<Real> > & delta_primal_,
                                                                             const dotk::lp::EqualityConstraintFirstDerivative & first_derivative_,
                                                                             const dotk::lp::EqualityConstraintAdjointFirstDerivative & adjoint_first_derivative_,
                                                                             const std::tr1::shared_ptr<
                                                                                     dotk::vector<Real> > & primal_,
                                                                             const std::tr1::shared_ptr<
                                                                                     dotk::vector<Real> > & dual_)
{
    std::tr1::shared_ptr<dotk::vector<Real> > first_derivative_times_dprimal = dual_->clone();
    first_derivative_(m_EqualityConstraint, primal_, delta_primal_, first_derivative_times_dprimal);
    std::tr1::shared_ptr<dotk::vector<Real> > adjoint_first_derivative_times_dual = delta_primal_->clone();
    adjoint_first_derivative_(m_EqualityConstraint, primal_, dual_, adjoint_first_derivative_times_dual);

    Real dual_dot_first_derivative_times_dprimal = dual_->dot(*first_derivative_times_dprimal);
    Real dprimal_adjoint_first_derivative_times_dual = adjoint_first_derivative_times_dual->dot(*delta_primal_);

    Real absolute_difference = std::fabs(dual_dot_first_derivative_times_dprimal
            - dprimal_adjoint_first_derivative_times_dual);

    return (absolute_difference);
}

void DOTk_DiagnosticsTypeLP::checkSecondDerivativeVectorValuedFunction(const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                                                       const dotk::lp::EqualityConstraintAdjointFirstDerivative & adjoint_first_derivative_,
                                                                       const dotk::lp::EqualityConstraintSecondDerivative & adjoint_second_derivative_,
                                                                       const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                                       const std::tr1::shared_ptr<dotk::vector<Real> > & dual_)
{
    std::tr1::shared_ptr<dotk::vector<Real> > true_second_derivative_times_dprimal = primal_->clone();
    adjoint_second_derivative_(m_EqualityConstraint,
                               primal_,
                               dual_,
                               delta_primal_,
                               true_second_derivative_times_dprimal);

    Real true_norm_second_derivative_times_dprimal = true_second_derivative_times_dprimal->norm();

    std::tr1::shared_ptr<dotk::vector<Real> > adjoint_first_derivative = primal_->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > finite_diff_second_derivative_times_dprimal = primal_->clone();

    Int superscript_lower_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsLowerLimit();
    Int superscript_upper_limit =
            dotk::DOTk_DerivativeDiagnosticsTool::getFiniteDifferenceDiagnosticsSuperScriptsUpperLimit();

    for(Int superscript = superscript_lower_limit; superscript <= superscript_upper_limit; ++ superscript)
    {
        Real epsilon = std::pow(static_cast<Real>(0.1), superscript);

        // four point finite difference approximation
        primal_->axpy(epsilon, *delta_primal_);
        adjoint_first_derivative_(m_EqualityConstraint, primal_, dual_, adjoint_first_derivative);
        finite_diff_second_derivative_times_dprimal->axpy(static_cast<Real>(8.), *adjoint_first_derivative);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(-epsilon, *delta_primal_);
        adjoint_first_derivative_(m_EqualityConstraint, primal_, dual_, adjoint_first_derivative);
        finite_diff_second_derivative_times_dprimal->axpy(static_cast<Real>(-8.), *adjoint_first_derivative);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(2.) * epsilon, *delta_primal_);
        adjoint_first_derivative_(m_EqualityConstraint, primal_, dual_, adjoint_first_derivative);
        finite_diff_second_derivative_times_dprimal->axpy(static_cast<Real>(-1.), *adjoint_first_derivative);
        primal_->copy(*m_OriginalPrimal);

        primal_->axpy(static_cast<Real>(-2.) * epsilon, *delta_primal_);
        adjoint_first_derivative_(m_EqualityConstraint, primal_, dual_, adjoint_first_derivative);
        finite_diff_second_derivative_times_dprimal->axpy(static_cast<Real>(1.), *adjoint_first_derivative);
        primal_->copy(*m_OriginalPrimal);

        Real alpha = static_cast<Real>(1.) / (static_cast<Real>(12.) * epsilon);
        finite_diff_second_derivative_times_dprimal->scale(alpha);
        Real finite_difference_approximation = finite_diff_second_derivative_times_dprimal->norm();

        finite_diff_second_derivative_times_dprimal->axpy(static_cast<Real>(-1.),
                                                          *true_second_derivative_times_dprimal);
        Real numerator = finite_diff_second_derivative_times_dprimal->norm();
        Real denominator = std::numeric_limits<Real>::epsilon() + true_norm_second_derivative_times_dprimal;
        Real relative_error = numerator / denominator;

        dotk::DOTk_DerivativeDiagnosticsTool::saveFiniteDifferenceDiagnostics(true_norm_second_derivative_times_dprimal,
                                                                              finite_difference_approximation,
                                                                              relative_error,
                                                                              epsilon);
    }
}

}
