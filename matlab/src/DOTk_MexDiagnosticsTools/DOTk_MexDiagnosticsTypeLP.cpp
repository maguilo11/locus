/*
 * DOTk_MexDiagnosticsTypeLP.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <tr1/memory>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexDiagnosticsTypeLP.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInequalityConstraint.hpp"

#include "DOTk_Dual.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_DiagnosticsTypeLP.hpp"

namespace dotk
{

DOTk_MexDiagnosticsTypeLP::DOTk_MexDiagnosticsTypeLP(const mxArray* input_[]) :
        dotk::DOTk_MexDiagnostics(input_[0])
{
}

DOTk_MexDiagnosticsTypeLP::~DOTk_MexDiagnosticsTypeLP()
{
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    switch (type)
    {
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_LP_BOUND:
        {
            this->checkFirstDerivativeTypeULP(input_);
            break;
        }
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_ELP_BOUND:
        {
            this->checkFirstDerivativeTypeELP(input_);
            break;
        }
        case dotk::types::TYPE_CLP:
        {
            this->checkFirstDerivativeTypeCLP(input_);
            break;
        }
        case dotk::types::TYPE_ILP:
        {
            this->checkFirstDerivativeTypeILP(input_);
            break;
        }
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_CNLP:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string msg(" DOTk/MEX ERROR: Invalid Linear Programming Problem Type. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    switch (type)
    {
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_LP_BOUND:
        {
            this->checkSecondDerivativeTypeULP(input_);
            break;
        }
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_ELP_BOUND:
        {
            this->checkSecondDerivativeTypeELP(input_);
            break;
        }
        case dotk::types::TYPE_CLP:
        {
            this->checkSecondDerivativeTypeCLP(input_);
            break;
        }
        case dotk::types::TYPE_ILP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_CNLP:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string msg(" DOTk/MEX ERROR: Invalid Linear Programming Problem Type. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeULP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeILP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    mxArray* mx_inequality = dotk::mex::parseInequalityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexInequalityConstraint>
        inequality(new dotk::DOTk_MexInequalityConstraint(mx_inequality, problem_type));
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality_vector(1, inequality);
    mxDestroyArray(mx_inequality);

    // Set dummy inequality constraint
    std::shared_ptr<dotk::DOTk_EqualityConstraint<double> > equality(new dotk::DOTk_EqualityConstraint<double>);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality, inequality_vector);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Inequality Constraint Jacobian **** \n");
    diagnostics.checkInequalityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeELP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    mxArray* mx_equality = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
    equality(new dotk::DOTk_MexEqualityConstraint(mx_equality, problem_type));
    mxDestroyArray(mx_equality);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian **** \n");
    diagnostics.checkEqualityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());

    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector mx_dual(num_duals, 0.);
    dotk::DOTk_Dual dual(mx_dual);

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Adjoint of Jacobain **** \n");
    diagnostics.checkEqualityConstraintAdjointJacobian(control, dual, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeCLP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    mxArray* mx_equality = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
    equality(new dotk::DOTk_MexEqualityConstraint(mx_equality, problem_type));
    mxDestroyArray(mx_equality);

    mxArray* mx_inequality = dotk::mex::parseInequalityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexInequalityConstraint>
        inequality(new dotk::DOTk_MexInequalityConstraint(mx_inequality, problem_type));
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality_vector(1, inequality);
    mxDestroyArray(mx_inequality);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality, inequality_vector);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian **** \n");
    diagnostics.checkEqualityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());

    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector mx_dual(num_duals, 0.);
    dotk::DOTk_Dual dual(mx_dual);

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Adjoint of Jacobian **** \n");
    diagnostics.checkEqualityConstraintAdjointJacobian(control, dual, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Inequality Constraint Jacobian **** \n");
    diagnostics.checkInequalityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivativeTypeULP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Hessian **** \n");
    diagnostics.checkObjectiveHessian(control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivativeTypeELP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    mxArray* mx_equality = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
    equality(new dotk::DOTk_MexEqualityConstraint(mx_equality, problem_type));
    mxDestroyArray(mx_equality);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Hessian **** \n");
    diagnostics.checkObjectiveHessian(control, msg);
    mexPrintf(msg.str().c_str());

    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector mx_dual(num_duals, 0.);
    dotk::DOTk_Dual dual(mx_dual);

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian Derivative **** \n");
    diagnostics.checkEqualityConstraintJacobianDerivative(control, dual, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivativeTypeCLP(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    mxArray* mx_equality = dotk::mex::parseObjectiveFunction(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
    equality(new dotk::DOTk_MexEqualityConstraint(mx_equality, problem_type));
    mxDestroyArray(mx_equality);

    mxArray* mx_inequality = dotk::mex::parseInequalityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexInequalityConstraint >
        inequality(new dotk::DOTk_MexInequalityConstraint(mx_inequality, problem_type));
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality_vector(1, inequality);
    mxDestroyArray(mx_inequality);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality, inequality_vector);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Hessian **** \n");
    diagnostics.checkObjectiveHessian(control, msg);
    mexPrintf(msg.str().c_str());

    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector mx_dual(num_duals, 0.);
    dotk::DOTk_Dual dual(mx_dual);

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian Derivative **** \n");
    diagnostics.checkEqualityConstraintJacobianDerivative(control, dual, msg);
    mexPrintf(msg.str().c_str());
}

}
