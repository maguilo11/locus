/*
 * DOTk_MexDiagnosticsTypeNLP.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <tr1/memory>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexDiagnosticsTypeLP.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexDiagnosticsTypeNLP.hpp"
#include "DOTk_MexInequalityConstraint.hpp"

#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_DiagnosticsTypeLP.hpp"
#include "DOTk_DiagnosticsEqualityTypeNP.hpp"
#include "DOTk_DiagnosticsObjectiveTypeNP.hpp"
#include "DOTk_DiagnosticsInequalityTypeNP.hpp"

namespace dotk
{

DOTk_MexDiagnosticsTypeNLP::DOTk_MexDiagnosticsTypeNLP(const mxArray* input_[]) :
        dotk::DOTk_MexDiagnostics(input_[0])
{
}

DOTk_MexDiagnosticsTypeNLP::~DOTk_MexDiagnosticsTypeNLP()
{
}

void DOTk_MexDiagnosticsTypeNLP::checkFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    switch(problem_type)
    {
        case dotk::types::TYPE_UNLP:
        {
            this->checkFirstDerivativeTypeUNLP(input_);
            break;
        }
        case dotk::types::TYPE_ENLP:
        {
            this->checkFirstDerivativeTypeENLP(input_);
            break;
        }
        case dotk::types::TYPE_ENLP_BOUND:
        {
            this->checkFirstDerivativeTypeENLP(input_);
            break;
        }
        case dotk::types::TYPE_CNLP:
        {
            this->checkFirstDerivativeTypeCNLP(input_);
            break;
        }
        case dotk::types::TYPE_NLP_BOUND:
        {
            this->checkFirstDerivativeTypeUNLP(input_);
            break;
        }
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_ILP:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> UNDEFINED PROBLEM TYPE. SEE USERS' MANUAL FOR VALID OPTIONS.\n";
            mexWarnMsgTxt(msg.str().c_str());
            break;
        }
    }
}

void DOTk_MexDiagnosticsTypeNLP::checkSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();
    switch(problem_type)
    {
        case dotk::types::TYPE_UNLP:
        {
            this->checkSecondDerivativeTypeUNLP(input_);
            break;
        }
        case dotk::types::TYPE_ENLP:
        {
            this->checkSecondDerivativeTypeENLP(input_);
            break;
        }
        case dotk::types::TYPE_ENLP_BOUND:
        {
            this->checkSecondDerivativeTypeENLP(input_);
            break;
        }
        case dotk::types::TYPE_CNLP:
        {
            this->checkSecondDerivativeTypeCNLP(input_);
            break;
        }
        case dotk::types::TYPE_NLP_BOUND:
        {
            this->checkSecondDerivativeTypeUNLP(input_);
            break;
        }
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_ILP:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> UNDEFINED PROBLEM TYPE. SEE USERS' MANUAL FOR VALID OPTIONS.\n";
            mexWarnMsgTxt(msg.str().c_str());
            break;
        }
    }
}

void DOTk_MexDiagnosticsTypeNLP::checkFirstDerivativeTypeUNLP(const mxArray* input_[])
{
    this->checkObjectiveFunctionFirstDerivative(input_);
    this->checkEqualityConstraintFirstDerivative(input_);
}

void DOTk_MexDiagnosticsTypeNLP::checkFirstDerivativeTypeENLP(const mxArray* input_[])
{
    this->checkObjectiveFunctionFirstDerivative(input_);
    this->checkEqualityConstraintFirstDerivative(input_);
}

void DOTk_MexDiagnosticsTypeNLP::checkFirstDerivativeTypeCNLP(const mxArray* input_[])
{
    this->checkObjectiveFunctionFirstDerivative(input_);
    this->checkEqualityConstraintFirstDerivative(input_);
    this->checkInequalityConstraintFirstDerivative(input_);
}

void DOTk_MexDiagnosticsTypeNLP::checkSecondDerivativeTypeUNLP(const mxArray* input_[])
{
    this->checkObjectiveFunctionSecondDerivative(input_);
    this->checkEqualityConstraintSecondDerivative(input_);
}

void DOTk_MexDiagnosticsTypeNLP::checkSecondDerivativeTypeENLP(const mxArray* input_[])
{
    this->checkObjectiveFunctionSecondDerivative(input_);
    this->checkEqualityConstraintSecondDerivative(input_);
}

void DOTk_MexDiagnosticsTypeNLP::checkSecondDerivativeTypeCNLP(const mxArray* input_[])
{
    this->checkObjectiveFunctionSecondDerivative(input_);
    this->checkEqualityConstraintSecondDerivative(input_);
    this->checkInequalityConstraintSecondDerivative(input_);
}

void DOTk_MexDiagnosticsTypeNLP::checkObjectiveFunctionFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    dotk::DOTk_DiagnosticsObjectiveTypeNP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector mx_states(num_states, 0.);
    dotk::DOTk_State state(mx_states);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function First Derivative W.R.T. State **** \n");
    diagnostics.checkPartialDerivativeState(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Objective Function First Derivative W.R.T. Control **** \n");
    diagnostics.checkPartialDerivativeControl(state, control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeNLP::checkObjectiveFunctionSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_objective = dotk::mex::parseObjectiveFunction(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(mx_objective, problem_type));
    mxDestroyArray(mx_objective);

    dotk::DOTk_DiagnosticsObjectiveTypeNP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector mx_states(num_states, 0.);
    dotk::DOTk_State state(mx_states);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    std::ostringstream msg;
    mexPrintf("\n **** Check Objective Function Second Derivative W.R.T. State-State **** \n");
    diagnostics.checkPartialDerivativeStateState(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Objective Function Second Derivative W.R.T. State-Control **** \n");
    diagnostics.checkPartialDerivativeStateControl(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Objective Function Second Derivative W.R.T. Control-State **** \n");
    diagnostics.checkPartialDerivativeControlState(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Objective Function Second Derivative W.R.T. Control-Control **** \n");
    diagnostics.checkPartialDerivativeControlControl(state, control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeNLP::checkEqualityConstraintFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_equality = dotk::mex::parseObjectiveFunction(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
    equality(new dotk::DOTk_MexEqualityConstraint(mx_equality, problem_type));
    mxDestroyArray(mx_equality);

    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector mx_dual(num_duals, 0.);
    dotk::DOTk_Dual dual(mx_dual);

    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector mx_states(num_states, 0.);
    dotk::DOTk_State state(mx_states);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    dotk::DOTk_DiagnosticsEqualityTypeNP diagnostics(equality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    mexPrintf("\n **** Check Equality Constraint First Derivative W.R.T. State **** \n");
    diagnostics.checkPartialDerivativeState(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint First Derivative W.R.T. Control **** \n");
    diagnostics.checkPartialDerivativeControl(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Adjoint First Derivative W.R.T. State **** \n");
    diagnostics.checkAdjointPartialDerivativeState(state, control, dual, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Adjoint First Derivative W.R.T. Control **** \n");
    diagnostics.checkAdjointPartialDerivativeControl(state, control, dual, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeNLP::checkEqualityConstraintSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_equality = dotk::mex::parseObjectiveFunction(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
    equality(new dotk::DOTk_MexEqualityConstraint(mx_equality, problem_type));
    mxDestroyArray(mx_equality);

    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector mx_dual(num_duals, 0.);
    dotk::DOTk_Dual dual(mx_dual);

    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector mx_states(num_states, 0.);
    dotk::DOTk_State state(mx_states);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    dotk::DOTk_DiagnosticsEqualityTypeNP diagnostics(equality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    mexPrintf("\n **** Check Equality Constraint Second Derivative W.R.T. State-State **** \n");
    diagnostics.checkAdjointPartialDerivativeStateState(state, control, dual, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Second Derivative W.R.T. State-Control **** \n");
    diagnostics.checkAdjointPartialDerivativeStateControl(state, control, dual, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Second Derivative W.R.T. Control-State **** \n");
    diagnostics.checkAdjointPartialDerivativeControlState(state, control, dual, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Second Derivative W.R.T. Control-Control **** \n");
    diagnostics.checkAdjointPartialDerivativeControlControl(state, control, dual, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeNLP::checkInequalityConstraintFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t problem_type = DOTk_MexDiagnostics::getProblemType();

    mxArray* mx_inequality = dotk::mex::parseInequalityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint >
        inequality(new dotk::DOTk_MexInequalityConstraint(mx_inequality, problem_type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality_vector(1, inequality);
    mxDestroyArray(mx_inequality);

    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector mx_states(num_states, 0.);
    dotk::DOTk_State state(mx_states);

    size_t num_controls = dotk::mex::parseNumberControls(input_[0]);
    dotk::MexVector mx_control(num_controls, 0.);
    dotk::DOTk_Control control(mx_control);

    dotk::DOTk_DiagnosticsInequalityTypeNP diagnostics(inequality_vector);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    mexPrintf("\n **** Check Inequality Constraint First Derivative W.R.T. State **** \n");
    diagnostics.checkPartialDerivativeState(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Inequality Constraint First Derivative W.R.T. Control **** \n");
    diagnostics.checkPartialDerivativeControl(state, control, msg);
    mexPrintf(msg.str().c_str());
}

void DOTk_MexDiagnosticsTypeNLP::checkInequalityConstraintSecondDerivative(const mxArray* input_[])
{
}

}
