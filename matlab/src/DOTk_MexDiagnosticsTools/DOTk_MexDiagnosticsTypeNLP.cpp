/*
 * DOTk_MexDiagnosticsTypeNLP.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <tr1/memory>

#include "vector.hpp"
#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_DiagnosticsTypeLP.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexDiagnosticsTypeLP.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexDiagnosticsTypeNLP.hpp"
#include "DOTk_MexInequalityConstraint.cpp"
#include "DOTk_MexInequalityConstraint.hpp"
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
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    switch(type)
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
            std::string msg(" DOTk/MEX ERROR: Invalid Nonlinear Programming Problem Type. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexDiagnosticsTypeNLP::checkSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    switch(type)
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
            std::string msg(" DOTk/MEX ERROR: Invalid Nonlinear Programming Problem Type. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
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
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));

    dotk::DOTk_DiagnosticsObjectiveTypeNP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_State state;
    dotk::mex::buildStateContainer(input_[0], state);
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function First Derivative W.R.T. State **** \n");
    diagnostics.checkPartialDerivativeState(state, control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Objective Function First Derivative W.R.T. Control **** \n");
    diagnostics.checkPartialDerivativeControl(state, control, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
}

void DOTk_MexDiagnosticsTypeNLP::checkObjectiveFunctionSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));

    dotk::DOTk_DiagnosticsObjectiveTypeNP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_State state;
    dotk::mex::buildStateContainer(input_[0], state);
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

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

    objective_ptr.release();
}

void DOTk_MexDiagnosticsTypeNLP::checkEqualityConstraintFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    dotk::DOTk_MexArrayPtr equality_ptr;
    dotk::mex::parseEqualityConstraint(input_[1], equality_ptr);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(equality_ptr.get(), type));

    dotk::DOTk_Dual dual;
    dotk::mex::buildDualContainer(input_[0], dual);
    dotk::DOTk_State state;
    dotk::mex::buildStateContainer(input_[0], state);
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

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

    equality_ptr.release();
}

void DOTk_MexDiagnosticsTypeNLP::checkEqualityConstraintSecondDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    dotk::DOTk_MexArrayPtr equality_ptr;
    dotk::mex::parseEqualityConstraint(input_[1], equality_ptr);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(equality_ptr.get(), type));

    dotk::DOTk_Dual dual;
    dotk::mex::buildDualContainer(input_[0], dual);
    dotk::DOTk_State state;
    dotk::mex::buildStateContainer(input_[0], state);
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

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

    equality_ptr.release();
}

void DOTk_MexDiagnosticsTypeNLP::checkInequalityConstraintFirstDerivative(const mxArray* input_[])
{
    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();

    dotk::DOTk_MexArrayPtr matlab_ptr;
    dotk::mex::parseInequalityConstraint(input_[1], matlab_ptr);
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        shared_ptr(new dotk::DOTk_MexInequalityConstraint<double>(matlab_ptr.get(), type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequalities(1, shared_ptr);

    dotk::DOTk_State state;
    dotk::mex::buildStateContainer(input_[0], state);

    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    dotk::DOTk_DiagnosticsInequalityTypeNP diagnostics(inequalities);
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

    matlab_ptr.release();
}

void DOTk_MexDiagnosticsTypeNLP::checkInequalityConstraintSecondDerivative(const mxArray* input_[])
{
}

}
