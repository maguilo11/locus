/*
 * DOTk_MexDiagnosticsTypeLP.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "vector.hpp"
#include "DOTk_Dual.hpp"
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
#include "DOTk_MexInequalityConstraint.cpp"
#include "DOTk_MexInequalityConstraint.hpp"

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
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeILP(const mxArray* input_[])
{
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    dotk::DOTk_MexArrayPtr inequality_ptr;
    dotk::mex::parseInequalityConstraint(input_[1], inequality_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<double> >
        equality(new dotk::DOTk_EqualityConstraint<double>);
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        shared_ptr(new dotk::DOTk_MexInequalityConstraint<double>(inequality_ptr.get(), type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality(1, shared_ptr);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality, inequality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Inequality Constraint Jacobian **** \n");
    diagnostics.checkInequalityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
    inequality_ptr.release();
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeELP(const mxArray* input_[])
{
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    dotk::DOTk_MexArrayPtr equality_ptr;
    dotk::mex::parseEqualityConstraint(input_[1], equality_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(equality_ptr.get(), type));

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);


    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian **** \n");
    diagnostics.checkEqualityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    dotk::DOTk_Dual dual;
    dotk::mex::buildDualContainer(input_[0], dual);

    mexPrintf("\n **** Check Equality Constraint Adjoint of Jacobain **** \n");
    diagnostics.checkEqualityConstraintAdjointJacobian(control, dual, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
    equality_ptr.release();
}

void DOTk_MexDiagnosticsTypeLP::checkFirstDerivativeTypeCLP(const mxArray* input_[])
{
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    dotk::DOTk_MexArrayPtr equality_ptr;
    dotk::mex::parseEqualityConstraint(input_[1], equality_ptr);
    dotk::DOTk_MexArrayPtr inequality_ptr;
    dotk::mex::parseInequalityConstraint(input_[1], inequality_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(equality_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        shared_ptr(new dotk::DOTk_MexInequalityConstraint<double>(inequality_ptr.get(), type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality(1, shared_ptr);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality, inequality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Gradient **** \n");
    diagnostics.checkObjectiveGradient(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian **** \n");
    diagnostics.checkEqualityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());

    dotk::DOTk_Dual dual;
    dotk::mex::buildDualContainer(input_[0], dual);

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Adjoint of Jacobian **** \n");
    diagnostics.checkEqualityConstraintAdjointJacobian(control, dual, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    mexPrintf("\n **** Check Inequality Constraint Jacobian **** \n");
    diagnostics.checkInequalityConstraintJacobian(control, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
    equality_ptr.release();
    inequality_ptr.release();
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivativeTypeULP(const mxArray* input_[])
{
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Hessian **** \n");
    diagnostics.checkObjectiveHessian(control, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivativeTypeELP(const mxArray* input_[])
{
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    dotk::DOTk_MexArrayPtr equality_ptr;
    dotk::mex::parseEqualityConstraint(input_[1], equality_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(equality_ptr.get(), type));

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Hessian **** \n");
    diagnostics.checkObjectiveHessian(control, msg);
    mexPrintf(msg.str().c_str());

    dotk::DOTk_Dual dual;
    dotk::mex::buildDualContainer(input_[0], dual);

    msg.str("");
    mexPrintf("\n **** Check Equality Constraint Jacobian Derivative **** \n");
    diagnostics.checkEqualityConstraintJacobianDerivative(control, dual, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
    equality_ptr.release();
}

void DOTk_MexDiagnosticsTypeLP::checkSecondDerivativeTypeCLP(const mxArray* input_[])
{
    dotk::DOTk_MexArrayPtr objective_ptr;
    dotk::mex::parseObjectiveFunction(input_[1], objective_ptr);
    dotk::DOTk_MexArrayPtr equality_ptr;
    dotk::mex::parseEqualityConstraint(input_[1], equality_ptr);
    dotk::DOTk_MexArrayPtr inequality_ptr;
    dotk::mex::parseInequalityConstraint(input_[1], inequality_ptr);

    dotk::types::problem_t type = DOTk_MexDiagnostics::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(objective_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(equality_ptr.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        shared_ptr(new dotk::DOTk_MexInequalityConstraint<double>(inequality_ptr.get(), type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality(1, shared_ptr);

    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality, inequality);
    int lower_super_subscript = DOTk_MexDiagnostics::getLowerSuperScript();
    int upper_super_subscript = DOTk_MexDiagnostics::getUpperSuperScript();
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(lower_super_subscript, upper_super_subscript);

    std::ostringstream msg;
    dotk::DOTk_Control control;
    dotk::mex::buildControlContainer(input_[0], control);

    mexPrintf("\n **** Check Objective Function Hessian **** \n");
    diagnostics.checkObjectiveHessian(control, msg);
    mexPrintf(msg.str().c_str());

    msg.str("");
    dotk::DOTk_Dual dual;
    dotk::mex::buildDualContainer(input_[0], dual);

    mexPrintf("\n **** Check Equality Constraint Jacobian Derivative **** \n");
    diagnostics.checkEqualityConstraintJacobianDerivative(control, dual, msg);
    mexPrintf(msg.str().c_str());

    objective_ptr.release();
    equality_ptr.release();
    inequality_ptr.release();
}

}
