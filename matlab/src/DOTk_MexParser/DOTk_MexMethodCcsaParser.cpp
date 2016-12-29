/*
 * DOTk_MexMethodCcsaParser.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexMethodCcsaParser.hpp"

namespace dotk
{

namespace mex
{

dotk::ccsa::dual_solver_t getDualSolverType(const mxArray* input_);
dotk::types::nonlinearcg_t getDualSolverNLCG(const mxArray* input_);

double parseMovingAsymptoteUpperBoundScale(const mxArray* input_)
{
    double output = 10;
    if(mxGetField(input_, 0, "AsymptotesUpperBoundScaling") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> AsymptotesUpperBoundScaling keyword is NULL. AsymptotesUpperBoundScaling set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "AsymptotesUpperBoundScaling"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMovingAsymptoteLowerBoundScale(const mxArray* input_)
{
    double output = 0.01;
    if(mxGetField(input_, 0, "AsymptotesLowerBoundScaling") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> AsymptotesLowerBoundScaling keyword is NULL. AsymptotesLowerBoundScaling set to 0.01.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "AsymptotesLowerBoundScaling"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMovingAsymptoteExpansionParameter(const mxArray* input_)
{
    double output = 1.2;
    if(mxGetField(input_, 0, "AsymptotesExpansionParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> AsymptotesExpansionParameter keyword is NULL. AsymptotesExpansionParameter set to 1.2.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "AsymptotesExpansionParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMovingAsymptoteContractionParameter(const mxArray* input_)
{
    double output = 0.4;
    if(mxGetField(input_, 0, "AsymptotesContractionParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> AsymptotesContractionParameter keyword is NULL. AsymptotesContractionParameter set to 0.4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "AsymptotesContractionParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualSolverGradientTolerance(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "DualSolverGradientTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualSolverGradientTolerance keyword is NULL. DualSolverGradientTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualSolverGradientTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualSolverStepTolerance(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "DualSolverStepTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualSolverStepTolerance keyword is NULL. DualSolverStepTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualSolverStepTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualSolverObjectiveStagnationTolerance(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "DualObjectiveStagnationTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualObjectiveStagnationTolerance keyword is NULL. DualObjectiveStagnationTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualObjectiveStagnationTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualObjectiveRelaxationParameter(const mxArray* input_)
{
    double output = 1e-6;
    if(mxGetField(input_, 0, "DualObjectiveRelaxationParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualObjectiveEpsilonParameter keyword is NULL. DualObjectiveRelaxationParameter set to 1e-6.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualObjectiveRelaxationParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualObjectiveControlBoundsScaling(const mxArray* input_)
{
    double output = 0.5;
    if(mxGetField(input_, 0, "DualObjectiveControlBoundsScaling") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualObjectiveControlBoundsScaling keyword is NULL. DualObjectiveControlBoundsScaling set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualObjectiveControlBoundsScaling"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseLineSearchStepLowerBound(const mxArray* input_)
{
    double output = 1e-3;
    if(mxGetField(input_, 0, "LineSearchStepLowerBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> LineSearchStepLowerBound keyword is NULL. LineSearchStepLowerBound set to 1e-3.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "LineSearchStepLowerBound"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseLineSearchStepUpperBound(const mxArray* input_)
{
    double output = 0.5;
    if(mxGetField(input_, 0, "LineSearchStepUpperBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> LineSearchStepUpperBound keyword is NULL. LineSearchStepUpperBound set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "LineSearchStepUpperBound"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseSubProblemResidualTolerance(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "SubProblemResidualTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> SubProblemResidualTolerance keyword is NULL. SubProblemResidualTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "SubProblemResidualTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseSubProblemStagnationTolerance(const mxArray* input_)
{
    double output = 1e-6;
    if(mxGetField(input_, 0, "SubProblemStagnationTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> SubProblemStagnationTolerance keyword is NULL. SubProblemStagnationTolerance set to 1e-6.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "SubProblemStagnationTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseDualSolverMaxNumberIterations(const mxArray* input_)
{
    size_t output = 10;
    if(mxGetField(input_, 0, "DualSolverMaxNumberIterations") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualSolverMaxNumberIterations keyword is NULL. DualSolverMaxNumberIterations set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualSolverMaxNumberIterations"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseMaxNumberSubProblemIterations(const mxArray* input_)
{
    size_t output = 10;
    if(mxGetField(input_, 0, "MaxNumberSubProblemIterations") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumberSubProblemIterations keyword is NULL. MaxNumberSubProblemIterations set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberSubProblemIterations"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

dotk::ccsa::dual_solver_t parseDualSolverType(const mxArray* input_)
{
    dotk::ccsa::dual_solver_t output = dotk::ccsa::dual_solver_t::NONLINEAR_CG;
    if(mxGetField(input_, 0, "DualSolverType") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualSolverType keyword is NULL. DualSolverType set to Nonlinear Conjugate Gradient.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualSolverType"));
        output = dotk::mex::getDualSolverType(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::nonlinearcg_t parseDualSolverTypeNLCG(const mxArray* input_)
{
    dotk::types::nonlinearcg_t output = dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG;
    if(mxGetField(input_, 0, "NonlinearCG_Type") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualSolverTypeNLCG keyword is NULL. NonlinearCG_Type set to POLAK RIBIERE.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NonlinearCG_Type"));
        output = dotk::mex::getDualSolverNLCG(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::ccsa::dual_solver_t getDualSolverType(const mxArray* input_)
{
    dotk::ccsa::dual_solver_t dual_solver_type;
    std::string method(mxArrayToString(input_));
    if(method.compare("NLCG") == 0)
    {
        mexPrintf(" Dual Solver = Nonlinear Conjugate Gradient \n");
        dual_solver_type = dotk::ccsa::dual_solver_t::NONLINEAR_CG;
    }
    else if(method.compare("QUASI_NEWTON") == 0)
    {
        mexPrintf(" Dual Solver = Quasi-Newton \n");
        dual_solver_type = dotk::ccsa::dual_solver_t::QUASI_NEWTON;
    }
    else
    {
        dual_solver_type = dotk::ccsa::dual_solver_t::NONLINEAR_CG;
        std::string msg(" DOTk/MEX WARNING: Invalid Dual Solver. Default = NONLINEAR CONJUGATE GRADIENT. \n");
        mexWarnMsgTxt(msg.c_str());
    }
    return (dual_solver_type);
}

dotk::types::nonlinearcg_t getDualSolverNLCG(const mxArray* input_)
{
    dotk::types::nonlinearcg_t nonlinear_cg_type;
    std::string method(mxArrayToString(input_));
    if(method.compare("FLETCHER_REEVES") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = FLETCHER REEVES \n");
        nonlinear_cg_type = dotk::types::nonlinearcg_t::FLETCHER_REEVES_NLCG;
    }
    else if(method.compare("POLAK_RIBIERE") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = POLAK RIBIERE \n");
        nonlinear_cg_type = dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG;
    }
    else if(method.compare("HESTENES_STIEFEL") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = HESTENES STIEFEL \n");
        nonlinear_cg_type = dotk::types::nonlinearcg_t::HESTENES_STIEFEL_NLCG;
    }
    else if(method.compare("CONJUGATE_DESCENT") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = CONJUGATE DESCENT \n");
        nonlinear_cg_type = dotk::types::nonlinearcg_t::CONJUGATE_DESCENT_NLCG;
    }
    else if(method.compare("DAI_YUAN") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = DAI YUAN \n");
        nonlinear_cg_type = dotk::types::nonlinearcg_t::DAI_YUAN_NLCG;
    }
    else if(method.compare("LIU_STOREY") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = LIU STOREY \n");
        nonlinear_cg_type = dotk::types::nonlinearcg_t::LIU_STOREY_NLCG;
    }
    else
    {
        nonlinear_cg_type = dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG;
        std::string msg(" DOTk/MEX WARNING: Invalid Nonlinear Conjugate Gradient Method. Default = POLAK RIBIERE. \n");
        mexWarnMsgTxt(msg.c_str());
    }
    return (nonlinear_cg_type);
}

}

}
