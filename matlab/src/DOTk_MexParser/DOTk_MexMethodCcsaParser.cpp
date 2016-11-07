/*
 * DOTk_MexMethodCcsaParser.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>

#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexMethodCcsaParser.hpp"

namespace dotk
{

namespace mex
{

dotk::ccsa::dual_solver_t getDualSolverType(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::nonlinearcg_t getDualSolverNLCG(const dotk::DOTk_MexArrayPtr & ptr_);

void parseMovingAsymptoteUpperBoundScale(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MovingAsymptoteUpperBoundScale")) == true)
    {
        output_ = 10.;
        std::string msg(" DOTk/MEX WARNING: MovingAsymptoteUpperBoundScale is NOT Defined. Default = 10. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr mma_upper_bound_scale;
    mma_upper_bound_scale.reset(mxDuplicateArray(mxGetField(options_, 0, "MovingAsymptoteUpperBoundScale")));
    output_ = mxGetScalar(mma_upper_bound_scale.get());
    mma_upper_bound_scale.release();
}

void parseMovingAsymptoteLowerBoundScale(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MovingAsymptoteLowerBoundScale")) == true)
    {
        output_ = 0.01;
        std::string msg(" DOTk/MEX WARNING: MovingAsymptoteLowerBoundScale is NOT Defined. Default = 0.01. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr mma_lower_bound_scale;
    mma_lower_bound_scale.reset(mxDuplicateArray(mxGetField(options_, 0, "MovingAsymptoteLowerBoundScale")));
    output_ = mxGetScalar(mma_lower_bound_scale.get());
    mma_lower_bound_scale.release();
}

void parseMovingAsymptoteExpansionParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MovingAsymptoteExpansionParameter")) == true)
    {
        output_ = 1.2;
        std::string msg(" DOTk/MEX WARNING: MovingAsymptoteExpansionParameter is NOT Defined. Default = 1.2. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr mma_expansion_parameter;
    mma_expansion_parameter.reset(mxDuplicateArray(mxGetField(options_, 0, "MovingAsymptoteExpansionParameter")));
    output_ = mxGetScalar(mma_expansion_parameter.get());
    mma_expansion_parameter.release();
}

void parseMovingAsymptoteContractionParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MovingAsymptoteContractionParameter")) == true)
    {
        output_ = 0.4;
        std::string msg(" DOTk/MEX WARNING: MovingAsymptoteContractionParameter is NOT Defined. Default = 0.4. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr mma_contraction_parameter;
    mma_contraction_parameter.reset(mxDuplicateArray(mxGetField(options_, 0, "MovingAsymptoteContractionParameter")));
    output_ = mxGetScalar(mma_contraction_parameter.get());
    mma_contraction_parameter.release();
}

void parseDualSolverGradientTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualSolverGradientTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: DualSolverGradientTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr gradient_tolerance;
    gradient_tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "DualSolverGradientTolerance")));
    output_ = mxGetScalar(gradient_tolerance.get());
    gradient_tolerance.release();
}

void parseDualSolverTrialStepTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualSolverTrialStepTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: DualSolverTrialStepTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr trial_step_tolerance;
    trial_step_tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "DualSolverTrialStepTolerance")));
    output_ = mxGetScalar(trial_step_tolerance.get());
    trial_step_tolerance.release();
}

void parseDualSolverObjectiveStagnationTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualSolverObjectiveStagnationTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: DualSolverObjectiveStagnationTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr objective_stagnation_tolerance;
    objective_stagnation_tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "DualSolverObjectiveStagnationTolerance")));
    output_ = mxGetScalar(objective_stagnation_tolerance.get());
    objective_stagnation_tolerance.release();
}

void parseDualSolverMaxNumberIterations(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualSolverMaxNumberIterations")) == true)
    {
        output_ = 10;
        std::string msg(" DOTk/MEX WARNING: DualSolverMaxNumberIterations is NOT Defined. Default = 10. \n");
        mexWarnMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr max_num_iterations;
    max_num_iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "DualSolverMaxNumberIterations")));
    output_ = static_cast<size_t>(mxGetScalar(max_num_iterations.get()));
    max_num_iterations.release();
}

void parseDualObjectiveEpsilonParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualObjectiveEpsilonParameter")) == true)
    {
        output_ = 1e-6;
        std::string msg(" DOTk/MEX WARNING: DualObjectiveEpsilonParameter is NOT Defined. Default = 1e-6. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr dual_objective_epsilon_parameter;
    dual_objective_epsilon_parameter.reset(mxDuplicateArray(mxGetField(options_, 0, "DualObjectiveEpsilonParameter")));
    output_ = mxGetScalar(dual_objective_epsilon_parameter.get());
    dual_objective_epsilon_parameter.release();
}

void parseDualSolverType(const mxArray* options_, dotk::ccsa::dual_solver_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualSolverType")) == true)
    {
        output_ = dotk::ccsa::dual_solver_t::NONLINEAR_CG;
        std::string msg(" DOTk/MEX WARNING: DualSolverType is NOT Defined. Default = Nonlinear Conjugate Gradient. \n");
        mexWarnMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr dual_solver_type;
    dual_solver_type.reset(mxDuplicateArray(mxGetField(options_, 0, "DualSolverType")));
    output_ = dotk::mex::getDualSolverType(dual_solver_type);
    dual_solver_type.release();
}

void parseDualObjectiveTrialControlBoundScaling(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualObjectiveTrialControlBoundScaling")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: DualObjectiveTrialControlBoundScaling is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr trial_control_bound_scale_parameter;
    trial_control_bound_scale_parameter.reset(mxDuplicateArray(mxGetField(options_, 0, "DualObjectiveTrialControlBoundScaling")));
    output_ = mxGetScalar(trial_control_bound_scale_parameter.get());
    trial_control_bound_scale_parameter.release();
}

void parseDualSolverTypeNLCG(const mxArray* options_, dotk::types::nonlinearcg_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualSolverTypeNLCG")) == true)
    {
        output_ = dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG;
        std::string msg(" DOTk/MEX WARNING: DualSolverTypeNLCG is NOT Defined. Default = POLAK-RIBIERE. \n");
        mexWarnMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr nonlinear_cg_type;
    nonlinear_cg_type.reset(mxDuplicateArray(mxGetField(options_, 0, "DualSolverTypeNLCG")));
    output_ = dotk::mex::getDualSolverNLCG(nonlinear_cg_type);
    nonlinear_cg_type.release();
}

void parseLineSearchStepLowerBound(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "LineSearchStepLowerBound")) == true)
    {
        output_ = 1e-3;
        std::string msg(" DOTk/MEX WARNING: LineSearchStepLowerBound is NOT Defined. Default = 1e-3. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr line_search_step_lower_bound;
    line_search_step_lower_bound.reset(mxDuplicateArray(mxGetField(options_, 0, "LineSearchStepLowerBound")));
    output_ = mxGetScalar(line_search_step_lower_bound.get());
    line_search_step_lower_bound.release();
}

void parseLineSearchStepUpperBound(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "LineSearchStepUpperBound")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: LineSearchStepUpperBound is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr line_search_step_upper_bound;
    line_search_step_upper_bound.reset(mxDuplicateArray(mxGetField(options_, 0, "LineSearchStepUpperBound")));
    output_ = mxGetScalar(line_search_step_upper_bound.get());
    line_search_step_upper_bound.release();
}

void parseSubProblemResidualTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "SubProblemResidualTolerance")) == true)
    {
        output_ = 1e-6;
        std::string msg(" DOTk/MEX WARNING: SubProblemResidualTolerance is NOT Defined. Default = 1e-6. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr sub_problem_residual_tolerance;
    sub_problem_residual_tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "SubProblemResidualTolerance")));
    output_ = mxGetScalar(sub_problem_residual_tolerance.get());
    sub_problem_residual_tolerance.release();
}

void parseSubProblemStagnationTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "SubProblemStagnationTolerance")) == true)
    {
        output_ = 1e-6;
        std::string msg(" DOTk/MEX WARNING: SubProblemStagnationTolerance is NOT Defined. Default = 1e-6. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr sub_problem_stagnation_tolerance;
    sub_problem_stagnation_tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "SubProblemStagnationTolerance")));
    output_ = mxGetScalar(sub_problem_stagnation_tolerance.get());
    sub_problem_stagnation_tolerance.release();
}

void parseMaxNumberSubProblemIterations(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumberSubProblemIterations")) == true)
    {
        output_ = 10;
        std::string msg(" DOTk/MEX WARNING: MaxNumberSubProblemIterations is NOT Defined. Default = 10. \n");
        mexWarnMsgTxt(msg.c_str());
    }
    dotk::DOTk_MexArrayPtr max_num_sub_problem_iterations;
    max_num_sub_problem_iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumberSubProblemIterations")));
    output_ = static_cast<size_t>(mxGetScalar(max_num_sub_problem_iterations.get()));
    max_num_sub_problem_iterations.release();
}

dotk::ccsa::dual_solver_t getDualSolverType(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::ccsa::dual_solver_t dual_solver_type;
    std::string method(mxArrayToString(ptr_.get()));

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

dotk::types::nonlinearcg_t getDualSolverNLCG(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::nonlinearcg_t nonlinear_cg_type;
    std::string method(mxArrayToString(ptr_.get()));

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
