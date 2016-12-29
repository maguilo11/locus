/*
 * DOTk_MexApiUtilities.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexApiUtilities.hpp"

namespace dotk
{

namespace mex
{

void destroy(mxArray* input_)
{
    if(input_ != nullptr)
    {
        mxDestroyArray(input_);
    }
    input_ = nullptr;
}

void handleException(mxArray* input_, std::string output_)
{
    if(input_ != nullptr)
    {
        // In the cass of an exception, grab the report
        mxArray* input[1] = {input_};
        mxArray* output[1];
        mexCallMATLABWithObject(1, output, 1, input, "getReport");
        // Turn the report into a string
        mwSize char_limit = 256;
        char report_[char_limit];
        mxGetString(output[0], report_, char_limit);

        // The report has extra information that we don't want.  Hence,
        // we eliminate both the first line as well as the last two lines.
        // The first line is supposed to say what function this occured in,
        // but Matlab gets confused since we're doing mex trickery. The
        // last two lines will automatically be repeated by mexErrMsgTxt.
        std::string report = report_;
        size_t position = report.find("\n");
        report = report.substr(position + 1);
        position = report.rfind("\n");
        report = report.substr(0, position);
        position = report.rfind("\n");
        report = report.substr(0, position);
        position = report.rfind("\n");
        report = report.substr(0, position);

        // Now, tack on our additional error message and then return control to Matlab.
        std::stringstream string_stream;
        string_stream << output_ << std::endl << std::endl << report;
        mexErrMsgTxt(string_stream.str().c_str());
    }
}

dotk::types::problem_t getProblemType(const mxArray* input_)
{
    std::string option(mxArrayToString(input_));
    dotk::types::problem_t type = dotk::types::PROBLEM_TYPE_UNDEFINED;

    if(option.compare("ULP") == 0)
    {
        mexPrintf(" Problem Type = UNCONSTRAINED LINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_ULP;
    }
    else if(option.compare("UNLP") == 0)
    {
        mexPrintf(" Problem Type = UNCONSTRAINED NONLINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_UNLP;
    }
    else if(option.compare("ELP") == 0)
    {
        mexPrintf(" Problem Type = EQUALITY CONSTRAINED LINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_ELP;
    }
    else if(option.compare("ENLP") == 0)
    {
        mexPrintf(" Problem Type = EQUALITY CONSTRAINED NONLINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_ENLP;
    }
    else if(option.compare("LP_BOUND") == 0)
    {
        mexPrintf(" Problem Type = BOUND CONSTRAINED LINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_LP_BOUND;
    }
    else if(option.compare("NLP_BOUND") == 0)
    {
        mexPrintf(" Problem Type = BOUND CONSTRAINED NONLINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_NLP_BOUND;
    }
    else if(option.compare("ELP_BOUND") == 0)
    {
        mexPrintf(" Problem Type = EQUALITY CONSTRAINED LINEAR PROGRAMMING PLUS BOUNDS \n");
        type = dotk::types::TYPE_ELP_BOUND;
    }
    else if(option.compare("ENLP_BOUND") == 0)
    {
        mexPrintf(" Problem Type = EQUALITY CONSTRAINED NONLINEAR PROGRAMMING PLUS BOUNDS \n");
        type = dotk::types::TYPE_ENLP_BOUND;
    }
    else if(option.compare("CLP") == 0)
    {
        mexPrintf(" Problem Type = CONSTRAINED LINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_CLP;
    }
    else if(option.compare("CNLP") == 0)
    {
        mexPrintf(" Problem Type = CONSTRAINED NONLINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_CNLP;
    }
    else if(option.compare("ILP") == 0)
    {
        mexPrintf(" Problem Type = INEQUALITY CONSTRAINED LINEAR PROGRAMMING \n");
        type = dotk::types::TYPE_ILP;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> PROBLEM TYPE keyword is misspelled. See Users' Manual for valid options.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    return (type);
}

dotk::types::display_t getDiagnosticsDisplayOption(const mxArray* input_)
{
    std::string option(mxArrayToString(input_));
    dotk::types::display_t type = dotk::types::OFF;
    if(option.compare("ITERATION") == 0)
    {
        mexPrintf(" Diagnostics Display Option = ITERATION \n");
        type = dotk::types::ITERATION;
    }
    else if(option.compare("FINAL") == 0)
    {
        mexPrintf(" Diagnostics Display Option = FINAL \n");
        type = dotk::types::FINAL;
    }
    else if(option.compare("OFF") == 0)
    {
        mexPrintf(" Diagnostics Display Option = OFF \n");
        type = dotk::types::OFF;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Diagnostics display option keyword is misspelled. Option set to OFF.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    return (type);
}

dotk::types::line_search_t getLineSearchMethod(const mxArray* input_)
{
    std::string line_search_method(mxArrayToString(input_));
    dotk::types::line_search_t step_t = dotk::types::BACKTRACKING_CUBIC_INTRP;

    if(line_search_method.compare("ARMIJO") == 0)
    {
        mexPrintf(" LineSearchMethod = ARMIJO \n");
        step_t = dotk::types::BACKTRACKING_ARMIJO;
    }
    else if(line_search_method.compare("GOLDSTEIN") == 0)
    {
        mexPrintf(" LineSearchMethod = GOLDSTEIN \n");
        step_t = dotk::types::BACKTRACKING_GOLDSTEIN;
    }
    else if(line_search_method.compare("CUBIC_INTRP") == 0)
    {
        mexPrintf(" LineSearchMethod = CUBIC INTRP \n");
        step_t = dotk::types::BACKTRACKING_CUBIC_INTRP;
    }
    else if(line_search_method.compare("GOLDENSECTION") == 0)
    {
        mexPrintf(" LineSearchMethod = GOLDENSECTION \n");
        step_t = dotk::types::GOLDENSECTION;
    }
    else if(line_search_method.compare("HAGER_ZHANG") == 0)
    {
        mexPrintf(" LineSearchMethod = HAGER ZHANG \n");
        step_t = dotk::types::LINE_SEARCH_HAGER_ZHANG;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> LineSearchMethod keyword is misspelled. LineSearchMethod set to BACKTRACKINGCUBIC INTRPOLATION.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    return (step_t);
}

dotk::types::trustregion_t getTrustRegionMethod(const mxArray* input_)
{
    std::string trust_region_method(mxArrayToString(input_));
    dotk::types::trustregion_t type = dotk::types::TRUST_REGION_DOGLEG;

    if(trust_region_method.compare("CAUCHY") == 0)
    {
        mexPrintf(" TrustRegionMethod = CAUCHY \n");
        type = dotk::types::TRUST_REGION_CAUCHY;
    }
    else if(trust_region_method.compare("DOGLEG") == 0)
    {
        mexPrintf(" TrustRegionMethod = DOGLEG \n");
        type = dotk::types::TRUST_REGION_DOGLEG;
    }
    else if(trust_region_method.compare("DOUBLE_DOGLEG") == 0)
    {
        mexPrintf(" TrustRegionMethod = DOUBLE_DOGLEG \n");
        type = dotk::types::TRUST_REGION_DOUBLE_DOGLEG;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TrustRegionMethod keyword is misspelled. TrustRegionMethod set to DOGLEG.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }

    return (type);
}

dotk::types::nonlinearcg_t getNonlinearCgMethod(const mxArray* input_)
{
    std::string method(mxArrayToString(input_));
    dotk::types::nonlinearcg_t type = dotk::types::HAGER_ZHANG_NLCG;
    if(method.compare("FLETCHER_REEVES") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = FLETCHER REEVES \n");
        type = dotk::types::FLETCHER_REEVES_NLCG;
    }
    else if(method.compare("POLAK_RIBIERE") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = POLAK RIBIERE \n");
        type = dotk::types::POLAK_RIBIERE_NLCG;
    }
    else if(method.compare("HESTENES_STIEFEL") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = HESTENES STIEFEL \n");
        type = dotk::types::HESTENES_STIEFEL_NLCG;
    }
    else if(method.compare("CONJUGATE_DESCENT") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = CONJUGATE DESCENT \n");
        type = dotk::types::CONJUGATE_DESCENT_NLCG;
    }
    else if(method.compare("HAGER_ZHANG") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = HAGER ZHANG \n");
        type = dotk::types::HAGER_ZHANG_NLCG;
    }
    else if(method.compare("DAI_LIAO") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = DAI LIAO \n");
        type = dotk::types::DAI_LIAO_NLCG;
    }
    else if(method.compare("DAI_YUAN") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = DAI YUAN \n");
        type = dotk::types::DAI_YUAN_NLCG;
    }
    else if(method.compare("DAI_YUAN_HYBRID") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = DAI YUAN HYBRID \n");
        type = dotk::types::DAI_YUAN_HYBRID_NLCG;
    }
    else if(method.compare("PERRY_SHANNO") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = PERRY SHANNO \n");
        type = dotk::types::PERRY_SHANNO_NLCG;
    }
    else if(method.compare("LIU_STOREY") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = LIU STOREY \n");
        type = dotk::types::LIU_STOREY_NLCG;
    }
    else if(method.compare("DANIELS") == 0)
    {
        mexPrintf(" Nonlinear Conjugate Gradient Method = DANIELS \n");
        type = dotk::types::LIU_STOREY_NLCG;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Nonlinear Conjugate Gradient keyword is misspelled. Algorithm set to HAGER-ZHANG.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    return (type);
}

dotk::types::gradient_t getGradientComputationMethod(const mxArray* input_)
{
    std::string method(mxArrayToString(input_));
    dotk::types::gradient_t type = dotk::types::CENTRAL_DIFF_GRAD;
    if(method.compare("FORWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" GradientComputationMethod = FORWARD DIFFERENCE \n");
        type = dotk::types::FORWARD_DIFF_GRAD;
    }
    else if(method.compare("BACKWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" GradientComputationMethod = BACKWARD DIFFERENCE \n");
        type = dotk::types::BACKWARD_DIFF_GRAD;
    }
    else if(method.compare("CENTRAL_DIFFERENCE") == 0)
    {
        mexPrintf(" GradientComputationMethod = CENTRAL DIFFERENCE \n");
        type = dotk::types::CENTRAL_DIFF_GRAD;
    }
    else if(method.compare("USER_DEFINED") == 0)
    {
        mexPrintf(" GradientComputationMethod = USER DEFINED \n");
        type = dotk::types::USER_DEFINED_GRAD;
    }
    else if(method.compare("PARALLEL_FORWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" GradientComputationMethod = PARALLEL FORWARD DIFFERENCE \n");
        type = dotk::types::PARALLEL_FORWARD_DIFF_GRAD;
    }
    else if(method.compare("PARALLEL_BACKWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" GradientComputationMethod = PARALLEL BACKWARD DIFFERENCE \n");
        type = dotk::types::PARALLEL_BACKWARD_DIFF_GRAD;
    }
    else if(method.compare("PARALLEL_CENTRAL_DIFFERENCE") == 0)
    {
        mexPrintf(" GradientComputationMethod = PARALLEL CENTRAL DIFFERENCE \n");
        type = dotk::types::PARALLEL_CENTRAL_DIFF_GRAD;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> GradientComputationMethod keyword is misspelled.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    return (type);
}

dotk::types::hessian_t getHessianComputationMethod(const mxArray* input_)
{
    dotk::types::hessian_t type = dotk::types::DFP_HESS;
    std::string method(mxArrayToString(input_));
    if(method.compare("LBFGS") == 0)
    {
        mexPrintf(" Hessian Computation Method = LBFGS \n");
        type = dotk::types::LBFGS_HESS;
    }
    else if(method.compare("LDFP") == 0)
    {
        mexPrintf(" Hessian Computation Method = LDFP \n");
        type = dotk::types::LDFP_HESS;
    }
    else if(method.compare("LSR1") == 0)
    {
        mexPrintf(" Hessian Computation Method = LSR1 \n");
        type = dotk::types::LSR1_HESS;
    }
    else if(method.compare("SR1") == 0)
    {
        mexPrintf(" Hessian Computation Method = SR1 \n");
        type = dotk::types::SR1_HESS;
    }
    else if(method.compare("DFP") == 0)
    {
        mexPrintf(" Hessian Computation Method = DFP \n");
        type = dotk::types::DFP_HESS;
    }
    else if(method.compare("USER_DEFINED") == 0)
    {
        mexPrintf(" Hessian Computation Method = USER DEFINED \n");
        type = dotk::types::USER_DEFINED_HESS;
    }
    else if(method.compare("USER_DEFINED_TYPE_CNP") == 0)
    {
        mexPrintf(" Hessian Computation Method = USER DEFINED - SQP SOLVER \n");
        type = dotk::types::USER_DEFINED_HESS_TYPE_CNP;
    }
    else if(method.compare("BARZILAI_BORWEIN") == 0)
    {
        mexPrintf(" Hessian Computation Method = BARZILAI BORWEIN \n");
        type = dotk::types::BARZILAIBORWEIN_HESS;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> HessianComputationMethod keyword is misspelled.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    return (type);
}

dotk::types::constraint_method_t getBoundConstraintMethod(const mxArray* input_)
{
    std::string method(mxArrayToString(input_));
    dotk::types::constraint_method_t type = dotk::types::FEASIBLE_DIR;
    if(method.compare("FEASIBLE_DIR") == 0)
    {
        mexPrintf(" BoundConstraintMethod = FEASIBLE DIRECTION \n");
        type = dotk::types::FEASIBLE_DIR;
    }
    else if(method.compare("PROJECTION_ALONG_FEASIBLE_DIR") == 0)
    {
        mexPrintf(" BoundConstraintMethod = PROJECTION ALONG FEASIBLE DIRECTION \n");
        type = dotk::types::PROJECTION_ALONG_FEASIBLE_DIR;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> BoundConstraintMethod method keyword is misspelled."
                << " BoundConstraintMethod method set to FEASIBLE DIRECTION.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    return (type);
}

}

}
