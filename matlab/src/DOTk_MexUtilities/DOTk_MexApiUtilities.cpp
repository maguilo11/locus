/*
 * DOTk_MexApiUtilities.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <sstream>

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexApiUtilities.hpp"

namespace dotk
{

namespace mex
{

void handleException(mxArray* err_, std::string msg_)
{
    if(err_)
    {
        // In the cass of an exception, grab the report
        mxArray* input[1] = {err_};
        mxArray* output[1];
        mexCallMATLAB(1, output, 1, input, "getReport");
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
        string_stream << msg_ << std::endl << std::endl << report;
        mexErrMsgTxt(string_stream.str().c_str());
    }
}

size_t getMexArrayDim(const dotk::DOTk_MexArrayPtr & ptr_)
{
    if(mxIsEmpty(ptr_.get()))
    {
        std::string msg(" DOTK/MEX ERROR: NULL MexArrayPtr In dotk::mex::getMexArrayDim \n");
        mexErrMsgTxt(msg.c_str());
    }

    size_t nrows = mxGetM(ptr_.get());
    size_t ncols = mxGetN(ptr_.get());

    if( nrows > 1 && ncols > 1 )
    {
        std::string msg(" DOTK/MEX ERROR: Invalid MexArrayPtr Dimensions In dotk::mex::getMexArrayDim.\n"
                        " 2D-MEX Array Used Instead Of A 1D-MEX Array. \n");
        mexErrMsgTxt(msg.c_str());
    }

    size_t dim = nrows > ncols ? nrows : ncols;

    return(dim);
}

void setDOTkData(const dotk::DOTk_MexArrayPtr & ptr_, dotk::Vector<double> & data_)
{
    size_t data_dim = data_.size();
    size_t mex_array_dim = dotk::mex::getMexArrayDim(ptr_);

    if(mex_array_dim != data_dim)
    {
        std::string msg(" DOTK/MEX ERROR: Input MEX Array Dim IS NOT EQUAL to DOTk Vector Dim. Check Input Data Dimensions. \n");
        mexErrMsgTxt(msg.c_str());
    }

    dotk::mex::copyData(mex_array_dim, mxGetPr(ptr_.get()), data_);
}

void copyData(size_t input_dim_, double* input_, dotk::Vector<double> & output_)
{
    if(input_dim_ != output_.size())
    {
        std::string msg(" DOTK/MEX ERROR: Input Array Dim IS NOT EQUAL To Output Array Dim. Check dotk::mex::copyData. \n");
        mexErrMsgTxt(msg.c_str());
    }

    for(size_t index = 0; index < input_dim_; ++index)
    {
        output_[index] = input_[index];
    }
}

dotk::types::problem_t getProblemType(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::problem_t type = dotk::types::PROBLEM_TYPE_UNDEFINED;
    std::string option(mxArrayToString(ptr_.get()));

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
        std::string msg(" DOTk/MEX ERROR: Invalid Problem Type. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }

    return (type);
}

dotk::types::container_t getContainerType(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::container_t type = dotk::types::USER_DEFINED_CONTAINER;
    std::string option(mxArrayToString(ptr_.get()));

    if(option.compare("SERIAL_VECTOR") == 0)
    {
        type = dotk::types::SERIAL_VECTOR;
    }
    else if(option.compare("SERIAL_ARRAY") == 0)
    {
        type = dotk::types::SERIAL_ARRAY;
    }
    else
    {
        std::string msg(" DOTk/MEX WARNING: Invalid DOTk Container Type. Default = Serial C Array. \n");
        mexWarnMsgTxt(msg.c_str());
        type = dotk::types::SERIAL_ARRAY;
    }

    return (type);
}

dotk::types::display_t getDiagnosticsDisplayOption(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::display_t type = dotk::types::OFF;
    std::string option(mxArrayToString(ptr_.get()));

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
        std::string msg(" DOTk/MEX ERROR: Invalid Diagnostics Display Option. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }

    return (type);
}

dotk::types::line_search_t getLineSearchMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    std::string line_search_method(mxArrayToString(ptr_.get()));
    dotk::types::line_search_t step_t = dotk::types::LINE_SEARCH_DISABLED;

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
        step_t = dotk::types::BACKTRACKING_CUBIC_INTRP;
        std::string msg(" DOTk/MEX WARNING: Invalid Line Search Method. Default = CUBIC INTRP. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (step_t);
}

dotk::types::trustregion_t getTrustRegionMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    std::string trust_region_method(mxArrayToString(ptr_.get()));
    dotk::types::trustregion_t type = dotk::types::TRUST_REGION_DISABLED;

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
        type = dotk::types::TRUST_REGION_DOGLEG;
        std::string msg(" DOTk/MEX WARNING: Invalid Trust Region Method. Default = Dogleg. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (type);
}

dotk::types::nonlinearcg_t getNonlinearCgMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::nonlinearcg_t type;
    std::string method(mxArrayToString(ptr_.get()));

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
        type = dotk::types::HAGER_ZHANG_NLCG;
        std::string msg(" DOTk/MEX WARNING: Invalid Nonlinear Conjugate Gradient Method. Default = HAGER ZHANG. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (type);
}

dotk::types::gradient_t getGradientComputationMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    std::string method(mxArrayToString(ptr_.get()));
    dotk::types::gradient_t type = dotk::types::GRADIENT_OPERATOR_DISABLED;

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
        std::string msg(" DOTk/MEX ERROR: Invalid Gradient Computation Method. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }

    return (type);
}

dotk::types::hessian_t getHessianComputationMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::hessian_t type = dotk::types::HESSIAN_DISABLED;
    std::string method(mxArrayToString(ptr_.get()));

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
        std::string msg(" DOTk/MEX ERROR: Invalid Hessian Computation Method. See Users' Manual. \n");
        mexErrMsgTxt(msg.c_str());
    }

    return (type);
}

dotk::types::constraint_method_t getBoundConstraintMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    std::string method(mxArrayToString(ptr_.get()));
    dotk::types::constraint_method_t type = dotk::types::CONSTRAINT_METHOD_DISABLED;

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
        type = dotk::types::FEASIBLE_DIR;
        std::string msg(" DOTk/MEX WARNING: Invalid Bound Constraint Computation Method. Default = FEASIBLE DIRECTION. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (type);
}

}

}
