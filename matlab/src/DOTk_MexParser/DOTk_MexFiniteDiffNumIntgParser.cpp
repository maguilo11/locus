/*
 * DOTk_MexFiniteDiffNumIntgParser.cpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexFiniteDiffNumIntgParser.hpp"

namespace dotk
{

namespace mex
{

dotk::types::numerical_integration_t getFiniteDiffNumIntgMethod(const mxArray* input_)
{
    std::string method(mxArrayToString(input_));
    dotk::types::numerical_integration_t type = dotk::types::FORWARD_FINITE_DIFF;
    if(method.compare("FORWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" FiniteDiffNumericalIntgMethod = FORWARD DIFFERENCE \n");
        type = dotk::types::FORWARD_FINITE_DIFF;
    }
    else if(method.compare("BACKWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" FiniteDiffNumericalIntgMethod = BACKWARD DIFFERENCE \n");
        type = dotk::types::BACKWARD_FINITE_DIFF;
    }
    else if(method.compare("CENTRAL_DIFFERENCE") == 0)
    {
        mexPrintf(" FiniteDiffNumericalIntgMethod = CENTRAL DIFFERENCE \n");
        type = dotk::types::CENTRAL_FINITE_DIFF;
    }
    else if(method.compare("SECOND_ORDER_FORWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" FiniteDiffNumericalIntgMethod = SECOND ORDER FORWARD DIFFERENCE \n");
        type = dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF;
    }
    else if(method.compare("THIRD_ORDER_FORWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" FiniteDiffNumericalIntgMethod = THIRD ORDER FORWARD DIFFERENCE \n");
        type = dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF;
    }
    else if(method.compare("THIRD_ORDER_BACKWARD_DIFFERENCE") == 0)
    {
        mexPrintf(" FiniteDiffNumericalIntgMethod = THIRD ORDER BACKWARD DIFFERENCE \n");
        type = dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF;
    }
    else
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> NumericalDifferentiationMethod keyword is misspelled."
                << " NumericalDifferentiationMethod set to FORWARD FINITE DIFFERENCE.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    return (type);
}

double parseNumericalDifferentiationEpsilon(const mxArray* input_)
{
    double output = 1e-6;
    if(mxGetField(input_, 0, "NumericalDifferentiationEpsilon") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> NumericalDifferentiationEpsilon keyword is NULL. NumericalDifferentiationEpsilon set to 1e-6.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumericalDifferentiationEpsilon"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::numerical_integration_t parseNumericalDifferentiationMethod(const mxArray* input_)
{
    dotk::types::numerical_integration_t output = dotk::types::FORWARD_FINITE_DIFF;
    if(mxGetField(input_, 0, "NumericalDifferentiationMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> NumericalDifferentiationMethod keyword is NULL."
                << " NumericalDifferentiationMethod set to FORWARD FINITE DIFFERENCE.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumericalDifferentiationMethod"));
        output = dotk::mex::getFiniteDiffNumIntgMethod(value);
        mxDestroyArray(value);
    }
    return (output);
}

}

}
