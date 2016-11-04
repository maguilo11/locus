/*
 * DOTk_MexFiniteDiffNumIntgParser.cpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexFiniteDiffNumIntgParser.hpp"

namespace dotk
{

namespace mex
{

dotk::types::numerical_integration_t getFiniteDiffNumIntgMethod(dotk::DOTk_MexArrayPtr & ptr_)
{
    std::string method(mxArrayToString(ptr_.get()));
    dotk::types::numerical_integration_t type = dotk::types::NUM_INTG_DISABLED;

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
        type = dotk::types::FORWARD_FINITE_DIFF;
        std::string msg(" DOTk/MEX WARNING: Invalid Finite Difference Numerical Integration Method. Default = FORWARD DIFFERENCE. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (type);
}

void parseNumericalDifferentiationEpsilon(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "NumericalDifferentiationEpsilon")) == true)
    {
        output_ = 1e-6;
        std::string msg(" DOTk/MEX WARNING: NumericalDifferentiationEpsilon is NOT Defined. Default = 1e-6. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr iterations;
    iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "NumericalDifferentiationEpsilon")));
    output_ = mxGetScalar(iterations.get());
    iterations.release();
}

void parseNumericalDifferentiationMethod(const mxArray* options_, dotk::types::numerical_integration_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "NumericalDifferentiationMethod")) == true)
    {
        output_ = dotk::types::FORWARD_FINITE_DIFF;
        std::string msg(" DOTk/MEX WARNING: NumericalDifferentiationMethod is NOT Defined. Default = FORWARD DIFFERENCE. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "NumericalDifferentiationMethod")));
    output_ = dotk::mex::getFiniteDiffNumIntgMethod(type);
    type.release();
}

}

}
