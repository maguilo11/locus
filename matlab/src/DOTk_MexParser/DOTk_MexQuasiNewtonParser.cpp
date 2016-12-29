/*
 * DOTk_MexQuasiNewtonParser.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexQuasiNewtonParser.hpp"

namespace dotk
{

namespace mex
{

dotk::types::invhessian_t getQuasiNewtonMethod(const mxArray* input_);

size_t parseQuasiNewtonStorage(const mxArray* input_)
{
    size_t output = 4;
    if(mxGetField(input_, 0, "QuasiNewtonStorage") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> QuasiNewtonStorage keyword is NULL. QuasiNewtonStorage set to 4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "QuasiNewtonStorage"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::invhessian_t parseQuasiNewtonMethod(const mxArray* input_)
{
    dotk::types::invhessian_t output = dotk::types::BFGS_INV_HESS;
    if(mxGetField(input_, 0, "QuasiNewtonMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> QuasiNewtonMethod keyword is NULL. QuasiNewtonMethod set to BFGS.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "QuasiNewtonMethod"));
        output = dotk::mex::getQuasiNewtonMethod(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::invhessian_t getQuasiNewtonMethod(const mxArray* input_)
{
    dotk::types::invhessian_t type = dotk::types::INV_HESS_DISABLED;
    std::string method(mxArrayToString(input_));

    if(method.compare("LBFGS") == 0)
    {
        mexPrintf(" Quasi-Newton Method = LBFGS \n");
        type = dotk::types::LBFGS_INV_HESS;
    }
    else if(method.compare("LDFP") == 0)
    {
        mexPrintf(" Quasi-Newton Method = LDFP \n");
        type = dotk::types::LDFP_INV_HESS;
    }
    else if(method.compare("LSR1") == 0)
    {
        mexPrintf(" Quasi-Newton Method = LSR1 \n");
        type = dotk::types::LSR1_INV_HESS;
    }
    else if(method.compare("SR1") == 0)
    {
        mexPrintf(" Quasi-Newton Method = SR1 \n");
        type = dotk::types::SR1_INV_HESS;
    }
    else if(method.compare("BFGS") == 0)
    {
        mexPrintf(" Quasi-Newton Method = BFGS \n");
        type = dotk::types::BFGS_INV_HESS;
    }
    else if(method.compare("USER_DEFINED") == 0)
    {
        mexPrintf(" Quasi-Newton Method = USER_DEFINED \n");
        type = dotk::types::USER_DEFINED_INV_HESS;
    }
    else if(method.compare("BARZILAI_BORWEIN") == 0)
    {
        mexPrintf(" Quasi-Newton Method = BARZILAI_BORWEIN \n");
        type = dotk::types::BARZILAIBORWEIN_INV_HESS;
    }
    else
    {
        type = dotk::types::BFGS_INV_HESS;
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> QuasiNewtonMethod keyword is misspelled. QuasiNewtonMethod set to BFGS.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    return (type);
}

}

}
