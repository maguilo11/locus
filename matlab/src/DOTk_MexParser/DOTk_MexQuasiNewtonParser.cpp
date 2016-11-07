/*
 * DOTk_MexQuasiNewtonParser.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>

#include "DOTk_MexQuasiNewtonParser.hpp"

namespace dotk
{

namespace mex
{

dotk::types::invhessian_t getQuasiNewtonMethod(const dotk::DOTk_MexArrayPtr & ptr_);

void parseQuasiNewtonStorage(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "QuasiNewtonStorage")) == true)
    {
        output_ = 4;
        std::string msg(" DOTk/MEX WARNING: QuasiNewtonStorage is NOT Defined. Default = 4. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr data;
    data.reset(mxDuplicateArray(mxGetField(options_, 0, "QuasiNewtonStorage")));
    output_ = static_cast<size_t>(mxGetScalar(data.get()));
    data.release();
}

void parseQuasiNewtonMethod(const mxArray* options_, dotk::types::invhessian_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "QuasiNewtonMethod")) == true)
    {
        output_ = dotk::types::BFGS_INV_HESS;
        std::string msg(" DOTk/MEX WARNING: QuasiNewtonMethod is NOT Defined. Default = BFGS. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "QuasiNewtonMethod")));
    output_ = dotk::mex::getQuasiNewtonMethod(type);
    type.release();
}

dotk::types::invhessian_t getQuasiNewtonMethod(const dotk::DOTk_MexArrayPtr & ptr_)
{
    dotk::types::invhessian_t type = dotk::types::INV_HESS_DISABLED;
    std::string method(mxArrayToString(ptr_.get()));

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
        std::string msg(" DOTk/MEX WARNING: Invalid Quasi-Newton Method. Default = BFGS. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (type);
}

}

}
