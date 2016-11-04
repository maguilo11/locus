/*
 * mexDOTkCheckSecondDerivativeTypeLP.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexDiagnosticsTypeLP.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 0) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) diagnostics tool mexDOTkCheckSecondDerivativeTypeLP\n"
                        " should be used as follows: mexDOTkCheckSecondDerivativeTypeLP(Options, Interface)\n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg("\n DOTk Check Second-Order Derivative Operators - Type Linear Programming (LP) \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexDiagnosticsTypeLP diagnostics(pInput_);
    diagnostics.checkSecondDerivative(pInput_);
}
