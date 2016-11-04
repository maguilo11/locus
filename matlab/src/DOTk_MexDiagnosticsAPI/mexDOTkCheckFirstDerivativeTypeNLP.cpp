/*
 * mexDOTkCheckFirstDerivativeTypeNLP.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexDiagnosticsTypeNLP.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 0) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) diagnostics tool mexDOTkCheckFirstDerivativeTypeNLP\n"
                        " should be used as follows: mexDOTkCheckFirstDerivativeTypeNLP(Options, Interface)\n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg("\n DOTk Check First-Order Derivative Operators - Type Nonlinear Programming (NLP) \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexDiagnosticsTypeNLP diagnostics(pInput_);
    diagnostics.checkFirstDerivative(pInput_);
}
