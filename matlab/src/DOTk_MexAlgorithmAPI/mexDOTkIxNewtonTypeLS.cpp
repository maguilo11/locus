/*
 * mexDOTkIxNewtonTypeLS.cpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexInexactNewtonTypeLS.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Line Search Newton Algorithm - Num. Diff. Hessian MEX API\n"
                        " should be used as follows: [Output] = mexDOTkIxNewtonTypeLS(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Line Search Newton Algorithm - Numerically Differentiated Hessian \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexInexactNewtonTypeLS newton(pInput_);
    newton.solve(pInput_, pOutput_);
}
