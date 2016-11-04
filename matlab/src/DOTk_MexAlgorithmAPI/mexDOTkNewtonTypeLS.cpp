/*
 * mexDOTkNewtonTypeLS.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexNewtonTypeLS.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Line Search Newton Algorithm MEX API should\n"
                        " be used as follows: [Output] = mexDOTkNewtonTypeLS(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Line Search Newton Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexNewtonTypeLS newton(pInput_);
    newton.solve(pInput_, pOutput_);
}
