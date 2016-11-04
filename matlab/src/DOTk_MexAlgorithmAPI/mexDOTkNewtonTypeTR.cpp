/*
 * mexDOTkNewtonTypeTR.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexNewtonTypeTR.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Trust Region Newton Algorithm MEX API should\n"
                        " be used as follows: [Output] = mexDOTkNewtonTypeTR(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Trust Region Newton Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexNewtonTypeTR newton(pInput_);
    newton.solve(pInput_, pOutput_);
}
