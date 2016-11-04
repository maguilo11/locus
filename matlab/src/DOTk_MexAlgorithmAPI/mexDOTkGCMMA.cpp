/*
 * mexDOTkGCMMA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexGCMMA.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Globally Convergent Method of Moving Asymptotes (GCMMA)\n"
                        " Algorithm MEX API should be used as follows: [Output] = mexDOTkGCMMA(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Globally Convergent Method of Moving Asymptotes (GCMMA) Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexGCMMA algorithm(pInput_);
    algorithm.solve(pInput_, pOutput_);
}
