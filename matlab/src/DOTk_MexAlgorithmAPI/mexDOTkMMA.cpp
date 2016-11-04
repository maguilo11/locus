/*
 * mexDOTkMMA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexMMA.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Method of Moving Asymptotes (MMA) Algorithm MEX\n"
                        " API should be used as follows: [Output] = mexDOTkMMA(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Method of Moving Asymptotes (MMA) Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexMMA algorithm(pInput_);
    algorithm.solve(pInput_, pOutput_);
}
