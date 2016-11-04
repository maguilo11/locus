/*
 * mexDOTkGradientProjection.cpp
 *
 *  Created on: Oct 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexGradientProjection.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Gradient Projection Algorithm MEX API should be\n"
                        " used as follows: [Output] = mexDOTkGradientProjection(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Gradient Projection Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexGradientProjection algorithm(pInput_);
    algorithm.solve(pInput_, pOutput_);
}
