/*
 * mexDOTkIxSQPTypeTR.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexInexactTrustRegionSQP.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Inexact Trust Region Sequential Quadratic Programming (SQP)\n"
                        " Algorithm MEX API should be used as follows: [Output] = mexDOTkIxSQPTypeTR(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Inexact Trust Region Sequential Quadratic Programming (SQP) Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexInexactTrustRegionSQP sqp(pInput_);
    sqp.solve(pInput_, pOutput_);
}
