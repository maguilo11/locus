/*
 * mexDOTkIxLinMoreTrustRegion.cpp
 *
 *  Created on: Sep 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>

#include "DOTk_MexIxTrustRegionLinMore.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Inexact Lin-More Trust Region Newton Algorithm - MEX API\n"
                        " should be used as follows: [Output] = mexDOTkIxLinMoreTrustRegion(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Inexact Lin-More Trust Region Newton Algorithm (Numerically Differentiated Hessian) \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexIxTrustRegionLinMore algorithm(pInput_);
    algorithm.solve(pInput_, pOutput_);
}
