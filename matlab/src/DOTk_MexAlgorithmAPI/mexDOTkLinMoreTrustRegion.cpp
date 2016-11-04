/*
 * DOTk_MexTrustRegionLinMore.cpp
 *
 *  Created on: Sep 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexTrustRegionLinMore.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Lin-More Trust Region Newton Algorithm MEX API should\n"
                        " be used as follows: [Output] = mexDOTkLinMoreTrustRegion(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Lin-More Trust Region Newton Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexTrustRegionLinMore algorithm(pInput_);
    algorithm.solve(pInput_, pOutput_);
}
