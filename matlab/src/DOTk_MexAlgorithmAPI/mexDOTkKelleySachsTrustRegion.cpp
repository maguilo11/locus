/*
 * mexDOTkKelleySachsTrustRegion.cpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexTrustRegionKelleySachs.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Kelley-Sachs Trust Region Newton Algorithm MEX API should\n"
                        " be used as follows: [Output] = mexDOTkKelleySachsTrustRegion(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Kelley-Sachs Trust Region Newton Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexTrustRegionKelleySachs algorithm(pInput_);
    algorithm.solve(pInput_, pOutput_);
}
