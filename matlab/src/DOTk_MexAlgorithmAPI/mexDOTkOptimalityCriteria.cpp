/*
 * mexDOTkOptimalityCriteria.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexOptimalityCriteria.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) Optimality Criteria (OC) Algorithm MEX API should\n"
                        " be used as follows: [Output] = mexDOTkOptimalityCriteria(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Optimality Criteria (OC) Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexOptimalityCriteria optimality_criteria(pInput_);
    optimality_criteria.solve(pInput_, pOutput_);
}
