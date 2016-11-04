/*
 * mexDOTkQuasiNewton.cpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexQuasiNewton.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) QuasiNewton Algorithm MEX API should be\n"
                        " used as follows: [Output] = mexDOTkQuasiNewton(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Quasi-Newton Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexQuasiNewton quasi_newton(pInput_);
    quasi_newton.solve(pInput_, pOutput_);
}
