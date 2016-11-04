/*
 * mexDOTkNonLinearCG.cpp
 *
 *  Created on: Mar 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexNonlinearCG.hpp"

void mexFunction(int nOutput_, mxArray* pOutput_[], int nInput_, const mxArray* pInput_[])
{
    if((nInput_ == 2 && nOutput_ == 1) == false)
    {
        std::string msg(" Design Optimization Toolkit (DOTk) NonLinearCG Algorithm MEX API should be\n"
                        " used as follows: [Output] = mexDOTkNonLinearCG(AlgorithmOptions, Operators) \n");
        mexErrMsgTxt(msg.c_str());
    }

    std::string msg(" DOTk Nonlinear Conjugate Gradient Algorithm \n");
    mexPrintf(msg.c_str());

    dotk::DOTk_MexNonlinearCG nonlinear_cg(pInput_);
    nonlinear_cg.solve(pInput_, pOutput_);
}
