/*
 * TRROM_MxDirectSolverTest.cpp
 *
 *  Created on: Nov 28, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxDirectSolver.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX DIRECT SOLVER INTERFACE\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 2 && nOutput == 1) )
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES TWO INPUTS AND RETURNS ONE OUTPUT.\n");
        mexErrMsgTxt(error.c_str());
    }

    trrom::MxMatrix A(pInput[0]);
    trrom::MxVector rhs(pInput[1]);
    const int length = rhs.size();
    trrom::MxVector lhs(length);

    msg.assign("solve");
    trrom::MxDirectSolver solver;
    solver.solve(A, rhs, lhs);
    trrom::MxVector gold(length);
    gold[0] = 1; gold[1] = 2; gold[2] = 3; gold[3] = 4; gold[4] = 5;
    bool did_test_pass = trrom::mx::checkResults(gold, lhs);
    trrom::mx::assert_test(msg, did_test_pass);

    pOutput[0] = mxCreateDoubleMatrix(length, 1, mxREAL);
    trrom::mx::setMxArray(lhs.array(), pOutput[0]);
}
