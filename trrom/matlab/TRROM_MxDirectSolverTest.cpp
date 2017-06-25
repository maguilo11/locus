/*
 * TRROM_MxDirectSolverTest.cpp
 *
 *  Created on: Nov 28, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <string>
#include <memory>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxDirectSolver.hpp"

namespace trrom
{

namespace mx
{

inline void setSolverTestData(std::shared_ptr<MxMatrix> & matrix_,
                              std::shared_ptr<MxVector> & rhs_)
{
    // Set matrix data
    const int num_rows = 5;
    const int num_columns = 5;
    matrix_ = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    (*matrix_)(0, 0) = 1;
    (*matrix_)(0, 1) = 5;
    (*matrix_)(0, 2) = 0;
    (*matrix_)(0, 3) = 0;
    (*matrix_)(0, 4) = 0;
    (*matrix_)(1, 0) = 0;
    (*matrix_)(1, 1) = 2;
    (*matrix_)(1, 2) = 8;
    (*matrix_)(1, 3) = 0;
    (*matrix_)(1, 4) = 0;
    (*matrix_)(2, 0) = 0;
    (*matrix_)(2, 1) = 0;
    (*matrix_)(2, 2) = 3;
    (*matrix_)(2, 3) = 9;
    (*matrix_)(2, 4) = 0;
    (*matrix_)(3, 0) = 0;
    (*matrix_)(3, 1) = 0;
    (*matrix_)(3, 2) = 0;
    (*matrix_)(3, 3) = 4;
    (*matrix_)(3, 4) = 10;
    (*matrix_)(4, 0) = 0;
    (*matrix_)(4, 1) = 0;
    (*matrix_)(4, 2) = 0;
    (*matrix_)(4, 3) = 0;
    (*matrix_)(4, 4) = 5;

    // Set right hand side vector data
    rhs_ = std::make_shared<trrom::MxVector>(num_columns);
    (*rhs_)[0] = 1;
    (*rhs_)[1] = 2;
    (*rhs_)[2] = 3;
    (*rhs_)[3] = 4;
    (*rhs_)[4] = 5;
}

}

}

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR DIRECT SOLVER\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 0 && nOutput == 0) )
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES NO INPUTS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    std::shared_ptr<trrom::MxMatrix> A;
    std::shared_ptr<trrom::MxVector> rhs;
    trrom::mx::setSolverTestData(A, rhs);
    const int length = rhs->size();
    trrom::MxVector lhs(length);

    msg.assign("solve - rhs transpose");
    trrom::MxDirectSolver solver;
    solver.solve(*A, *rhs, lhs);
    trrom::MxVector gold(length);
    gold[0] = 106; gold[1] = -21; gold[2] = 5.5; gold[3] = -1.5; gold[4] = 1;
    bool did_test_pass = trrom::mx::checkResults(gold, lhs);
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("solve - rhs not transpose");
    mxArray* mx_rhs = mxCreateDoubleMatrix(length, 1, mxREAL);
    rhs = std::make_shared<trrom::MxVector>(mx_rhs);
    trrom::mx::fill(*rhs);
    solver.solve(*A, *rhs, lhs);
    did_test_pass = trrom::mx::checkResults(gold, lhs);
    trrom::mx::assert_test(msg, did_test_pass);
    mxDestroyArray(mx_rhs);
}
