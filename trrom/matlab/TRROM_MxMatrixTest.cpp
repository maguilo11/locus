/*
 * TRROM_MxMatrixTest.cpp
 *
 *  Created on: Nov 22, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <cmath>
#include <string>

#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxDenseMatrix.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX MATRIX INTERFACE\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 0 && nOutput == 0) )
    {
        std::string error("\nTEST DOES NOT TAKE INPUT AND OUTPUT AGUMENTS\n");
        mexErrMsgTxt(error.c_str());
    }

    int A_num_rows = 10;
    int A_num_columns = 5;
    trrom::MxDenseMatrix A(A_num_rows, A_num_columns);

    // TEST 1: getNumRows
    msg.assign("getNumRows");
    bool did_test_pass = A.getNumRows() == 10;
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 2: getNumCols
    msg.assign("getNumCols");
    did_test_pass = A.getNumCols() == 5;
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 3: fill
    msg.assign("fill");
    A.fill(10);
    trrom::MxDenseMatrix gold(A_num_rows, A_num_columns, 10);
    did_test_pass = trrom::mx::checkResults(gold, A);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 4: scale
    msg.assign("scale");
    A.scale(2);
    gold.fill(20);
    did_test_pass = trrom::mx::checkResults(gold, A);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 5: copy
    msg.assign("copy");
    gold.fill(30);
    A.update(1., gold, 0.);
    did_test_pass = trrom::mx::checkResults(gold, A);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 6: axpy
    msg.assign("axpy");
    A.update(3., gold, 2.);
    gold.fill(150);
    did_test_pass = trrom::mx::checkResults(gold, A);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 7: gemv - transpose
    msg.assign("gemv_transpose");
    trrom::MxVector in2(A_num_rows, 1);
    trrom::MxVector out2(A_num_columns);
    A.gemv(true, 1., in2, 0., out2);
    trrom::MxVector gold3(A_num_columns, 1500);
    did_test_pass = trrom::mx::checkResults(gold3, out2);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 8: gemv - transpose
    msg.assign("gemv_transpose_scaled");
    out2.fill(1);
    A.gemv(true, 2., in2, 2., out2);
    gold3.fill(3002);
    did_test_pass = trrom::mx::checkResults(gold3, out2);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 9: gemv - not transpose
    msg.assign("gemv_not_transpose");
    trrom::MxVector out1(A_num_rows);
    trrom::MxVector in1(A_num_columns, 1);
    A.gemv(false, 1., in1, 0., out1);
    trrom::MxVector gold2(A_num_rows, 750);
    did_test_pass = trrom::mx::checkResults(gold2, out1);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 10: gemv - not transpose
    msg.assign("gemv_not_transpose_scaled");
    out1.fill(1);
    A.gemv(false, 2., in1, 2., out1);
    gold2.fill(1502);
    did_test_pass = trrom::mx::checkResults(gold2, out1);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 11: gemm - A=B=not transpose
    msg.assign("gemm_A_NT_and_B_NT");
    int B_num_rows = 5;
    int B_num_columns = 10;
    trrom::MxDenseMatrix IN1(B_num_rows, B_num_columns, 2);
    trrom::MxDenseMatrix OUT1(A_num_rows, B_num_columns);
    A.gemm(false, false, 1., IN1, 0., OUT1);
    trrom::MxDenseMatrix GOLD1(A_num_rows, B_num_columns, 1500);
    did_test_pass = trrom::mx::checkResults(GOLD1, OUT1);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 11: gemm - A=T; B=NT
    msg.assign("gemm_A_T_and_B_NT");
    trrom::MxDenseMatrix IN2(10, 5, 2);
    trrom::MxDenseMatrix OUT2(5, 5);
    A.gemm(true, false, 1., IN2, 0., OUT2);
    trrom::MxDenseMatrix GOLD2(5, 5, 3000);
    did_test_pass = trrom::mx::checkResults(GOLD2, OUT2);
    trrom::mx::assert_test(msg, did_test_pass);
}
