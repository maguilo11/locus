/*
 * TRROM_MxMatrixTest.cpp
 *
 *  Created on: Nov 22, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <cmath>
#include <string>

#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR TRROM::MATRIX\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 0 && nOutput == 0) )
    {
        std::string error("\nTEST DOES NOT TAKE INPUT AND OUTPUT AGUMENTS\n");
        mexErrMsgTxt(error.c_str());
    }

    int A_num_rows = 10;
    int A_num_columns = 5;
    trrom::MxMatrix A(A_num_rows, A_num_columns);

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
    trrom::MxMatrix gold(A_num_rows, A_num_columns, 10);
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
    int in_num_rows = 5;
    int in_num_columns = 10;
    int out_num_rows = 10;
    int out_num_columns = 10;
    trrom::MxMatrix IN1(in_num_rows, in_num_columns, 2);
    trrom::MxMatrix OUT1(out_num_rows, out_num_columns);
    A.gemm(false, false, 1., IN1, 0., OUT1);
    trrom::MxMatrix GOLD1(out_num_rows, out_num_rows, 1500);
    did_test_pass = trrom::mx::checkResults(GOLD1, OUT1);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 12: gemm - A=T; B=NT
    msg.assign("gemm_A_T_and_B_NT");
    in_num_rows = 10;
    in_num_columns = 5;
    out_num_rows = 5;
    out_num_columns = 5;
    trrom::MxMatrix IN2(in_num_rows, in_num_columns, 2);
    trrom::MxMatrix OUT2(out_num_rows, out_num_columns);
    A.gemm(true, false, 1., IN2, 0., OUT2);
    trrom::MxMatrix GOLD2(out_num_rows, out_num_columns, 3000);
    did_test_pass = trrom::mx::checkResults(GOLD2, OUT2);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 13: gemm - A=NT; B=T
    msg.assign("gemm_A_NT_and_B_T");
    in_num_rows = 10;
    in_num_columns = 5;
    out_num_rows = 10;
    out_num_columns = 10;
    A.fill(10);
    trrom::MxMatrix IN3(in_num_rows, in_num_columns, 2);
    trrom::MxMatrix OUT3(out_num_rows, out_num_rows);
    A.gemm(false, true, 1., IN3, 0., OUT3);
    trrom::MxMatrix GOLD3(out_num_rows, out_num_columns, 100);
    did_test_pass = trrom::mx::checkResults(GOLD3, OUT3);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 14: gemm - A=T; B=T
    msg.assign("gemm_A_T_and_B_T");
    in_num_rows = 5;
    in_num_columns = 10;
    out_num_rows = 5;
    out_num_columns = 5;
    trrom::MxMatrix IN4(in_num_rows, in_num_columns, 2);
    trrom::MxMatrix OUT4(out_num_rows, out_num_rows);
    A.gemm(true, true, 1., IN4, 0., OUT4);
    trrom::MxMatrix GOLD4(out_num_rows, out_num_columns, 200);
    did_test_pass = trrom::mx::checkResults(GOLD4, OUT4);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 15: replaceGlobalValue
    msg.assign("replaceGlobalValue");
    did_test_pass = A(1,1) == 10;
    trrom::mx::assert_test(msg, did_test_pass);
    A.replaceGlobalValue(1, 1, 33);
    did_test_pass = A(1,1) == 33;
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 16: create
    msg.assign("create_default");
    std::tr1::shared_ptr< trrom::Matrix<double> > copy = A.create();
    did_test_pass = copy->getNumCols() == A.getNumCols();
    trrom::mx::assert_test(msg, did_test_pass);
    did_test_pass = copy->getNumRows() == A.getNumRows();
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 17: create
    msg.assign("create_non_default");
    copy = A.create(6,7);
    did_test_pass = copy->getNumCols() == 7;
    trrom::mx::assert_test(msg, did_test_pass);
    did_test_pass = copy->getNumRows() == 6;
    trrom::mx::assert_test(msg, did_test_pass);
}
