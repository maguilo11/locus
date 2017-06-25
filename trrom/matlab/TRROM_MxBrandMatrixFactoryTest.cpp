/*
 * TRROM_MxBrandMatrixFactoryTest.cpp
 *
 *  Created on: Dec 6, 2016
 *      Author: maguilo
 */

#include <string>

#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxBrandMatrixFactory.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR BRAND MATRIX FACTORY\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 0 && nOutput == 0) )
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES NO INPUTS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    trrom::MxBrandMatrixFactory factory;

    // **** TEST 1: buildMatrixUbar ****
    msg.assign("buildMatrixUbar");
    // Construct Matrix A
    int num_rows = 3;
    int num_columns = 4;
    std::shared_ptr<trrom::Matrix<double> > A = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*A);
    // Construct Matrix B
    num_rows = 3;
    num_columns = 6;
    std::shared_ptr<trrom::Matrix<double> > B = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*B);
    // Construct Matrix C
    num_rows = 10;
    num_columns = 2;
    std::shared_ptr<trrom::Matrix<double> > C = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*C);
    std::shared_ptr<trrom::Matrix<double> > D;

    factory.buildMatrixUbar(A, B, C, D);

    // ASSERT TEST 1 RESULTS
    num_rows = 3;
    num_columns = 2;
    std::shared_ptr<trrom::Matrix<double> > gold = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    (*gold)(0,0) = 505; (*gold)(1,0) = 560; (*gold)(2,0) = 615;
    (*gold)(0,1) = 1235; (*gold)(1,1) = 1390; (*gold)(2,1) = 1545;
    bool did_test_pass = trrom::mx::checkResults(*gold, *D);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 2: buildMatrixVbar ****
    msg.assign("buildMatrixVbar");
    num_rows = 5;
    num_columns = 6;
    A = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*A);
    num_rows = 10;
    num_columns = 2;
    B = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*B);

    factory.buildMatrixVbar(A, B, D);

    // ASSERT TEST 2 RESULTS
    num_rows = 9;
    num_columns = 2;
    gold = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    (*gold)(0,0) = 371; (*gold)(0,1) = 1181;
    (*gold)(1,0) = 392; (*gold)(1,1) = 1262;
    (*gold)(2,0) = 413; (*gold)(2,1) = 1343;
    (*gold)(3,0) = 434; (*gold)(3,1) = 1424;
    (*gold)(4,0) = 455; (*gold)(4,1) = 1505;
    (*gold)(5,0) = 7; (*gold)(5,1) = 17;
    (*gold)(6,0) = 8; (*gold)(6,1) = 18;
    (*gold)(7,0) = 9; (*gold)(7,1) = 19;
    (*gold)(8,0) = 10; (*gold)(8,1) = 20;
    did_test_pass = trrom::mx::checkResults(*gold, *D);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 3: buildMatrixK ****
    msg.assign("buildMatrixK");
    num_rows = 4;
    num_columns = 5;
    A = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*A);
    num_rows = 5;
    num_columns = 5;
    B = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    trrom::mx::fill(*B);
    const int num_singular_values = 4;
    std::shared_ptr<trrom::Vector<double> > sigma = std::make_shared<trrom::MxVector>(num_singular_values);
    trrom::mx::fill(*sigma);

    factory.buildMatrixK(sigma, A, B, D);

    // ASSERT TEST 3 RESULTS
    num_rows = 9;
    num_columns = 9;
    gold = std::make_shared<trrom::MxMatrix>(num_rows, num_columns);
    // BLOCK 11
    (*gold)(0,0)=1; (*gold)(1,1)=2; (*gold)(2,2)=3; (*gold)(3,3)=4;
    // BLOCK 12
    (*gold)(0,4)=1; (*gold)(1,4)=2; (*gold)(2,4)=3; (*gold)(3,4)=4;
    (*gold)(0,5)=5; (*gold)(1,5)=6; (*gold)(2,5)=7; (*gold)(3,5)=8;
    (*gold)(0,6)=9; (*gold)(1,6)=10; (*gold)(2,6)=11; (*gold)(3,6)=12;
    (*gold)(0,7)=13; (*gold)(1,7)=14; (*gold)(2,7)=15; (*gold)(3,7)=16;
    (*gold)(0,8)=17; (*gold)(1,8)=18; (*gold)(2,8)=19; (*gold)(3,8)=20;
    // BLOCK 22
    (*gold)(4,4)=1; (*gold)(5,4)=2; (*gold)(6,4)=3; (*gold)(7,4)=4; (*gold)(8,4)=5;
    (*gold)(4,5)=6; (*gold)(5,5)=7; (*gold)(6,5)=8; (*gold)(7,5)=9; (*gold)(8,5)=10;
    (*gold)(4,6)=11; (*gold)(5,6)=12; (*gold)(6,6)=13; (*gold)(7,6)=14; (*gold)(8,6)=15;
    (*gold)(4,7)=16; (*gold)(5,7)=17; (*gold)(6,7)=18; (*gold)(7,7)=19; (*gold)(8,7)=20;
    (*gold)(4,8)=21; (*gold)(5,8)=22; (*gold)(6,8)=23; (*gold)(7,8)=24; (*gold)(8,8)=25;
    did_test_pass = trrom::mx::checkResults(*gold, *D);
    trrom::mx::assert_test(msg, did_test_pass);
}

