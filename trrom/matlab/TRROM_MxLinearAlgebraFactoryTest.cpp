/*
 * TRROM_MxLinearAlgebraFactoryTest.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxLinearAlgebraFactory.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR LINEAR ALGEBRA FACTORY\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 0 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES NO INPUTS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    trrom::MxLinearAlgebraFactory factory;

    // **** TEST 1: reshape ****
    msg.assign("reshape");
    const int length = 25;
    std::tr1::shared_ptr<trrom::Vector<double> > vector(new trrom::MxVector(length));
    trrom::mx::fill(*vector);

    const int num_rows = 5;
    const int num_columns = 5;
    std::tr1::shared_ptr<trrom::Matrix<double> > matrix;
    factory.reshape(num_rows, num_columns, vector, matrix);

    // ASSERT TEST 1 RESULTS
    std::tr1::shared_ptr<trrom::Matrix<double> > gold = matrix->create();
    (*gold)(0,0) = 1; (*gold)(0,1) = 6 ; (*gold)(0,2) = 11; (*gold)(0,3) = 16; (*gold)(0,4) = 21;
    (*gold)(1,0) = 2; (*gold)(1,1) = 7 ; (*gold)(1,2) = 12; (*gold)(1,3) = 17; (*gold)(1,4) = 22;
    (*gold)(2,0) = 3; (*gold)(2,1) = 8 ; (*gold)(2,2) = 13; (*gold)(2,3) = 18; (*gold)(2,4) = 23;
    (*gold)(3,0) = 4; (*gold)(3,1) = 9 ; (*gold)(3,2) = 14; (*gold)(3,3) = 19; (*gold)(3,4) = 24;
    (*gold)(4,0) = 5; (*gold)(4,1) = 10; (*gold)(4,2) = 15; (*gold)(4,3) = 20; (*gold)(4,4) = 25;
    bool did_test_pass = trrom::mx::checkResults(*gold, *matrix);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 2: buildVector ****
    std::tr1::shared_ptr<trrom::Vector<double> > x;
    msg.assign("build vector: count = 0");
    did_test_pass = x.use_count() == 0;
    trrom::mx::assert_test(msg, did_test_pass);
    factory.buildLocalVector(length, x);
    msg.assign("build vector: count = 1");
    did_test_pass = x.use_count() == 1;
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("build vector: size");
    did_test_pass = x->size() == 25u;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 3: buildMatrix ****
    std::tr1::shared_ptr<trrom::Matrix<double> > A;
    msg.assign("build matrix: count = 0");
    did_test_pass = A.use_count() == 0;
    trrom::mx::assert_test(msg, did_test_pass);
    factory.buildLocalMatrix(num_rows, num_columns, A);
    msg.assign("build matrix: count = 1");
    did_test_pass = A.use_count() == 1;
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("build matrix: number of rows");
    did_test_pass = A->getNumRows() == 5u;
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("build matrix: number of columns");
    did_test_pass = A->getNumCols() == 5u;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST 4: buildMultiVector ****
    std::tr1::shared_ptr<trrom::Matrix<double> > B;
    msg.assign("build multivector: count = 0");
    did_test_pass = B.use_count() == 0;
    trrom::mx::assert_test(msg, did_test_pass);
    factory.buildMultiVector(num_columns, x, B);
    msg.assign("build multivector: count = 1");
    did_test_pass = B.use_count() == 1;
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("build multivector: degrees of freedom");
    did_test_pass = B->getNumRows() == 25u;
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("build multivector: number of vectors");
    did_test_pass = B->getNumCols() == 5u;
    trrom::mx::assert_test(msg, did_test_pass);
}
