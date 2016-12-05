/*
 * TRROM_MxVectorTest.cpp
 *
 *  Created on: Nov 22, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <cmath>
#include <string>

#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX VECTOR INTERFACE\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 0 && nOutput == 0) )
    {
        std::string error("\nTEST DOES NOT TAKE INPUT AND OUTPUT AGUMENTS\n");
        mexErrMsgTxt(error.c_str());
    }

    int length = 10;
    trrom::MxVector x(length);

    // TEST 1: size
    msg.assign("size");
    bool did_test_pass = x.size() == 10;
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 2: fill
    msg.assign("fill");
    x.fill(10);
    trrom::MxVector gold(length, 10);
    did_test_pass = trrom::mx::checkResults(gold, x);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 3: scale
    msg.assign("scale");
    x.scale(3);
    gold.fill(30);
    did_test_pass = trrom::mx::checkResults(gold, x);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 4: elementWiseMultiplication
    msg.assign("elementWiseMultiplication");
    x.elementWiseMultiplication(gold);
    gold.fill(900);
    did_test_pass = trrom::mx::checkResults(gold, x);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 5: update: copy
    msg.assign("copy");
    gold.fill(23);
    x.update(1., gold, 0.);
    did_test_pass = trrom::mx::checkResults(gold, x);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 6: update: axpy
    msg.assign("axpy");
    x.update(1., gold, 2.);
    gold.fill(69);
    did_test_pass = trrom::mx::checkResults(gold, x);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 7: max
    x[3] = 123;
    int index = 0;
    double value = x.max(index);
    msg.assign("max_value");
    did_test_pass = (value == 123);
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("max_index");
    did_test_pass = (index == 3);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 8: min
    x[7] = 1;
    index = 0;
    value = x.min(index);
    msg.assign("min_value");
    did_test_pass = (value == 1);
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("min_index");
    did_test_pass = (index == 7);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 9: modulus
    msg.assign("modulus");
    x.fill(-11);
    x.modulus();
    gold.fill(11);
    did_test_pass = trrom::mx::checkResults(gold, x);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 10: sum
    msg.assign("sum");
    value = x.sum();
    did_test_pass = (value == 110);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 11: dot
    msg.assign("dot");
    value = x.dot(gold);
    did_test_pass = (value == 1210);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 12: norm
    msg.assign("norm");
    value = x.norm();
    double epsilon = std::abs(value - 34.785054261);
    did_test_pass = (epsilon < 1e-6);
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 13: default create
    msg.assign("create_default");
    std::tr1::shared_ptr<trrom::Vector<double> > y = x.create();
    did_test_pass = y->size() == x.size();
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 14: create
    msg.assign("create_new_length");
    int new_length = 22;
    y = x.create(new_length);
    did_test_pass = y->size() == new_length;
    trrom::mx::assert_test(msg, did_test_pass);

    // TEST 15: constructor
    msg.assign("constructor");
    length = 30;
    mxArray* array = mxCreateDoubleMatrix(1, length, mxREAL);
    trrom::MxVector z(array);
    did_test_pass = z.size() == length;
    trrom::mx::assert_test(msg, did_test_pass);
    mxDestroyArray(array);

    // TEST 16: setMxArray
    msg.assign("setMxArray");
    z.fill(10);
    length = z.size();
    trrom::MxVector vector(length);
    vector.setMxArray(z.array());
    did_test_pass = trrom::mx::checkResults(z, vector);
    trrom::mx::assert_test(msg, did_test_pass);
}
