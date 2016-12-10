/*
 * TRROM_MxSpectralDecompositionTest.cpp
 *
 *  Created on: Nov 29, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxSingularValueDecomposition.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR SINGULAR VALUE DECOMPOSITION ALGORITHM\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 1 && nOutput == 3) )
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES ONE INPUT AND RETURNS THREE OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    std::tr1::shared_ptr<trrom::MxMatrix> A(new trrom::MxMatrix(pInput[0]));
    std::tr1::shared_ptr<trrom::Vector<double> > singular_values(new trrom::MxVector(1));
    std::tr1::shared_ptr<trrom::Matrix<double> > left_singular_vectors(new trrom::MxMatrix(1, 1));
    std::tr1::shared_ptr<trrom::Matrix<double> > right_singular_vectors(new trrom::MxMatrix(1, 1));

    trrom::MxSingularValueDecomposition svd;
    svd.solve(A, singular_values, left_singular_vectors, right_singular_vectors);

    // **** ASSERT SINGULAR VALUES ****
    msg.assign("singular values");
    const int num_singular_values = singular_values->size();
    trrom::MxVector vec_gold(num_singular_values);
    vec_gold[0] = 2.460504870018764; vec_gold[1] = 1.699628148275318; vec_gold[2] = 0.239123278256554;
    bool did_test_pass = trrom::mx::checkResults(vec_gold, *singular_values);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** ASSERT LEFT SINGULAR VECTORS ****
    msg.assign("left singular vectors");
    int num_rows = left_singular_vectors->getNumRows();
    int num_columns = left_singular_vectors->getNumCols();
    trrom::MxMatrix mat_gold(num_rows, num_columns);
    mat_gold(0,0) = -0.120000260381534; mat_gold(0,1) = -0.809712281592778; mat_gold(0,2) = 0.574426634607223;
    mat_gold(1,0) = 0.901752646908814; mat_gold(1,1) = 0.153122822484369; mat_gold(1,2) = 0.404222172854692;
    mat_gold(2,0) = -0.415261485453819; mat_gold(2,1) = 0.566497504206538; mat_gold(2,2) = 0.711785414592383;
    did_test_pass = trrom::mx::checkResults(mat_gold, *left_singular_vectors);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** ASSERT RIGHT SINGULAR VECTORS ****
    msg.assign("right singular vectors");
    num_rows = right_singular_vectors->getNumRows();
    num_columns = right_singular_vectors->getNumCols();
    mat_gold(0,0) = -0.415261485453819; mat_gold(0,1) = -0.566497504206539; mat_gold(0,2) = 0.711785414592383;
    mat_gold(1,0) = -0.901752646908814; mat_gold(1,1) = 0.153122822484370; mat_gold(1,2) = -0.404222172854692;
    mat_gold(2,0) = 0.120000260381534; mat_gold(2,1) = -0.809712281592779; mat_gold(2,2) = -0.574426634607224;
    did_test_pass = trrom::mx::checkResults(mat_gold, *right_singular_vectors);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** Set left singular vectors output ****
    num_rows = left_singular_vectors->getNumRows();
    num_columns = left_singular_vectors->getNumCols();
    pOutput[0] = mxCreateDoubleMatrix(num_rows, num_columns, mxREAL);
    trrom::MxMatrix & mx_left_singular_vectors = dynamic_cast<trrom::MxMatrix &>(*left_singular_vectors);
    trrom::mx::setMxArray(mx_left_singular_vectors.array(), pOutput[0]);

    // **** Set singular values output ****
    pOutput[1] = mxCreateDoubleMatrix(num_singular_values, 1, mxREAL);
    trrom::MxVector & mx_singular_values = dynamic_cast<trrom::MxVector &>(*singular_values);
    trrom::mx::setMxArray(mx_singular_values.array(), pOutput[1]);

    // **** Set right singular vectors output ****
    num_rows = right_singular_vectors->getNumRows();
    num_columns = right_singular_vectors->getNumCols();
    pOutput[2] = mxCreateDoubleMatrix(num_rows, num_columns, mxREAL);
    trrom::MxMatrix & mx_right_singular_vectors = dynamic_cast<trrom::MxMatrix &>(*right_singular_vectors);
    trrom::mx::setMxArray(mx_right_singular_vectors.array(), pOutput[2]);
}
