/*
 * TRROM_MxOrthogonalDecompositionTest.cpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxOrthogonalDecomposition.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR ORTHOGONAL DECOMPOSITION ALGORITHM\n");
    mexPrintf("%s", msg.c_str());
    if( !(nInput == 1 && nOutput == 3) )
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES ONE INPUT AND RETURNS THREE OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    std::tr1::shared_ptr<trrom::Matrix<double> > Q;
    std::tr1::shared_ptr<trrom::Matrix<double> > R;
    std::tr1::shared_ptr<trrom::MxMatrix> A(new trrom::MxMatrix(pInput[0]));

    trrom::MxOrthogonalDecomposition qr;
    qr.factorize(A, Q, R);

    // **** ASSERT UNITARY MATRIX Q ****
    msg.assign("unitary matrix Q");
    int num_rows = Q->getNumRows();
    int num_columns = Q->getNumCols();
    trrom::MxMatrix Q_gold(num_rows, num_columns);
    Q_gold(0,0) = -0.109108945117996; Q_gold(0,1) = 0.354649682807595; Q_gold(0,2) = 0.782508045057500;
    Q_gold(1,0) = -0.327326835353989; Q_gold(1,1) = -0.591082804679325; Q_gold(1,2) = 0.541736338885961;
    Q_gold(2,0) = -0.545544725589981; Q_gold(2,1) = 0.669893845303236; Q_gold(2,2) = -0.060192926542884;
    Q_gold(3,0) = -0.763762615825974; Q_gold(3,1) = -0.275838642183685; Q_gold(3,2) = -0.300964632714423;
    bool did_test_pass = trrom::mx::checkResults(Q_gold, *Q);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** ASSERT UPPER TRIANGULAR MATRIX R ****
    msg.assign("upper triangular matrix R");
    num_rows = R->getNumRows();
    num_columns = R->getNumCols();
    trrom::MxMatrix R_gold(num_rows, num_columns);
    R_gold(0,0) = -9.165151389911680; R_gold(0,1) = -2.618614682831909; R_gold(0,2) = -0.436435780471985;
    R_gold(1,1) = -3.625307868699863; R_gold(1,2) = -1.891464974973841;
    R_gold(2,2) = -0.481543412343077;
    did_test_pass = trrom::mx::checkResults(R_gold, *R);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** ASSERT PERMUTATION MATRIX E ****
    msg.assign("permutation matrix E");
    num_rows = qr.getPermutationData().getNumRows();
    num_columns = qr.getPermutationData().getNumCols();
    trrom::MxMatrix E_gold(num_rows, num_columns);
    E_gold(0,0) = 3; E_gold(0,1) = 2; E_gold(0,2) = 1;
    did_test_pass = trrom::mx::checkResults(E_gold, qr.getPermutationData());
    trrom::mx::assert_test(msg, did_test_pass);

    // **** Set output for m-by-n unitary matrix Q ****
    num_rows = Q->getNumRows();
    num_columns = Q->getNumCols();
    pOutput[0] = mxCreateDoubleMatrix(num_rows, num_columns, mxREAL);
    trrom::MxMatrix & mx_unitary_matrix = dynamic_cast<trrom::MxMatrix &>(*Q);
    trrom::mx::setMxArray(mx_unitary_matrix.array(), pOutput[0]);

    // **** Set output for n-by-n upper triangular matrix R ****
    num_rows = R->getNumRows();
    num_columns = R->getNumCols();
    pOutput[1] = mxCreateDoubleMatrix(num_rows, num_columns, mxREAL);
    trrom::MxMatrix & mx_upper_triangular_matrix = dynamic_cast<trrom::MxMatrix &>(*R);
    trrom::mx::setMxArray(mx_upper_triangular_matrix.array(), pOutput[1]);

    // **** Set output for permutation matrix E ****
    num_rows = qr.getPermutationData().getNumRows();
    num_columns = qr.getPermutationData().getNumCols();
    pOutput[2] = mxCreateDoubleMatrix(num_rows, num_columns, mxREAL);
    trrom::mx::setMxArray(qr.getPermutationData().array(), pOutput[2]);
}
