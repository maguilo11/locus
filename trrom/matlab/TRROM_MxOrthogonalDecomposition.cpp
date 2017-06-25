/*
 * TRROM_MxOrthogonalDecomposition.cpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxOrthogonalDecomposition.hpp"

namespace trrom
{

MxOrthogonalDecomposition::MxOrthogonalDecomposition() :
        m_PermutationData()
{
}

MxOrthogonalDecomposition::~MxOrthogonalDecomposition()
{
}

trrom::types::ortho_factorization_t MxOrthogonalDecomposition::type() const
{
    return (trrom::types::MATLAB_QR);
}

void MxOrthogonalDecomposition::factorize(const std::shared_ptr<trrom::Matrix<double> > & input_,
                                          std::shared_ptr<trrom::Matrix<double> > & Q_,
                                          std::shared_ptr<trrom::Matrix<double> > & R_)
{
    const trrom::MxMatrix & input = dynamic_cast<const trrom::MxMatrix &>(*input_);

    // Perform orthogonal-triangular decomposition with column pivoting
    mxArray* mex_output[3];
    mxArray* mx_economy_size_flag = mxCreateDoubleScalar(0);
    mxArray* mx_input_data = const_cast<mxArray*>(input.array());
    mxArray* mex_inputs[2] = { mx_input_data, mx_economy_size_flag };
    mxArray* error = mexCallMATLABWithTrapWithObject(3, mex_output, 2, mex_inputs, "qr");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling qr.\n";
    trrom::mx::handleException(error, msg.str());

    // Delete memory allocated by mxCreateDoubleScalar(0)
    mxDestroyArray(mx_economy_size_flag);

    // Copy m-by-n unitary matrix Q output from Matlab into TRROM matrix data structure
    int num_rows = mxGetM(mex_output[0]);
    int num_columns = mxGetN(mex_output[0]);
    Q_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & Q_matrix = dynamic_cast<trrom::MxMatrix &>(*Q_);
    Q_matrix.setMxArray(mex_output[0]);

    // Copy n-by-n upper triangular matrix R output from Matlab into TRROM matrix data structure
    num_rows = mxGetM(mex_output[1]);
    num_columns = mxGetN(mex_output[1]);
    R_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & R_matrix = dynamic_cast<trrom::MxMatrix &>(*R_);
    R_matrix.setMxArray(mex_output[1]);

    // Copy permutation matrix output from Matlab into TRROM matrix data structure
    num_rows = mxGetM(mex_output[2]);
    num_columns = mxGetN(mex_output[2]);
    m_PermutationData.reset(new trrom::MxMatrix(num_rows, num_columns));
    m_PermutationData->setMxArray(mex_output[2]);
}

const trrom::MxMatrix & MxOrthogonalDecomposition::getPermutationData() const
{
    return (*m_PermutationData);
}

}
