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

void MxOrthogonalDecomposition::factorize(const trrom::Matrix<double> & input_,
                                          trrom::Matrix<double> & Q_,
                                          trrom::Matrix<double> & R_)
{
    const trrom::MxMatrix & input = dynamic_cast<const trrom::MxMatrix &>(input_);
    const mxArray* input_mex_array = input.array();

    // Perform orthogonal-triangular decomposition
    mxArray* mex_inputs[2];
    mex_inputs[0] = const_cast<mxArray*>(input_mex_array);
    mex_inputs[1] = mxCreateDoubleScalar(0); // flag: denotes economy-size decomposition
    mxArray* mex_output[3];
    mxArray* error = mexCallMATLABWithTrapWithObject(3, mex_output, 2, mex_inputs, "qr");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling qr.\n";
    trrom::mx::handleException(error, msg.str());

    // Copy m-by-n unitary matrix Q output from Matlab into TRROM matrix data structure
    trrom::MxMatrix & Q_matrix = dynamic_cast<trrom::MxMatrix &>(Q_);
    trrom::mx::setMxArray(mex_output[0], Q_matrix.array());

    // Copy n-by-n upper triangular matrix R output from Matlab into TRROM matrix data structure
    trrom::MxMatrix & R_matrix = dynamic_cast<trrom::MxMatrix &>(R_);
    trrom::mx::setMxArray(mex_output[1], R_matrix.array());

    // Copy permutation matrix output from Matlab into TRROM matrix data structure
    int num_rows = mxGetM(mex_output[2]);
    int num_columns = mxGetN(mex_output[2]);
    m_PermutationData.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::mx::setMxArray(mex_output[2], m_PermutationData->array());

    // Delete memory allocated by call to mxCreateDoubleScalar(0)
    mxDestroyArray(mex_inputs[1]);
}

const trrom::MxMatrix & MxOrthogonalDecomposition::getPermutationData() const
{
    return (*m_PermutationData);
}

}
