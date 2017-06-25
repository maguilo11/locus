/*
 * TRROM_MxSingularValueDecomposition.cpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxSingularValueDecomposition.hpp"

namespace trrom
{

MxSingularValueDecomposition::MxSingularValueDecomposition()
{
}

MxSingularValueDecomposition::~MxSingularValueDecomposition()
{
}

void MxSingularValueDecomposition::solve(const std::shared_ptr<trrom::Matrix<double> > & data_,
                                         std::shared_ptr<trrom::Vector<double> > & singular_values_,
                                         std::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                                         std::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    const trrom::MxMatrix & data = dynamic_cast<const trrom::MxMatrix &>(*data_);

    // Perform singular value decomposition
    mxArray* output[3];
    mxArray* mx_data = const_cast<mxArray*>(data.array());
    mxArray* mx_economy_size_flag = mxCreateDoubleScalar(0);
    mxArray* inputs[2] = { mx_data, mx_economy_size_flag };
    mxArray* error = mexCallMATLABWithTrapWithObject(3, output, 2, inputs, "svd");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling svd.\n";
    trrom::mx::handleException(error, msg.str());

    // Delete memory allocated by mxCreateDoubleScalar(0)
    mxDestroyArray(mx_economy_size_flag);

    // Copy singular values output from Matlab into TRROM vector data structure
    const int num_singular_values = mxGetM(output[1]);
    singular_values_.reset(new trrom::MxVector(num_singular_values));
    trrom::MxVector & singular_values = dynamic_cast<trrom::MxVector &>(*singular_values_);
    double* singular_values_data = mxGetPr(output[1]);
    double* singular_values_output_data = singular_values.data();
    for(int singular_value = 0; singular_value < num_singular_values; ++singular_value)
    {
        int index = singular_value + (num_singular_values * singular_value);
        singular_values_output_data[singular_value] = singular_values_data[index];
    }

    // Copy left singular vectors output from Matlab into TRROM matrix data structure
    int num_rows = mxGetM(output[0]);
    int num_columns = mxGetN(output[0]);
    left_singular_vectors_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & left_singular_vectors = dynamic_cast<trrom::MxMatrix &>(*left_singular_vectors_);
    trrom::mx::setMxArray(output[0], left_singular_vectors.array());

    // Copy right singular vectors output from Matlab into TRROM matrix data structure
    num_rows = mxGetM(output[2]);
    num_columns = mxGetN(output[2]);
    right_singular_vectors_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & right_singular_vectors = dynamic_cast<trrom::MxMatrix &>(*right_singular_vectors_);
    trrom::mx::setMxArray(output[2], right_singular_vectors.array());
}

}
