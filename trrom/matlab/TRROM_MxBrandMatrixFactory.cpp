/*
 * TRROM_MxBrandMatrixFactory.cpp
 *
 *  Created on: Dec 6, 2016
 *      Author: maguilo
 */

#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxBrandMatrixFactory.hpp"

namespace trrom
{

MxBrandMatrixFactory::MxBrandMatrixFactory()
{
}

MxBrandMatrixFactory::~MxBrandMatrixFactory()
{
}

void MxBrandMatrixFactory::buildMatrixK(const std::shared_ptr<trrom::Vector<double> > & sigma_,
                                        const std::shared_ptr<trrom::Matrix<double> > & M_,
                                        const std::shared_ptr<trrom::Matrix<double> > & R_,
                                        std::shared_ptr<trrom::Matrix<double> > & K_)
{
    // Cast const trrom::Matrix references into const trrom::MxMatrix references
    const trrom::MxVector & sigma = dynamic_cast<const trrom::MxVector &>(*sigma_);
    const trrom::MxMatrix & M_matrix = dynamic_cast<const trrom::MxMatrix &>(*M_);
    const trrom::MxMatrix & R_matrix = dynamic_cast<const trrom::MxMatrix &>(*R_);

    // Cast const mxArray* into non-const mxArray*
    mxArray* mx_sigma = const_cast<mxArray*>(sigma.array());
    mxArray* mx_M_matrix = const_cast<mxArray*>(M_matrix.array());
    mxArray* mx_R_matrix = const_cast<mxArray*>(R_matrix.array());

    // Create diagonal matrix
    mxArray* mex_output_one[1];
    mxArray* mex_inputs_one[1] = { mx_sigma };
    mxArray* error_one = mexCallMATLABWithTrapWithObject(1, mex_output_one, 1, mex_inputs_one, "diag");
    std::ostringstream msg_one;
    msg_one << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling diag.\n";
    trrom::mx::handleException(error_one, msg_one.str());

    // Create array of all zeros
    double array_num_rows = static_cast<double>(R_->getNumRows());
    mxArray* mx_num_rows = mxCreateDoubleScalar(array_num_rows);
    double array_columns = static_cast<double>(sigma_->size());
    mxArray* mx_num_columns = mxCreateDoubleScalar(array_columns);

    mxArray* mex_output_two[1];
    mxArray* mex_inputs_two[2] = { mx_num_rows, mx_num_columns };
    mxArray* error_two = mexCallMATLABWithTrapWithObject(1, mex_output_two, 2, mex_inputs_two, "zeros");
    std::ostringstream msg_two;
    msg_two << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling zeros.\n";
    trrom::mx::handleException(error_two, msg_two.str());

    mxDestroyArray(mx_num_columns);
    mxDestroyArray(mx_num_rows);

    // Concatenate arrays vertically
    mxArray* mex_output_three[1];
    mxArray* mex_inputs_three[2] = { mex_output_one[0], mex_output_two[0] };
    mxArray* error_three = mexCallMATLABWithTrapWithObject(1, mex_output_three, 2, mex_inputs_three, "vertcat");
    std::ostringstream msg_three;
    msg_three << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling vertcat.\n";
    trrom::mx::handleException(error_three, msg_three.str());

    // Concatenate arrays vertically
    mxArray* mex_output_four[1];
    mxArray* mex_inputs_four[2] = { mx_M_matrix, mx_R_matrix };
    mxArray* error_four = mexCallMATLABWithTrapWithObject(1, mex_output_four, 2, mex_inputs_four, "vertcat");
    std::ostringstream msg_four;
    msg_four << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling vertcat.\n";
    trrom::mx::handleException(error_four, msg_four.str());

    // Concatenate arrays horizontally and create matrix K
    mxArray* mex_output_five[1];
    mxArray* mex_inputs_five[2] = { mex_output_three[0], mex_output_four[0] };
    mxArray* error_five = mexCallMATLABWithTrapWithObject(1, mex_output_five, 2, mex_inputs_five, "horzcat");
    std::ostringstream msg_five;
    msg_five << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling horzcat.\n";
    trrom::mx::handleException(error_five, msg_five.str());

    /* Reset output data shared pointer, cast trrom::Matrix references into trrom::MxMatrix
     * references, and assign new contents to output matrix. */
    const int num_rows = mxGetM(mex_output_five[0]);
    const int num_columns = mxGetN(mex_output_five[0]);
    K_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & K_matrix = dynamic_cast<trrom::MxMatrix &>(*K_);
    K_matrix.setMxArray(mex_output_five[0]);
}

void MxBrandMatrixFactory::buildMatrixUbar(const std::shared_ptr<trrom::Matrix<double> > & Uc_,
                                           const std::shared_ptr<trrom::Matrix<double> > & Q_,
                                           const std::shared_ptr<trrom::Matrix<double> > & Ur_,
                                           std::shared_ptr<trrom::Matrix<double> > & Un_)
{
    // Cast const trrom::Matrix references into const trrom::MxMatrix references
    const trrom::MxMatrix & unitary_matrix = dynamic_cast<const trrom::MxMatrix &>(*Q_);
    const trrom::MxMatrix & current_left_singular_vectors = dynamic_cast<const trrom::MxMatrix &>(*Uc_);
    const trrom::MxMatrix & reduced_left_singular_vectors = dynamic_cast<const trrom::MxMatrix &>(*Ur_);

    // Cast const mxArray* into non-const mxArray*
    mxArray* mx_unitary_matrix = const_cast<mxArray*>(unitary_matrix.array());
    mxArray* mx_current_left_singular_vectors = const_cast<mxArray*>(current_left_singular_vectors.array());
    mxArray* mx_reduced_left_singular_vectors = const_cast<mxArray*>(reduced_left_singular_vectors.array());

    // Concatenate arrays horizontally
    mxArray* mex_output_one[1];
    mxArray* mex_inputs_one[2] = { mx_current_left_singular_vectors, mx_unitary_matrix };
    mxArray* error_one = mexCallMATLABWithTrapWithObject(1, mex_output_one, 2, mex_inputs_one, "horzcat");
    std::ostringstream msg_one;
    msg_one << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling horzcat.\n";
    trrom::mx::handleException(error_one, msg_one.str());

    // Perform matrix-matrix multiplication
    mxArray* mex_output_two[1];
    mxArray* mex_inputs_two[2] = { mex_output_one[0], mx_reduced_left_singular_vectors };
    mxArray* error_two = mexCallMATLABWithTrapWithObject(1, mex_output_two, 2, mex_inputs_two, "mtimes");
    std::ostringstream msg_two;
    msg_two << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling mtimes.\n";
    trrom::mx::handleException(error_two, msg_two.str());

    /* Reset output data shared pointer, cast trrom::Matrix references into trrom::MxMatrix
     * references, and assign new contents to output matrix. */
    const int num_rows = mxGetM(mex_output_two[0]);
    const int num_columns = mxGetN(mex_output_two[0]);
    Un_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & new_left_singular_vectors = dynamic_cast<trrom::MxMatrix &>(*Un_);
    new_left_singular_vectors.setMxArray(mex_output_two[0]);
}

void MxBrandMatrixFactory::buildMatrixVbar(const std::shared_ptr<trrom::Matrix<double> > & Vc_,
                                           const std::shared_ptr<trrom::Matrix<double> > & Vr_,
                                           std::shared_ptr<trrom::Matrix<double> > & Vn_)
{
    // Cast const trrom::Matrix references into const trrom::MxMatrix references
    const trrom::MxMatrix & current_right_singular_vectors = dynamic_cast<const trrom::MxMatrix &>(*Vc_);
    const trrom::MxMatrix & reduced_right_singular_vectors = dynamic_cast<const trrom::MxMatrix &>(*Vr_);

    // Cast const mxArray* into non-const mxArray*
    mxArray* mx_current_right_singular_vectors = const_cast<mxArray*>(current_right_singular_vectors.array());
    mxArray* mx_reduced_right_singular_vectors = const_cast<mxArray*>(reduced_right_singular_vectors.array());

    // Create mxArray Double Scalar pointer for identity matrix dimension input
    double identity_dim = Vr_->getNumRows() - Vc_->getNumCols();
    mxArray* mx_identity_dim = mxCreateDoubleScalar(identity_dim);

    // Construct identity matrix
    mxArray* mex_output_one[1];
    mxArray* mex_inputs_one[1] = { mx_identity_dim };
    mxArray* error_one = mexCallMATLABWithTrapWithObject(1, mex_output_one, 1, mex_inputs_one, "eye");
    std::ostringstream msg_one;
    msg_one << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling eye.\n";
    trrom::mx::handleException(error_one, msg_one.str());

    // Construct block diagonal matrix from input arguments
    mxArray* mex_output_two[1];
    mxArray* mex_inputs_two[2] = { mx_current_right_singular_vectors, mex_output_one[0] };
    mxArray* error_two = mexCallMATLABWithTrapWithObject(1, mex_output_two, 2, mex_inputs_two, "blkdiag");
    std::ostringstream msg_two;
    msg_two << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling blkdiag.\n";
    trrom::mx::handleException(error_two, msg_two.str());

    // Perform matrix-matrix multiplication
    mxArray* mex_output_three[1];
    mxArray* mex_inputs_three[2] = { mex_output_two[0], mx_reduced_right_singular_vectors };
    mxArray* error_three = mexCallMATLABWithTrapWithObject(1, mex_output_three, 2, mex_inputs_three, "mtimes");
    std::ostringstream msg_three;
    msg_three << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling mtimes.\n";
    trrom::mx::handleException(error_three, msg_three.str());

    /* Reset output data shared pointer, cast trrom::Matrix references into trrom::MxMatrix
     * references, and assign new contents to output matrix. */
    const int num_rows = mxGetM(mex_output_three[0]);
    const int num_columns = mxGetN(mex_output_three[0]);
    Vn_.reset(new trrom::MxMatrix(num_rows, num_columns));
    trrom::MxMatrix & new_right_singular_vectors = dynamic_cast<trrom::MxMatrix &>(*Vn_);
    new_right_singular_vectors.setMxArray(mex_output_three[0]);
}

}
