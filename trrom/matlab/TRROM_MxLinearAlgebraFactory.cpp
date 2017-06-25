/*
 * TRROM_MxLinearAlgebraFactory.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: maguilo
 */

#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxLinearAlgebraFactory.hpp"

namespace trrom
{

MxLinearAlgebraFactory::MxLinearAlgebraFactory()
{
}

MxLinearAlgebraFactory::~MxLinearAlgebraFactory()
{
}

void MxLinearAlgebraFactory::reshape(const int & num_rows_,
                                     const int & num_columns_,
                                     const std::shared_ptr<trrom::Vector<double> > & input_,
                                     std::shared_ptr<trrom::Matrix<double> > & output_)
{
    // Set input data
    mxArray* mx_num_rows = mxCreateDoubleScalar(num_rows_);
    mxArray* mx_num_columns = mxCreateDoubleScalar(num_columns_);
    const trrom::MxVector & input = dynamic_cast<const trrom::MxVector &>(*input_);
    mxArray* mx_input = const_cast<mxArray*>(input.array());

    // Reshape array
    mxArray* output[1];
    mxArray* inputs[3] = { mx_input, mx_num_rows, mx_num_columns };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, output, 3, inputs, "reshape");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling reshape.\n";
    trrom::mx::handleException(error, msg.str());

    // Deallocate created memory
    mxDestroyArray(mx_num_rows);
    mxDestroyArray(mx_num_columns);

    // Set output
    output_ = std::make_shared<trrom::MxMatrix>(output[0]);
}

void MxLinearAlgebraFactory::buildLocalVector(const int & length_, std::shared_ptr<trrom::Vector<double> > & output_)
{
    output_ = std::make_shared<trrom::MxVector>(length_);
}

void MxLinearAlgebraFactory::buildLocalMatrix(const int & num_rows_,
                                              const int & num_columns_,
                                              std::shared_ptr<trrom::Matrix<double> > & output_)
{
    output_ = std::make_shared<trrom::MxMatrix>(num_rows_, num_columns_);
}

void MxLinearAlgebraFactory::buildMultiVector(const int & num_vectors_,
                                              const std::shared_ptr<trrom::Vector<double> > & vector_,
                                              std::shared_ptr<trrom::Matrix<double> > & output_)
{
    const int num_rows = vector_->size();
    output_ = std::make_shared<trrom::MxMatrix>(num_rows, num_vectors_);
}

}
