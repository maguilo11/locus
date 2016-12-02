/*
 * TRROM_MxDirectSolver.cpp
 *
 *  Created on: Nov 29, 2016
 *      Author: maguilo
 */

#include <mex.h>
#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxDirectSolver.hpp"

namespace trrom
{

MxDirectSolver::MxDirectSolver()
{
}

MxDirectSolver::~MxDirectSolver()
{
}

void MxDirectSolver::solve(const trrom::Matrix<double> & A_,
                           const trrom::Vector<double> & rhs_,
                           trrom::Vector<double> & lhs_)
{
    const trrom::MxVector & rhs = dynamic_cast<const trrom::MxVector &>(rhs_);
    const mxArray* rhs_mex_array = rhs.array();
    const trrom::MxMatrix & A = dynamic_cast<const trrom::MxMatrix &>(A_);
    const mxArray* A_mex_array = A.array();

    mxArray* output[1];
    mxArray* inputs[2];
    inputs[0] = const_cast<mxArray*>(A_mex_array);
    inputs[1] = const_cast<mxArray*>(rhs_mex_array);
    mxArray* error = mexCallMATLABWithTrapWithObject(1, output, 2, inputs, "mldivide");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling mldivide.\n";
    trrom::mx::handleException(error, msg.str());

    trrom::MxVector & lhs = dynamic_cast<trrom::MxVector &>(lhs_);
    lhs.setMxArray(output[0]);
}

}
