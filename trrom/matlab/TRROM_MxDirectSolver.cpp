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
    const trrom::MxMatrix & A = dynamic_cast<const trrom::MxMatrix &>(A_);
    mxArray* mx_A = const_cast<mxArray*>(A.array());

    // Set right-hand-side vector: Check if transpose operation is needed
    const trrom::MxVector & rhs = dynamic_cast<const trrom::MxVector &>(rhs_);
    mxArray* mx_rhs[1];
    if(mxGetN(mx_A) == mxGetM(rhs.array()))
    {
        mx_rhs[0] = const_cast<mxArray*>(rhs.array());
    }
    else
    {
        mxArray* input_rhs = const_cast<mxArray*>(rhs.array());
        mxArray* inputs[1] = { input_rhs };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_rhs, 1, inputs, "transpose");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling transpose.\n";
        trrom::mx::handleException(error, msg.str());
    }

    // Solve system of equations
    mxArray* solve_output[1];
    mxArray* solve_inputs[2] = { mx_A, mx_rhs[0] };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, solve_output, 2, solve_inputs, "mldivide");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling mldivide.\n";
    trrom::mx::handleException(error, msg.str());

    // Set solution vector output
    trrom::MxVector & lhs = dynamic_cast<trrom::MxVector &>(lhs_);
    lhs.setMxArray(solve_output[0]);
}

}
