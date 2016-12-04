/*
 * TRROM_MxReducedBasisPDE.cpp
 *
 *  Created on: Dec 3, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>
#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_MxReducedBasisPDE.hpp"

namespace trrom
{

MxReducedBasisPDE::MxReducedBasisPDE(const mxArray* input_) :
        m_Solve(mxDuplicateArray(mxGetField(input_, 0, "solve"))),
        m_ApplyInverseJacobianState(mxDuplicateArray(mxGetField(input_, 0, "applyInverseJacobianState"))),
        m_ApplyAdjointInverseJacobianState(mxDuplicateArray(mxGetField(input_, 0, "applyAdjointInverseJacobianState"))),
        m_PartialDerivativeState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeState"))),
        m_PartialDerivativeControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControl"))),
        m_AdjointPartialDerivativeState(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeState"))),
        m_AdjointPartialDerivativeControl(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeControl"))),
        m_AdjointPartialDerivativeStateState(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeStateState"))),
        m_AdjointPartialDerivativeControlState(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeControlState"))),
        m_AdjointPartialDerivativeStateControl(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeStateControl"))),
        m_AdjointPartialDerivativeControlControl(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeControlControl")))
{
}

MxReducedBasisPDE::~MxReducedBasisPDE()
{
    mxDestroyArray(m_AdjointPartialDerivativeControlControl);
    mxDestroyArray(m_AdjointPartialDerivativeStateControl);
    mxDestroyArray(m_AdjointPartialDerivativeControlState);
    mxDestroyArray(m_AdjointPartialDerivativeStateState);
    mxDestroyArray(m_AdjointPartialDerivativeControl);
    mxDestroyArray(m_AdjointPartialDerivativeState);
    mxDestroyArray(m_PartialDerivativeControl);
    mxDestroyArray(m_PartialDerivativeState);
    mxDestroyArray(m_ApplyAdjointInverseJacobianState);
    mxDestroyArray(m_ApplyInverseJacobianState);
    mxDestroyArray(m_Solve);
}

void MxReducedBasisPDE::solve(const trrom::Vector<double> & control_,
                              trrom::Vector<double> & solution_,
                              trrom::ReducedBasisData & data_)
{
    // Dynamic cast of active indices vector data structure
    trrom::MxVector & active_indices = dynamic_cast<trrom::MxVector &>(data_.getLeftHandSideActiveIndices());

    // Get fidelity, there are two options: low- or high-fidelity
    mxArray* mx_fidelity;
    if(data_.fidelity() == trrom::types::HIGH_FIDELITY)
    {
        mx_fidelity = mxCreateString("HIGH_FIDELITY");
        active_indices.fill(1);
    }
    else
    {
        mx_fidelity = mxCreateString("LOW_FIDELITY");
    }

    // Set control input
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    // Call solve function through the mex interface
    mxArray* mx_output[3];
    mxArray* mx_input[4] = { m_Solve, mx_control, active_indices.array(), mx_fidelity };
    mxArray* error = mexCallMATLABWithTrapWithObject(3, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling solve.\n";
    trrom::mx::handleException(error, msg.str());

    // Set solution output only when high fidelity solve is active
    if(data_.fidelity() == trrom::types::HIGH_FIDELITY)
    {
        assert(static_cast<size_t>(solution_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & solution = dynamic_cast<trrom::MxVector &>(solution_);
        solution.setMxArray(mx_output[0]);
    }
    else
    {
        solution_.fill(0);
    }

    // Set left-hand side snapshot output
    assert(static_cast<size_t>(data_.getLeftHandSideSnapshot().size()) == mxGetNumberOfElements(mx_output[1]));
    trrom::MxVector & lhs_snapshot = dynamic_cast<trrom::MxVector &>(data_.getLeftHandSideSnapshot());
    lhs_snapshot.setMxArray(mx_output[1]);

    // Set right-hand side snapshot output
    assert(static_cast<size_t>(data_.getRightHandSideSnapshot().size()) == mxGetNumberOfElements(mx_output[2]));
    trrom::MxVector & rhs_snapshot = dynamic_cast<trrom::MxVector &>(data_.getRightHandSideSnapshot());
    rhs_snapshot.setMxArray(mx_output[2]);

    mxDestroyArray(mx_fidelity);
}

void MxReducedBasisPDE::applyInverseJacobianState(const trrom::Vector<double> & state_,
                                                  const trrom::Vector<double> & control_,
                                                  const trrom::Vector<double> & rhs_,
                                                  trrom::Vector<double> & solution_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & right_hand_side = dynamic_cast<const trrom::MxVector &>(rhs_);
    mxArray* mx_right_hand_side = const_cast<mxArray*>(right_hand_side.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_ApplyInverseJacobianState, mx_state, mx_control, mx_right_hand_side };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling applyInverseJacobianState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for solution field
    assert(static_cast<size_t>(solution_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & solution = dynamic_cast<trrom::MxVector &>(solution_);
    solution.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::applyAdjointInverseJacobianState(const trrom::Vector<double> & state_,
                                                         const trrom::Vector<double> & control_,
                                                         const trrom::Vector<double> & rhs_,
                                                         trrom::Vector<double> & solution_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & right_hand_side = dynamic_cast<const trrom::MxVector &>(rhs_);
    mxArray* mx_right_hand_side = const_cast<mxArray*>(right_hand_side.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_ApplyAdjointInverseJacobianState, mx_state, mx_control, mx_right_hand_side };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling applyAdjointInverseJacobianState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for solution field
    assert(static_cast<size_t>(solution_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & solution = dynamic_cast<trrom::MxVector &>(solution_);
    solution.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::partialDerivativeState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_PartialDerivativeState, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling partialDerivativeState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::partialDerivativeControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & vector_,
                                                 trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_PartialDerivativeControl, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling partialDerivativeControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::adjointPartialDerivativeState(const trrom::Vector<double> & state_,
                                                      const trrom::Vector<double> & control_,
                                                      const trrom::Vector<double> & dual_,
                                                      trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_AdjointPartialDerivativeState, mx_state, mx_control, mx_dual };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::adjointPartialDerivativeControl(const trrom::Vector<double> & state_,
                                                        const trrom::Vector<double> & control_,
                                                        const trrom::Vector<double> & dual_,
                                                        trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_AdjointPartialDerivativeControl, mx_state, mx_control, mx_dual };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::adjointPartialDerivativeStateState(const trrom::Vector<double> & state_,
                                                           const trrom::Vector<double> & control_,
                                                           const trrom::Vector<double> & dual_,
                                                           const trrom::Vector<double> & vector_,
                                                           trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_AdjointPartialDerivativeStateState, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeStateState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::adjointPartialDerivativeStateControl(const trrom::Vector<double> & state_,
                                                             const trrom::Vector<double> & control_,
                                                             const trrom::Vector<double> & dual_,
                                                             const trrom::Vector<double> & vector_,
                                                             trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_AdjointPartialDerivativeStateControl, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeStateControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::adjointPartialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                               const trrom::Vector<double> & control_,
                                                               const trrom::Vector<double> & dual_,
                                                               const trrom::Vector<double> & vector_,
                                                               trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_AdjointPartialDerivativeControlControl, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeControlControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedBasisPDE::adjointPartialDerivativeControlState(const trrom::Vector<double> & state_,
                                                             const trrom::Vector<double> & control_,
                                                             const trrom::Vector<double> & dual_,
                                                             const trrom::Vector<double> & vector_,
                                                             trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_AdjointPartialDerivativeControlState, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeControlState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

}
