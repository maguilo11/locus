/*
 * TRROM_MxReducedObjectiveOperators.cpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#include <sstream>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxReducedObjectiveOperators.hpp"

namespace trrom
{

MxReducedObjectiveOperators::MxReducedObjectiveOperators(const mxArray* input_) :
        m_Value(mxDuplicateArray(mxGetField(input_, 0, "value"))),
        m_GradientError(mxDuplicateArray(mxGetField(input_, 0, "evaluateGradientInexactness"))),
        m_ObjectiveError(mxDuplicateArray(mxGetField(input_, 0, "evaluateObjectiveInexactness"))),
        m_PartialDerivativeState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeState"))),
        m_PartialDerivativeControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControl"))),
        m_PartialDerivativeControlState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControlState"))),
        m_PartialDerivativeControlControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControlControl"))),
        m_PartialDerivativeStateState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeStateState"))),
        m_PartialDerivativeStateControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeStateControl")))
{
}

MxReducedObjectiveOperators::~MxReducedObjectiveOperators()
{
    mxDestroyArray(m_PartialDerivativeStateControl);
    mxDestroyArray(m_PartialDerivativeStateState);
    mxDestroyArray(m_PartialDerivativeControlControl);
    mxDestroyArray(m_PartialDerivativeControlState);
    mxDestroyArray(m_PartialDerivativeControl);
    mxDestroyArray(m_PartialDerivativeState);
    mxDestroyArray(m_ObjectiveError);
    mxDestroyArray(m_GradientError);
    mxDestroyArray(m_Value);
}

double MxReducedObjectiveOperators::value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    // Call objective function evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[3] = {m_Value, mx_state, mx_control};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling value.\n";
    trrom::mx::handleException(error, msg.str());

    // Get objective function value from MATLAB's output
    assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
    double output = mxGetScalar(mx_output[0]);
    return (output);
}

double MxReducedObjectiveOperators::evaluateObjectiveInexactness(const trrom::Vector<double> & state_,
                                                                 const trrom::Vector<double> & control_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    // Call objective function evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[3] = {m_ObjectiveError, mx_state, mx_control};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling evaluateObjectiveInexactness.\n";
    trrom::mx::handleException(error, msg.str());

    // Get objective function value from MATLAB's output
    assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
    double output = mxGetScalar(mx_output[0]);
    return (output);
}

double MxReducedObjectiveOperators::evaluateGradientInexactness(const trrom::Vector<double> & state_,
                                                                const trrom::Vector<double> & control_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    // Call objective function evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[3] = {m_GradientError, mx_state, mx_control};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling evaluateGradientInexactness.\n";
    trrom::mx::handleException(error, msg.str());

    // Get objective function value from MATLAB's output
    assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
    double output = mxGetScalar(mx_output[0]);
    return (output);
}

void MxReducedObjectiveOperators::partialDerivativeState(const trrom::Vector<double> & state_,
                                                         const trrom::Vector<double> & control_,
                                                         trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[3] = {m_PartialDerivativeState, mx_state, mx_control};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative of the objective function with respect to the state variables
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedObjectiveOperators::partialDerivativeControl(const trrom::Vector<double> & state_,
                                                           const trrom::Vector<double> & control_,
                                                           trrom::Vector<double> & output_)
{
    const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    // Call partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[3] = {m_PartialDerivativeControl, mx_state, mx_control};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for partial derivative of the objective function with respect to the state variables
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedObjectiveOperators::partialDerivativeControlState(const trrom::Vector<double> & state_,
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

    // Call mixed partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = {m_PartialDerivativeControlState, mx_state, mx_control, mx_vector};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControlState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for the mixed partial derivative of the objective function with respect to the control and state variables
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedObjectiveOperators::partialDerivativeControlControl(const trrom::Vector<double> & state_,
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

    // Call mixed partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = {m_PartialDerivativeControlControl, mx_state, mx_control, mx_vector};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControlControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for second order partial derivative of the objective function with respect to the control variables
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedObjectiveOperators::partialDerivativeStateState(const trrom::Vector<double> & state_,
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

    // Call mixed partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = {m_PartialDerivativeStateState, mx_state, mx_control, mx_vector};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeStateState.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for second order partial derivative of the objective function with respect to the control variables
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void MxReducedObjectiveOperators::partialDerivativeStateControl(const trrom::Vector<double> & state_,
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

    // Call mixed partial derivative evaluation through the mex interface
    mxArray* mx_output[1];
    mxArray* mx_input[4] = {m_PartialDerivativeStateControl, mx_state, mx_control, mx_vector};
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeStateControl.\n";
    trrom::mx::handleException(error, msg.str());

    // Set output for second order partial derivative of the objective function with respect to the control variables
    assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
    trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
    output.setMxArray(mx_output[0]);
}

}
