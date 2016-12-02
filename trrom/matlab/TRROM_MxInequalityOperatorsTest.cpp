/*
 * TRROM_MxInequalityOperatorsTest.cpp
 *
 *  Created on: Dec 2, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_InequalityOperators.hpp"

namespace trrom
{

class MxInequalityOperators : public trrom::InequalityOperators
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxInequalityOperators object
     * Parameters:
     *    \param In
     *          input_: MEX array pointer
     *
     * \return Reference to MxInequalityOperators.
     *
     **/
    explicit MxInequalityOperators(const mxArray* input_) :
            m_Bound(mxDuplicateArray(mxGetField(input_, 0, "bound"))),
            m_Value(mxDuplicateArray(mxGetField(input_, 0, "value"))),
            m_PartialDerivativeState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeState"))),
            m_PartialDerivativeControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControl"))),
            m_PartialDerivativeControlState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControlState"))),
            m_PartialDerivativeControlControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControlControl"))),
            m_PartialDerivativeStateState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeStateState"))),
            m_PartialDerivativeStateControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeStateControl")))
    {
    }
    //! MxInequalityOperators destructor.
    virtual ~MxInequalityOperators()
    {
        mxDestroyArray(m_PartialDerivativeStateControl);
        mxDestroyArray(m_PartialDerivativeStateState);
        mxDestroyArray(m_PartialDerivativeControlControl);
        mxDestroyArray(m_PartialDerivativeControlState);
        mxDestroyArray(m_PartialDerivativeControl);
        mxDestroyArray(m_PartialDerivativeState);
        mxDestroyArray(m_Value);
        mxDestroyArray(m_Bound);
    }
    //@}

    /*!
     * MEX interface for the inequality constraint, i.e h(\mathbf{u}(\mathbf{z}),\mathbf{z})
     * \equiv value(\mathbf{u}(\mathbf{z}),\mathbf{z}) - bound \leq 0, where bound denotes
     * the condition that must be met for a given inequality constraint.
     **/
    double bound()
    {
        // Call inequality constraint bound through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[1] =
            { m_Bound };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 1, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling bound.\n";
        trrom::mx::handleException(error, msg.str());

        // Get inequality constraint value from MATLAB's output
        assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
        double output = mxGetScalar(mx_output[0]);
        return (output);
    }
    /*!
     * Evaluates the current value of the user-defined inequality constraint using the
     * MEX interface. Here \mathbf{u} denotes the state and \mathbf{z} denotes the control variables.
     **/
    double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());

        // Call inequality constraint evaluation through mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[3] =
            { m_Value, mx_state, mx_control };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling value.\n";
        trrom::mx::handleException(error, msg.str());

        // Get inequality constraint value from MATLAB's output
        assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
        double output = mxGetScalar(mx_output[0]);
        return (output);
    }
    /*!
     * Evaluates partial derivative of the inequality constraint with respect to the
     * state variables, \frac{\partial{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the inequality constraint with respect to
     *          the state variables
     **/
    void partialDerivativeState(const trrom::Vector<double> & state_,
                                const trrom::Vector<double> & control_,
                                trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[3] =
            { m_PartialDerivativeState, mx_state, mx_control };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling partialDerivativeState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative of the inequality constraint with respect to the state variables
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
    /*!
     * Evaluates partial derivative of the inequality constraint with respect to the state variables,
     * \frac{\partial{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}, using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the inequality constraint with respect to the control variables
     **/
    void partialDerivativeControl(const trrom::Vector<double> & state_,
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

        // Set output for partial derivative of the inequality constraint with respect to the state variables
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
    /*!
     * Evaluates mixed partial derivative of the inequality constraint with respect to the control and
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}\partial\mathbf{u}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the inequality constraint with respect to the control and state variables
     **/
    void partialDerivativeControlState(const trrom::Vector<double> & state_,
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

        // Set output for the mixed partial derivative of the inequality constraint with respect to the control and state variables
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
    /*!
     * Evaluates second order partial derivative of the inequality constraint with respect to the
     * control variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}^2},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the inequality constraint with respect to the control variables
     **/
    void partialDerivativeControlControl(const trrom::Vector<double> & state_,
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

        // Set output for second order partial derivative of the inequality constraint with respect to the control variables
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
    /*!
     * Evaluates second order partial derivative of the inequality constraint with respect to the
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}^2},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the inequality constraint with respect to the state variables
     **/
    void partialDerivativeStateState(const trrom::Vector<double> & state_,
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

        // Set output for second order partial derivative of the inequality constraint with respect to the control variables
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
    /*!
     * Evaluates mixed partial derivative of the inequality constraint with respect to the control and
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}\partial\mathbf{z}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the inequality constraint with respect to the state and control variables
     **/
    void partialDerivativeStateControl(const trrom::Vector<double> & state_,
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

        // Set output for second order partial derivative of the inequality constraint with respect to the control variables
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }

private:
    mxArray* m_Bound;
    mxArray* m_Value;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_PartialDerivativeControlState;
    mxArray* m_PartialDerivativeControlControl;
    mxArray* m_PartialDerivativeStateState;
    mxArray* m_PartialDerivativeStateControl;

private:
    MxInequalityOperators(const trrom::MxInequalityOperators &);
    trrom::MxInequalityOperators & operator=(const trrom::MxInequalityOperators & rhs_);
};

}

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX REDUCED INEQUALITY CONSTRAINT OPERATORS INTERFACE\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 1 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES ONE INPUT AND RETURNS NO OUTPUT.\n");
        mexErrMsgTxt(error.c_str());
    }

    const int num_states = 9;
    trrom::MxVector states(num_states, 1.);
    const int num_controls = 9;
    trrom::MxVector controls(num_controls, 1.);
    trrom::MxInequalityOperators inequality(pInput[0]);

    // **** TEST VALUE ****
    msg.assign("value");
    double output = inequality.value(states, controls);
    bool did_test_pass = static_cast<double>(9) == output;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST BOUND ****
    msg.assign("bound");
    output = inequality.bound();
    did_test_pass = static_cast<double>(99) == output;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("partialDerivativeState");
    trrom::MxVector out(num_controls);
    inequality.partialDerivativeState(states, controls, out);
    trrom::MxVector gold(num_controls, 1.);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("partialDerivativeControl");
    inequality.partialDerivativeControl(states, controls, out);
    gold.fill(2);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("partialDerivativeControlControl");
    trrom::MxVector vector(num_controls, 2.);
    inequality.partialDerivativeControlControl(states, controls, vector, out);
    gold.fill(5);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("partialDerivativeStateState");
    inequality.partialDerivativeStateState(states, controls, vector, out);
    gold.fill(7);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE STATE AND CONTROL VARIABLES ****
    msg.assign("partialDerivativeStateControl");
    inequality.partialDerivativeStateControl(states, controls, vector, out);
    gold.fill(9);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL AND STATE VARIABLES ****
    msg.assign("partialDerivativeControlState");
    inequality.partialDerivativeControlState(states, controls, vector, out);
    gold.fill(3);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);
}
