/*
 * TRROM_MxReducedObjectiveOperatorsTest.cpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxReducedObjectiveOperators.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR REDUCED OBJECTIVE OPERATORS\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 1 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT ARGUMENTS. FUNCTION TAKES ONE INPUT AND RETURNS NO OUTPUT.\n");
        mexErrMsgTxt(error.c_str());
    }

    const int num_states = 9;
    trrom::MxVector states(num_states, 1.);
    const int num_controls = 9;
    trrom::MxVector controls(num_controls, 1.);
    trrom::MxReducedObjectiveOperators objective(pInput[0]);

    // **** TEST VALUE ****
    msg.assign("value");
    double output = objective.value(states, controls);
    bool did_test_pass = static_cast<double>(9) == output;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST EVALUATE OBJECTIVE INEXACTNESS ****
    msg.assign("evaluateObjectiveInexactness");
    output = objective.evaluateObjectiveInexactness(states, controls);
    did_test_pass = static_cast<double>(13) == output;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST EVALUATE GRADIENT INEXACTNESS ****
    msg.assign("evaluateGradientInexactness");
    output = objective.evaluateGradientInexactness(states, controls);
    did_test_pass = static_cast<double>(14) == output;
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("partialDerivativeState");
    trrom::MxVector out(num_controls);
    objective.partialDerivativeState(states, controls, out);
    trrom::MxVector gold(num_controls, 1.);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("partialDerivativeControl");
    objective.partialDerivativeControl(states, controls, out);
    gold.fill(2);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("partialDerivativeControlControl");
    trrom::MxVector vector(num_controls, 2.);
    objective.partialDerivativeControlControl(states, controls, vector, out);
    gold.fill(5);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("partialDerivativeStateState");
    objective.partialDerivativeStateState(states, controls, vector, out);
    gold.fill(7);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE STATE AND CONTROL VARIABLES ****
    msg.assign("partialDerivativeStateControl");
    objective.partialDerivativeStateControl(states, controls, vector, out);
    gold.fill(9);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL AND STATE VARIABLES ****
    msg.assign("partialDerivativeControlState");
    objective.partialDerivativeControlState(states, controls, vector, out);
    gold.fill(3);
    did_test_pass = trrom::mx::checkResults(gold, out);
    trrom::mx::assert_test(msg, did_test_pass);
}
