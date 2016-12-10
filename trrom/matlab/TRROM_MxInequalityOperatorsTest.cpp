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
#include "TRROM_MxInequalityOperators.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR REDUCED INEQUALITY CONSTRAINT OPERATORS INTERFACE\n");
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
