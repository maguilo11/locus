/*
 * TRROM_MxReducedBasisPDETest.cpp
 *
 *  Created on: Dec 2, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_MxReducedBasisPDE.hpp"

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR REDUCED BASIS PDE CONSTRAINT INTERFACE\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 1 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES ONE INPUT AND RETURNS NO OUTPUT.\n");
        mexErrMsgTxt(error.c_str());
    }

    // Set data for unit test
    const int num_states = 9;
    trrom::MxVector dual(num_states, 3.);
    trrom::MxVector states(num_states, 1.);
    trrom::MxVector output(num_states, 1.);
    trrom::MxVector delta_states(num_states, 2.);

    const int num_controls = 9;
    trrom::MxVector controls(num_controls, 1.);
    trrom::MxVector delta_controls(num_controls, 3.);

    trrom::ReducedBasisData data;
    const int lhs_snapshot_length = num_states * num_states;
    trrom::MxVector left_hand_side(lhs_snapshot_length);
    data.setLeftHandSideSnapshot(left_hand_side);
    trrom::MxVector right_hand_side(num_states, 1.);
    data.setRightHandSideSnapshot(right_hand_side);
    data.setLeftHandSideActiveIndices(left_hand_side);

    // Instantiate instance from class MxReducedBasisPDE
    trrom::MxReducedBasisPDE equality(pInput[0]);

    // **** TEST APPLY INVERSE JACOBIAN STATE ****
    msg.assign("applyInverseJacobianState");
    equality.applyInverseJacobianState(states, controls, right_hand_side, output);
    trrom::MxVector gold(num_states, 2.);
    bool did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST APPLY ADJOINT INVERSE JACOBIAN STATE ****
    msg.assign("applyInverseAdjointJacobianState");
    equality.applyInverseAdjointJacobianState(states, controls, right_hand_side, output);
    gold.fill(3);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("partialDerivativeState");
    equality.partialDerivativeState(states, controls, delta_states, output);
    gold.fill(6);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("partialDerivativeControl");
    equality.partialDerivativeControl(states, controls, delta_states, output);
    gold.fill(4);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("adjointPartialDerivativeState");
    equality.adjointPartialDerivativeState(states, controls, dual, output);
    gold.fill(5);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("adjointPartialDerivativeControl");
    equality.adjointPartialDerivativeControl(states, controls, dual, output);
    gold.fill(10);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL AND STATE VARIABLES ****
    msg.assign("adjointPartialDerivativeControlState");
    equality.adjointPartialDerivativeControlState(states, controls, dual, delta_states, output);
    gold.fill(7);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT OF SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("adjointPartialDerivativeControlControl");
    equality.adjointPartialDerivativeControlControl(states, controls, dual, delta_controls, output);
    gold.fill(20);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT OF SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("adjointPartialDerivativeStateState");
    equality.adjointPartialDerivativeStateState(states, controls, dual, delta_states, output);
    gold.fill(19);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE STATE AND CONTROL VARIABLES ****
    msg.assign("adjointPartialDerivativeStateControl");
    equality.adjointPartialDerivativeStateControl(states, controls, dual, delta_controls, output);
    gold.fill(37);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST HIGH FIDELITY SOLVE ****
    msg.assign("solve - high fidelity - output");
    data.fidelity(trrom::types::HIGH_FIDELITY);
    equality.solve(controls, output, data);
    gold.fill(23);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // ASSERT RESULTS
    msg.assign("solve - high fidelity - rhs");
    for(int index = 0; index < gold.size(); ++index)
    {
        gold[index] = index + 1;
    }
    did_test_pass = trrom::mx::checkResults(gold, *data.getRightHandSideSnapshot());
    trrom::mx::assert_test(msg, did_test_pass);
    msg.assign("solve - high fidelity - lhs");
    trrom::MxVector lhs_gold(lhs_snapshot_length);
    for(int index = 0; index < lhs_gold.size(); ++index)
    {
        lhs_gold[index] = index + 1;
    }
    did_test_pass = trrom::mx::checkResults(lhs_gold, *data.getLeftHandSideSnapshot());
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST LOW FIDELITY SOLVE ****
    msg.assign("solve - low fidelity - output");
    data.fidelity(trrom::types::LOW_FIDELITY);
    (*data.getLeftHandSideActiveIndices())[22] = 0;
    (*data.getLeftHandSideActiveIndices())[30] = 0;
    (*data.getLeftHandSideActiveIndices())[80] = 0;
    equality.solve(controls, output, data);
    gold.fill(0);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("solve - low fidelity - rhs");
    for(int index = 0; index < gold.size(); ++index)
    {
        gold[index] = 2. * (index + 1.);
    }
    did_test_pass = trrom::mx::checkResults(gold, *data.getRightHandSideSnapshot());
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("solve - low fidelity - lhs");
    for(int index = 0; index < lhs_gold.size(); ++index)
    {
        lhs_gold[index] = index + 1;
    }
    lhs_gold[22] = 0; lhs_gold[30] = 0; lhs_gold[80] = 0;
    did_test_pass = trrom::mx::checkResults(lhs_gold, *data.getLeftHandSideSnapshot());
    trrom::mx::assert_test(msg, did_test_pass);
}
