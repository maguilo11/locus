/*
 * DOTk_DiagnosticsObjectiveTypeNPTest.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_ObjectiveTypeNP.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"
#include "DOTk_DiagnosticsObjectiveTypeNP.hpp"

#include "DOTk_NocedalAndWrightObjectiveNLP.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDiagnosticsObjectiveTypeNPTest
{

TEST(DOTk_NLP, variables2)
{
    size_t num_dual = 3;
    size_t num_states = 5;
    size_t num_controls = 2;

    dotk::StdVector<Real> dual(num_dual, 2.);
    dotk::StdVector<Real> state(num_states, 4.);
    dotk::StdVector<Real> control(num_controls, 1.);
    dotk::nlp::variables vars(state, control, dual);

    dotk::StdVector<Real> dual_gold(num_dual, 2.);
    dotk::StdVector<Real> state_gold(num_states, 4.);
    dotk::StdVector<Real> control_gold(num_controls, 1.);

    dotk::gtest::checkResults(*vars.mDual, dual_gold);
    dotk::gtest::checkResults(*vars.mState, state_gold);
    dotk::gtest::checkResults(*vars.mControl, control_gold);
}

TEST(DOTk_NLP, variables1)
{
    size_t num_states = 5;
    size_t num_controls = 2;

    dotk::StdVector<Real> state(num_states, 4.);
    dotk::StdVector<Real> control(num_controls, 1.);
    dotk::nlp::variables vars(state, control);

    dotk::StdVector<Real> dual_gold(num_states, 0.);
    dotk::StdVector<Real> state_gold(num_states, 4.);
    dotk::StdVector<Real> control_gold(num_controls, 1.);

    dotk::gtest::checkResults(*vars.mDual, dual_gold);
    dotk::gtest::checkResults(*vars.mState, state_gold);
    dotk::gtest::checkResults(*vars.mControl, control_gold);
}

TEST(DOTk_NLP, resetField)
{
    size_t num_states = 5;
    size_t num_controls = 2;
    dotk::StdVector<Real> state(num_states, 4.);
    dotk::StdVector<Real> control(num_controls, 1.);
    dotk::nlp::variables vars(state, control);

    dotk::StdVector<Real> original_state(num_states, 2.);
    dotk::StdVector<Real> original_control(num_controls, 3.);

    // TEST 1: DERIVATIVE W.R.T. CONTROL
    dotk::nlp::resetField(original_control, vars, dotk::types::Z);
    dotk::gtest::checkResults(*vars.mState, state);
    dotk::gtest::checkResults(*vars.mControl, original_control);

    // TEST 2: DERIVATIVE W.R.T. STATE
    vars.mControl->update(1., control, 0.);
    dotk::nlp::resetField(original_state, vars, dotk::types::U);
    dotk::gtest::checkResults(*vars.mState, original_state);
    dotk::gtest::checkResults(*vars.mControl, control);

    // TEST 3: DERIVATIVE W.R.T. STATE_CONTROL
    vars.mState->update(1., state, 0.);
    dotk::nlp::resetField(original_control, vars, dotk::types::UZ);
    dotk::gtest::checkResults(*vars.mState, state);
    dotk::gtest::checkResults(*vars.mControl, original_control);

    // TEST 4: DERIVATIVE W.R.T. CONTROL_STATE
    vars.mControl->update(1., control, 0.);
    dotk::nlp::resetField(original_state, vars, dotk::types::ZU);
    dotk::gtest::checkResults(*vars.mState, original_state);
    dotk::gtest::checkResults(*vars.mControl, control);

    // TEST 5: DERIVATIVE W.R.T. CONTROL_CONTROL
    vars.mState->update(1., state, 0.);
    dotk::nlp::resetField(original_control, vars, dotk::types::ZZ);
    dotk::gtest::checkResults(*vars.mState, state);
    dotk::gtest::checkResults(*vars.mControl, original_control);

    // TEST 6: DERIVATIVE W.R.T. STATE_STATE
    vars.mControl->update(1., control, 0.);
    dotk::nlp::resetField(original_state, vars, dotk::types::UU);
    dotk::gtest::checkResults(*vars.mState, original_state);
    dotk::gtest::checkResults(*vars.mControl, control);
}

TEST(DOTk_NLP, perturbField)
{
    size_t num_states = 5;
    size_t num_controls = 2;

    dotk::StdVector<Real> state(num_states, 4.);
    dotk::StdVector<Real> control(num_controls, 1.);
    dotk::nlp::variables vars(state, control);

    dotk::StdVector<Real> state_perturbation_vector(num_states, 4.);
    dotk::StdVector<Real> control_perturbation_vector(num_controls, 1.);

    Real epsilon = 1e-3;

    // TEST 1: DERIVATIVE W.R.T. CONTROL
    dotk::StdVector<Real> control_gold(num_controls, 1.001);
    dotk::nlp::perturbField(epsilon, control_perturbation_vector, vars, dotk::types::Z);
    dotk::gtest::checkResults(*vars.mState, state);
    dotk::gtest::checkResults(*vars.mControl, control_gold);

    // TEST 2: DERIVATIVE W.R.T. STATE
    vars.mControl->update(1., control, 0.);
    dotk::StdVector<Real> state_gold(num_states, 4.004);
    dotk::nlp::perturbField(epsilon, state_perturbation_vector, vars, dotk::types::U);
    dotk::gtest::checkResults(*vars.mState, state_gold);
    dotk::gtest::checkResults(*vars.mControl, control);

    // TEST 3: DERIVATIVE W.R.T. STATE_CONTROL
    vars.mState->update(1., state, 0.);
    dotk::nlp::perturbField(epsilon, control_perturbation_vector, vars, dotk::types::UZ);
    dotk::gtest::checkResults(*vars.mState, state);
    dotk::gtest::checkResults(*vars.mControl, control_gold);

    // TEST 4: DERIVATIVE W.R.T. CONTROL_STATE
    vars.mControl->update(1., control, 0.);
    dotk::nlp::perturbField(epsilon, state_perturbation_vector, vars, dotk::types::ZU);
    dotk::gtest::checkResults(*vars.mState, state_gold);
    dotk::gtest::checkResults(*vars.mControl, control);

    // TEST 5: DERIVATIVE W.R.T. CONTROL_CONTROL
    vars.mState->update(1., state, 0.);
    dotk::nlp::perturbField(epsilon, control_perturbation_vector, vars, dotk::types::ZZ);
    dotk::gtest::checkResults(*vars.mState, state);
    dotk::gtest::checkResults(*vars.mControl, control_gold);

    // TEST 6: DERIVATIVE W.R.T. STATE_STATE
    vars.mControl->update(1., control, 0.);
    dotk::nlp::perturbField(epsilon, state_perturbation_vector, vars, dotk::types::UU);
    dotk::gtest::checkResults(*vars.mState, state_gold);
    dotk::gtest::checkResults(*vars.mControl, control);
}

TEST(DOTk_DiagnosticsObjectiveTypeNP, checkPartialDerivativeState)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjectiveNLP> objective(new dotk::DOTk_NocedalAndWrightObjectiveNLP());
    dotk::DOTk_DiagnosticsObjectiveTypeNP operators(objective);

    srand(0);
    // TEST 1: CHECK DERIVATIVE OPERATOR
    std::ostringstream msg;
    operators.setFiniteDifferenceDiagnosticsSuperScripts(0, 8);
    operators.checkPartialDerivativeState(state, control, msg);
    EXPECT_EQ(dotk::types::STATE, operators.getCodomain());
    EXPECT_EQ(dotk::types::U, operators.getDerivativeType());

    std::cout << msg.str().c_str() << std::flush;

    // TEST 2: INVALID CODOMAIN DIMENSIONS
    msg.str("");
    num_states = 0;
    dotk::StdVector<Real> empty_state_data(num_states, 1.);
    dotk::DOTk_State empty_state(empty_state_data);
    operators.checkPartialDerivativeState(empty_state, control, msg);

    std::string gold;
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension codomain in "
                "dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeState, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsObjectiveTypeNP, checkPartialDerivativeControl)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjectiveNLP> objective(new dotk::DOTk_NocedalAndWrightObjectiveNLP());
    dotk::DOTk_DiagnosticsObjectiveTypeNP operators(objective);

    srand(0);
    std::ostringstream msg;
    operators.checkPartialDerivativeControl(state, control, msg);
    EXPECT_EQ(dotk::types::CONTROL, operators.getCodomain());
    EXPECT_EQ(dotk::types::Z, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension codomain in "
                     "dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsObjectiveTypeNP, checkPartialDerivativeStateState)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjectiveNLP> objective(new dotk::DOTk_NocedalAndWrightObjectiveNLP());
    dotk::DOTk_DiagnosticsObjectiveTypeNP operators(objective);

    srand(0);
    // TEST 1: CHECK DERIVATIVE OPERATOR
    std::ostringstream msg;
    operators.setFiniteDifferenceDiagnosticsSuperScripts(0, 8);
    operators.checkPartialDerivativeStateState(state, control, msg);
    EXPECT_EQ(dotk::types::STATE, operators.getCodomain());
    EXPECT_EQ(dotk::types::UU, operators.getDerivativeType());

    std::cout << msg.str().c_str() << std::flush;

    // TEST 2: INVALID CODOMAIN DIMENSIONS
    msg.str("");
    num_states = 0;
    dotk::StdVector<Real> empty_state_data(num_states, 1.);
    dotk::DOTk_State empty_state(empty_state_data);
    operators.checkPartialDerivativeStateState(empty_state, control, msg);

    std::string gold;
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension codomain in "
                "dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeStateState, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsObjectiveTypeNP, checkPartialDerivativeStateControl)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjectiveNLP> objective(new dotk::DOTk_NocedalAndWrightObjectiveNLP());
    dotk::DOTk_DiagnosticsObjectiveTypeNP operators(objective);

    srand(0);
    std::ostringstream msg;
    operators.checkPartialDerivativeStateControl(state, control, msg);
    EXPECT_EQ(dotk::types::STATE, operators.getCodomain());
    EXPECT_EQ(dotk::types::UZ, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension domain in "
                     "dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeStateControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsObjectiveTypeNP, checkPartialDerivativeControlControl)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjectiveNLP> objective(new dotk::DOTk_NocedalAndWrightObjectiveNLP());
    dotk::DOTk_DiagnosticsObjectiveTypeNP operators(objective);

    srand(0);
    std::ostringstream msg;
    operators.checkPartialDerivativeControlControl(state, control, msg);
    EXPECT_EQ(dotk::types::CONTROL, operators.getCodomain());
    EXPECT_EQ(dotk::types::ZZ, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension codomain in "
                     "dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeControlControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsObjectiveTypeNP, checkPartialDerivativeControlState)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjectiveNLP> objective(new dotk::DOTk_NocedalAndWrightObjectiveNLP());
    dotk::DOTk_DiagnosticsObjectiveTypeNP operators(objective);

    srand(0);
    std::ostringstream msg;
    operators.checkPartialDerivativeControlState(state, control, msg);
    EXPECT_EQ(dotk::types::CONTROL, operators.getCodomain());
    EXPECT_EQ(dotk::types::ZU, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension codomain in "
                     "dotk::DOTk_DiagnosticsObjectiveTypeNP.checkPartialDerivativeControlState, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}


}
