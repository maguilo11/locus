/*
 * DOTk_DiagnosticsEqualityTypeNPTest.cpp
 *
 *  Created on: Dec 1, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_EqualityTypeNP.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"
#include "DOTk_DiagnosticsEqualityTypeNP.hpp"

#include "DOTk_NocedalAndWrightEqualityNLP.hpp"

namespace DOTkDiagnosticsEqualityTypeNPTest
{

TEST(DOTk_DiagnosticsEqualityTypeNP, checkPartialDerivativeState)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    srand(0);
    // TEST 1: CHECK DERIVATIVE OPERATOR
    std::ostringstream msg;
    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);
    operators.checkPartialDerivativeState(state, control, msg);
    EXPECT_EQ(dotk::types::STATE, operators.getCodomain());
    EXPECT_EQ(dotk::types::U, operators.getDerivativeType());

    std::cout << msg.str().c_str() << std::flush;

    // TEST 2: INVALID CODOMAIN DIMENSIONS
    msg.str("");
    num_states = 0;
    dotk::StdVector<Real> empty_state_data(num_states, 0.);
    dotk::DOTk_State empty_state(empty_state_data);
    operators.checkPartialDerivativeState(empty_state, control, msg);

    std::string gold;
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension codomain in "
                "dotk::DOTk_DiagnosticsEqualityTypeNP.checkPartialDerivativeState, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkPartialDerivativeControl)
{
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    operators.checkPartialDerivativeControl(state, control, msg);
    EXPECT_EQ(dotk::types::CONTROL, operators.getCodomain());
    EXPECT_EQ(dotk::types::Z, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension codomain in "
                     "dotk::DOTk_DiagnosticsEqualityTypeNP.checkPartialDerivativeControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkAdjointPartialDerivativeState)
{
    size_t num_duals = 3;
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> dual_data(num_duals, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    dotk::gtools::generateRandomVector(dual.data());

    // TEST 1: CHECK DERIVATIVE OPERATOR
    Real absolute_difference = operators.checkAdjointPartialDerivativeState(state, control, dual, msg);
    EXPECT_EQ(dotk::types::DUAL, operators.getCodomain());
    EXPECT_EQ(dotk::types::U, operators.getDerivativeType());

    Real tolerance = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), absolute_difference, tolerance);
    std::string gold("The absolute difference between (dual_dot_first_derivative_times_direction) and "
                     "(adjoint_first_derivative_times_dual_dot_direction) = 2.220446e-16\n");

    EXPECT_STREQ(gold.c_str(), msg.str().c_str());

    // TEST 2: INVALID CODOMAIN DIMENSIONS
    msg.str("");
    num_states = 0;
    dotk::StdVector<Real> empty_state_data(num_states, 0.);
    dotk::DOTk_State empty_state(empty_state_data);
    absolute_difference = operators.checkAdjointPartialDerivativeState(empty_state, control, dual, msg);

    EXPECT_NEAR(-1., absolute_difference, tolerance);
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension domain in "
                "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeState, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkAdjointPartialDerivativeControl)
{
    size_t num_duals = 3;
    dotk::DOTk_Dual dual;
    dual.allocateSerialVector(num_duals, 0.);

    size_t num_states = 5;
    dotk::DOTk_State state;
    state.allocateSerialVector(num_states, 1.);

    size_t num_controls = 0;
    dotk::DOTk_Control control;
    control.allocateSerialVector(num_controls, 0.);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    dotk::gtools::generateRandomVector(dual.data());
    Real absolute_difference = operators.checkAdjointPartialDerivativeControl(state, control, dual, msg);
    EXPECT_EQ(dotk::types::DUAL, operators.getCodomain());
    EXPECT_EQ(dotk::types::Z, operators.getDerivativeType());

    Real tolerance = 1e-8;
    EXPECT_NEAR(-1., absolute_difference, tolerance);

    std::string gold("DOTk ERROR: ZERO dimension domain in "
                     "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkAdjointPartialDerivativeControlControl)
{
    size_t num_duals = 3;
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> dual_data(num_duals, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    dotk::gtools::generateRandomVector(dual.data());
    operators.checkAdjointPartialDerivativeControlControl(state, control, dual, msg);
    EXPECT_EQ(dotk::types::CONTROL, operators.getCodomain());
    EXPECT_EQ(dotk::types::ZZ, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension codomain in "
                     "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControlControl, "
                     "EXIT FUNCTION\n"
                     "DOTk ERROR: ZERO dimension domain in "
                     "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControlControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkAdjointPartialDerivativeControlState)
{
    size_t num_duals = 3;
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> dual_data(num_duals, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    dotk::gtools::generateRandomVector(dual.data());
    operators.checkAdjointPartialDerivativeControlState(state, control, dual, msg);
    EXPECT_EQ(dotk::types::CONTROL, operators.getCodomain());
    EXPECT_EQ(dotk::types::ZU, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension codomain in "
                     "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeControlState, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkAdjointPartialDerivativeStateState)
{
    size_t num_duals = 3;
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> dual_data(num_duals, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    dotk::gtools::generateRandomVector(dual.data());

    // TEST 1: CHECK DERIVATIVE OPERATOR
    operators.checkAdjointPartialDerivativeStateState(state, control, dual, msg);
    EXPECT_EQ(dotk::types::STATE, operators.getCodomain());
    EXPECT_EQ(dotk::types::UU, operators.getDerivativeType());

    std::cout << msg.str().c_str() << std::flush;

    // TEST 2: INVALID DOMAIN AND CODOMAIN DIMENSIONS
    msg.str("");
    num_states = 0;
    dotk::StdVector<Real> empty_state_data(num_states, 0.);
    dotk::DOTk_State empty_state(empty_state_data);
    operators.checkAdjointPartialDerivativeStateState(empty_state, control, dual, msg);

    std::string gold;
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension codomain in "
                "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeStateState, "
                "EXIT FUNCTION\n"
                "DOTk ERROR: ZERO dimension domain in "
                "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeStateState, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());

}

TEST(DOTk_DiagnosticsEqualityTypeNP, checkAdjointPartialDerivativeStateControl)
{
    size_t num_duals = 3;
    size_t num_states = 5;
    size_t num_controls = 0;
    dotk::StdVector<Real> dual_data(num_duals, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::StdVector<Real> state_data(num_states, 1.);
    dotk::DOTk_State state(state_data);
    dotk::StdVector<Real> control_data(num_controls, 0.);
    dotk::DOTk_Control control(control_data);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEqualityNLP> equality = std::make_shared<dotk::DOTk_NocedalAndWrightEqualityNLP>();
    dotk::DOTk_DiagnosticsEqualityTypeNP operators(equality);

    srand(0);
    std::ostringstream msg;
    dotk::gtools::generateRandomVector(dual.data());
    operators.checkAdjointPartialDerivativeStateControl(state, control, dual, msg);
    EXPECT_EQ(dotk::types::STATE, operators.getCodomain());
    EXPECT_EQ(dotk::types::UZ, operators.getDerivativeType());

    std::string gold("DOTk ERROR: ZERO dimension domain in "
                     "dotk::DOTk_DiagnosticsEqualityTypeNP.checkAdjointPartialDerivativeStateControl, "
                     "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

}
