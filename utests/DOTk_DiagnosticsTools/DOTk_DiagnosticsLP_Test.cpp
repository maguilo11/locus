/*
 * DOTk_DiagnosticsLP_Test.cpp
 *
 *  Created on: Mar 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_DiagnosticsTypeLP.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_NocedalAndWrightEquality.hpp"
#include "DOTk_NocedalAndWrightObjective.hpp"
#include "DOTk_GcmmaTestObjectiveFunction.hpp"

namespace DOTkDiagnosticsLPTest
{

TEST(DOTk_DiagnosticsTypeELP, checkObjectiveGradient)
{
    size_t num_dual = 3;
    dotk::StdVector<Real> dual_data(num_dual, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::gtools::generateRandomVector(dual.data());

    size_t num_state = 5;
    dotk::StdVector<Real> state_data(num_state, 1.);
    dotk::DOTk_State state(state_data);

    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality());
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective());
    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);

    srand(0);
    std::ostringstream msg;
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(0, 8);
    diagnostics.checkObjectiveGradient(state, msg);

    std::cout << msg.str().c_str() << std::flush;
}

TEST(DOTk_DiagnosticsTypeELP, checkObjectiveHessian)
{
    size_t num_dual = 3;
    dotk::StdVector<Real> dual_data(num_dual, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::gtools::generateRandomVector(dual.data());

    size_t num_state = 5;
    dotk::StdVector<Real> state_data(num_state, 1.);
    dotk::DOTk_State state(state_data);

    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality());
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective());
    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);

    srand(0);
    std::ostringstream msg;
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(0, 8);
    diagnostics.checkObjectiveHessian(state, msg);

    std::cout << msg.str().c_str() << std::flush;
}

TEST(DOTk_DiagnosticsTypeELP, checkEqualityConstraintJacobian)
{
    size_t num_state = 5;
    dotk::DOTk_State state;
    state.allocateSerialVector(num_state, 1.);

    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality());
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective());
    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);

    srand(0);
    std::ostringstream msg;
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(-4, 3);
    diagnostics.checkEqualityConstraintJacobian(state, msg);

    std::cout << msg.str().c_str() << std::flush;
}

TEST(DOTk_DiagnosticsTypeELP, checkEqualityConstraintAdjointJacobian)
{
    size_t num_dual = 3;
    dotk::StdVector<Real> dual_data(num_dual, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::gtools::generateRandomVector(dual.data());

    size_t num_state = 5;
    dotk::StdVector<Real> state_data(num_state, 1.);
    dotk::DOTk_State state(state_data);

    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality());
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective());
    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);

    srand(0);
    std::ostringstream msg;
    diagnostics.checkEqualityConstraintAdjointJacobian(state, dual, msg);

    std::string gold("The absolute difference between (dual_dot_first_derivative_times_direction) and"
            " (adjoint_first_derivative_times_dual_dot_direction) = 6.661338e-16\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsTypeELP, checkEqualityConstraintJacobianDerivative)
{
    size_t num_dual = 3;
    dotk::StdVector<Real> dual_data(num_dual, 0.);
    dotk::DOTk_Dual dual(dual_data);
    dotk::gtools::generateRandomVector(dual.data());

    size_t num_state = 5;
    dotk::StdVector<Real> state_data(num_state, 1.);
    dotk::DOTk_State state(state_data);

    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality());
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective());
    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective, equality);

    srand(0);
    std::ostringstream msg;
    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(-4, 3);
    diagnostics.checkEqualityConstraintJacobianDerivative(state, dual, msg);

    std::cout << msg.str().c_str() << std::flush;
}

TEST(DOTk_DiagnosticsTypeLP, checkGcmmaTestObjectiveFunctionFirstDerivative)
{
    size_t dim = 5;
    dotk::DOTk_Control control;
    control.allocateSerialVector(dim, 1.);

    srand(0);
    std::ostringstream msg;
    std::tr1::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction());
    dotk::DOTk_DiagnosticsTypeLP diagnostics(objective);

    diagnostics.setFiniteDifferenceDiagnosticsSuperScripts(-2, 6);
    diagnostics.checkObjectiveGradient(control, msg);

    std::cout << msg.str().c_str() << std::flush;
}

}
