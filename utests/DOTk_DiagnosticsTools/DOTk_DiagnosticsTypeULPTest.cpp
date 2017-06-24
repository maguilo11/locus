/*
 * DOTk_DiagnosticsTypeULPTest.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Control.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_EqualityTypeLP.hpp"
#include "DOTk_ObjectiveTypeLP.hpp"
#include "DOTk_InequalityTypeLP.hpp"
#include "DOTk_DiagnosticsTypeLP.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace DOTkDiagnosticsTypeULPTest
{

TEST(DOTk_DerivativeDiagnosticsTool, checkCodomainDimensions)
{
    dotk::DOTk_DerivativeDiagnosticsTool diagnostic;

    std::ostringstream msg;
    size_t num_controls = 0;
    std::shared_ptr< dotk::Vector<Real> > field = std::make_shared<dotk::StdVector<Real>>(num_controls);
    std::string function_name("FUNCTION NAME");
    diagnostic.checkCodomainDimensions(field, function_name, msg);

    std::string gold("DOTk ERROR: ZERO dimension codomain in FUNCTION NAME, EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DerivativeDiagnosticsTool, checkDomainDimensions)
{
    dotk::DOTk_DerivativeDiagnosticsTool diagnostic;

    std::ostringstream msg;
    size_t num_controls = 0;
    std::shared_ptr< dotk::Vector<Real> > field = std::make_shared<dotk::StdVector<Real>>(num_controls);
    std::string function_name("FUNCTION NAME");
    diagnostic.checkDomainDimensions(field, function_name, msg);

    std::string gold("DOTk ERROR: ZERO dimension domain in FUNCTION NAME, EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsTypeLP, printFiniteDifferenceDiagnostics)
{
    std::shared_ptr<dotk::DOTk_Rosenbrock> operators = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_DiagnosticsTypeLP diagnostic(operators);
    EXPECT_FALSE(diagnostic.willFiniteDifferenceDiagnosticsBePrinted());

    diagnostic.printFiniteDifferenceDiagnostics(true);
    EXPECT_TRUE(diagnostic.willFiniteDifferenceDiagnosticsBePrinted());
}

TEST(DOTk_DiagnosticsTypeLP, checkFirstDerivative)
{
    std::shared_ptr<dotk::DOTk_Rosenbrock> operators = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_DiagnosticsTypeLP objective_function(operators);

    srand(0);
    // TEST 1: CHECK DERIVATIVE OPERATOR
    std::ostringstream msg;
    size_t num_controls = 2;
    dotk::StdVector<Real> data(num_controls, 0.);
    dotk::DOTk_Control control(data);
    objective_function.checkObjectiveGradient(control, msg);
    EXPECT_EQ(dotk::types::CONTROL, objective_function.getCodomain());

    std::cout << msg.str().c_str() << std::flush;

    // TEST 2: INVALID CODOMAIN DIMENSIONS
    msg.str("");
    num_controls = 0;
    dotk::StdVector<Real> empty_data(num_controls, 2.);
    dotk::DOTk_Control empty_control(empty_data);
    objective_function.checkObjectiveGradient(empty_control, msg);

    std::string gold;
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension codomain in "
                "dotk::DOTk_DiagnosticsTypeLP.checkObjectiveGradient, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

TEST(DOTk_DiagnosticsTypeLP, checkSecondDerivative)
{
    std::shared_ptr<dotk::DOTk_Rosenbrock> operators = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_DiagnosticsTypeLP objective_function(operators);

    srand(0);
    // TEST 1: CHECK DERIVATIVE OPERATOR
    std::ostringstream msg;
    size_t num_controls = 2;
    dotk::StdVector<Real> data(num_controls, 2.);
    dotk::DOTk_Control control(data);
    objective_function.checkObjectiveHessian(control, msg);
    EXPECT_EQ(dotk::types::CONTROL, objective_function.getCodomain());

    std::cout << msg.str().c_str() << std::flush;

    // TEST 2: INVALID CODOMAIN DIMENSIONS
    msg.str("");
    num_controls = 0;
    dotk::StdVector<Real> empty_data(num_controls, 2.);
    dotk::DOTk_Control empty_control(empty_data);
    objective_function.checkObjectiveHessian(empty_control, msg);

    std::string gold;
    gold.clear();
    gold.assign("DOTk ERROR: ZERO dimension codomain in "
                "dotk::DOTk_DiagnosticsTypeLP.checkObjectiveHessian, "
                "EXIT FUNCTION\n");
    EXPECT_STREQ(gold.c_str(), msg.str().c_str());
}

}
