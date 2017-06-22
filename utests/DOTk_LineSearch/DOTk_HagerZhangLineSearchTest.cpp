/*
 * DOTk_HagerZhangLineSearchTest.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_HagerZhangLineSearch.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"

namespace DOTkHagerZhangLineSearchTest
{

TEST(DOTk_HagerZhangLineSearchTest, setAndGetStepFunctions)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_HagerZhangLineSearch step(mng.getTrialStep());

    Real tol = 1e-6;

    // Default
    EXPECT_EQ(5, step.getMaxShrinkIntervalIterations());
    // Test
    step.setMaxShrinkIntervalIterations(3);
    EXPECT_EQ(3, step.getMaxShrinkIntervalIterations());

    // Default
    EXPECT_EQ(0.1, step.getConstant());
    // Test
    step.setConstant(0.134);
    EXPECT_EQ(0.134, step.getConstant());

    // Default
    EXPECT_EQ(0.9, step.getCurvatureConstant());
    // Test
    step.setCurvatureConstant(0.133);
    EXPECT_EQ(0.133, step.getCurvatureConstant());

    // Default
    EXPECT_NEAR(0., step.getStepInterval(dotk::types::LOWER_BOUND), tol);
    EXPECT_NEAR(0., step.getStepInterval(dotk::types::UPPER_BOUND), tol);
    // Test
    step.setStepInterval(dotk::types::LOWER_BOUND, 1.1);
    step.setStepInterval(dotk::types::UPPER_BOUND, 2.3);
    EXPECT_NEAR(1.1, step.getStepInterval(dotk::types::LOWER_BOUND), tol);
    EXPECT_NEAR(2.3, step.getStepInterval(dotk::types::UPPER_BOUND), tol);

    // Default
    EXPECT_EQ(0., step.getOldObjectiveFunctionValue());
    // Test
    step.setOldObjectiveFunctionValue(0.1234);
    EXPECT_EQ(0.1234, step.getOldObjectiveFunctionValue());

    // Default
    Real gold_value = 2 / 3;
    EXPECT_EQ(gold_value, step.getObjectiveFunctionErrorEstimateParameter());
    // Test
    step.setObjectiveFunctionErrorEstimateParameter(0.75);
    EXPECT_EQ(0.75, step.getObjectiveFunctionErrorEstimateParameter());

    // Default
    EXPECT_EQ(0.5, step.getBisectionUpdateParameter());
    // Test
    step.setBisectionUpdateParameter(0.535);
    EXPECT_EQ(0.535, step.getBisectionUpdateParameter());

    // Default
    EXPECT_EQ(1e-6, step.getIntervalUpdateParameter());
    // Test
    step.setIntervalUpdateParameter(1e-3);
    EXPECT_EQ(1e-3, step.getIntervalUpdateParameter());
}

TEST(DOTk_HagerZhangLineSearchTest, secantStep)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::DOTk_HagerZhangLineSearch step(mng->getTrialStep());
    EXPECT_EQ(dotk::types::LINE_SEARCH_HAGER_ZHANG, step.type());

    mng->getTrialStep()->fill(0.1);
    mng->getOldGradient()->fill(0.);
    mng->getOldPrimal()->fill(1.5);
    step.setStepInterval(dotk::types::LOWER_BOUND, 1e-4);
    step.setStepInterval(dotk::types::UPPER_BOUND, 1.);
    step.secantStep(mng->getTrialStep(), mng->getOldPrimal(), mng->getNewPrimal(), mng->getOldGradient(), mng);
}

}
