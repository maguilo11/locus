/*
 * DOTk_GradientProjectionMethodTest.cpp
 *
 *  Created on: Sep 11, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_BealeObjective.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_GradientProjectionMethod.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace
{

TEST(GradientProjectionMethod, Rosenbrock_GoldsteinLineSearch)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(num_controls, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();

    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-4);
    EXPECT_EQ(2629, algorithm.getIterationCount());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, algorithm.getStoppingCriterion());
}

TEST(GradientProjectionMethod, Beale_GoldsteinLineSearch)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(num_controls, 1);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();

    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-4);
    EXPECT_EQ(397, algorithm.getIterationCount());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, algorithm.getStoppingCriterion());
}

TEST(GradientProjectionMethod, Beale_CubicLineSearch)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(num_controls, 1);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setCubicLineSearch(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();

    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-4);
    EXPECT_EQ(363, algorithm.getIterationCount());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, algorithm.getStoppingCriterion());
}

TEST(GradientProjectionMethod, Beale_ArmijoLineSearch)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(num_controls, 1);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();

    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-4);
    EXPECT_EQ(517, algorithm.getIterationCount());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, algorithm.getStoppingCriterion());
}

}
