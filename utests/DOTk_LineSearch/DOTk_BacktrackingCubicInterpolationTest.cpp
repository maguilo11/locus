/*
 * DOTk_BacktrackingCubicInterpolationTest.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_BacktrackingCubicInterpolation.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"

namespace DOTkBacktrackingCubicInterpolationTest
{

TEST(DOTk_BacktrackingCubicInterpolationTest, step)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    dotk::DOTk_BacktrackingCubicInterpolation line_search(mng->getTrialStep());
    EXPECT_EQ(dotk::types::BACKTRACKING_CUBIC_INTRP, line_search.type());

    // Test #1 (test decrease condition)
    (*mng->getOldPrimal())[0] = 1.997506234413967;
    (*mng->getOldPrimal())[1] = 3.990024937655861;
    Real old_objective_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(old_objective_value);
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    (*mng->getTrialStep())[0] = -0.996267103929714;
    (*mng->getTrialStep())[1] = -3.980093283615503;
    mng->getNewGradient()->update(1., *mng->getOldGradient(), 0.);
    line_search.step(mng);
    const Real tol = 1e-6;
    EXPECT_NEAR(0.1, line_search.getStepSize(), tol);
    EXPECT_EQ(50, line_search.getMaxNumLineSearchItr());

    // Test #2 (test quadratic fit)
    (*mng->getOldPrimal())[0] = 1.897879524021002;
    (*mng->getOldPrimal())[1] = 3.592015609294339;
    old_objective_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(old_objective_value);
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    (*mng->getTrialStep())[0] = -0.300674706720995;
    (*mng->getTrialStep())[1] = -1.131357660149725;
    line_search.dotk::DOTk_LineSearch::setStepSize(1.);
    mng->getNewGradient()->update(1., *mng->getOldGradient(), 0.);
    line_search.step(mng);
    EXPECT_NEAR(0.30496794684811185, line_search.getStepSize(), tol);
    EXPECT_EQ(50, line_search.getMaxNumLineSearchItr());

    // Test #3 (test cubic interpolation)
    (*mng->getOldPrimal())[0] = -4.998739760554468;
    (*mng->getOldPrimal())[1] = 24.987397605545060;
    old_objective_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(old_objective_value);
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    (*mng->getTrialStep())[0] = 5.99683492222362;
    (*mng->getTrialStep())[1] = -59.95323273819937;
    line_search.dotk::DOTk_LineSearch::setStepSize(1.);
    mng->getNewGradient()->update(1., *mng->getOldGradient(), 0.);
    line_search.step(mng);
    EXPECT_NEAR(0.05, line_search.getStepSize(), tol);
    EXPECT_EQ(50, line_search.getMaxNumLineSearchItr());
}

}


