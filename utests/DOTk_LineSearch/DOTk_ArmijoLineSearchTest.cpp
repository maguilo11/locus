/*
 * DOTk_ArmijoLineSearchTest.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ArmijoLineSearch.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"

namespace DOTkArmijoLineSearchTest
{

TEST(DOTk_ArmijoLineSearchTest, getAndSetFunctions)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>  mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::DOTk_ArmijoLineSearch step(mng->getTrialStep());

    Real tol = 1e-8;
    EXPECT_NEAR(0.5, step.getContractionFactor(), tol);
    step.setContractionFactor(0.1);
    EXPECT_NEAR(0.1, step.getContractionFactor(), tol);

    EXPECT_NEAR(1e-8, step.getStepStagnationTol(), tol);
    step.setStepStagnationTol(2e-3);
    EXPECT_NEAR(2e-3, step.getStepStagnationTol(), tol);

    EXPECT_NEAR(1e-4, step.getConstant(), tol);
    step.setConstant(0.53);
    EXPECT_NEAR(0.53, step.getConstant(), tol);

    EXPECT_NEAR(1., step.getStepSize(), tol);
    step.setStepSize(0.56);
    EXPECT_NEAR(0.56, step.getStepSize(), tol);

    EXPECT_EQ(0, step.getNumLineSearchItrDone());
    step.setNumLineSearchItrDone(3);
    EXPECT_EQ(3, step.getNumLineSearchItrDone());

    EXPECT_EQ(50, step.getMaxNumLineSearchItr());
    step.setMaxNumLineSearchItr(11);
    EXPECT_EQ(11, step.getMaxNumLineSearchItr());
}

TEST(DOTk_ArmijoLineSearchTest, step)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>  mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::DOTk_ArmijoLineSearch line_search(mng->getTrialStep());

    const Real tolerance = 1e-6;
    EXPECT_EQ(dotk::types::BACKTRACKING_ARMIJO, line_search.type());

    (*mng->getOldPrimal())[0] = 2.;
    (*mng->getOldPrimal())[1] = 3.;
    (*mng->getNewPrimal())[0] = 2.;
    (*mng->getNewPrimal())[1] = 3.;
    Real old_objective_func_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(old_objective_func_value);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), mng->getNewGradient());

    mng->getTrialStep()->update(-1., *mng->getNewGradient(), 0.);
    mng->getNewPrimal()->update(1., *mng->getTrialStep(), 1.);

    Real new_objective_value = mng->getRoutinesMng()->objective(mng->getNewGradient());
    mng->setNewObjectiveFunctionValue(new_objective_value);
    line_search.step(mng);

    EXPECT_NEAR(0.00048828125, line_search.getStepSize(), tolerance);
    EXPECT_EQ(50, line_search.getMaxNumLineSearchItr());
}

}
