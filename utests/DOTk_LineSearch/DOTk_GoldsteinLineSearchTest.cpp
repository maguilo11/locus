/*
 * DOTk_GoldsteinLineSearchTest.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_GoldsteinLineSearch.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"

namespace DOTkGoldsteinLineSearchTest
{

TEST(DOTk_GoldsteinLineSearchTest, getAndSetConstant)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    dotk::DOTk_GoldsteinLineSearch step(mng.getTrialStep());

    Real tol = 1e-8;
    EXPECT_NEAR(0.9, step.getConstant(), tol);
    step.setConstant(0.51);
    EXPECT_NEAR(0.51, step.getConstant(), tol);
}

TEST(DOTk_GoldsteinLineSearchTest, step)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::DOTk_GoldsteinLineSearch line_search(mng->getTrialStep());
    EXPECT_EQ(dotk::types::BACKTRACKING_GOLDSTEIN, line_search.type());

    mng->setOldPrimal(*primal->control());
    Real old_objective_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(old_objective_value);
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());

    mng->getTrialStep()->update(-1., *mng->getOldGradient(), 0.);
    mng->getNewPrimal()->update(1., *mng->getOldPrimal(), 0.);
    mng->getNewPrimal()->update(1., *mng->getTrialStep(), 1.);
    mng->getNewGradient()->update(1., *mng->getOldGradient(), 0.);

    Real new_objective_value = mng->getRoutinesMng()->objective(mng->getNewPrimal());
    mng->setNewObjectiveFunctionValue(new_objective_value);

    line_search.step(mng);

    const Real tol = 1e-6;
    EXPECT_NEAR(0.000244140625, line_search.getStepSize(), tol);
    EXPECT_EQ(13, line_search.getNumLineSearchItrDone());
}

}
