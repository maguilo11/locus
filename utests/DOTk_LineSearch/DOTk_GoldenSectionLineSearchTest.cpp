/*
 * DOTk_GoldenSectionLineSearchTest.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_GoldenSectionLineSearch.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace DOTkGoldenSectionLineSearchTest
{

TEST(GoldenSectionLineSearchTest, step)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    dotk::DOTk_GoldenSectionLineSearch step(mng->getTrialStep());
    EXPECT_EQ(dotk::types::GOLDENSECTION, step.type());

    // Test #1 (test decrease condition)
    (*mng->getOldPrimal())[0] = 1.997506234413967;
    (*mng->getOldPrimal())[1] = 3.990024937655861;
    Real old_objective_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(old_objective_value);

    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());

    mng->getTrialStep()->update(-1., *mng->getOldGradient(), 0.);
    mng->getNewPrimal()->update(1., *mng->getOldPrimal(), 0.);
    mng->getNewPrimal()->update(1., *mng->getTrialStep(), 1.);
    Real new_objective_value = mng->getRoutinesMng()->objective(mng->getNewPrimal());
    mng->setNewObjectiveFunctionValue(new_objective_value);

    step.step(mng);

    const Real tol = 1e-6;
    EXPECT_NEAR(0.00031313417779212587, step.getStepSize(), tol);
    EXPECT_EQ(32, step.getNumLineSearchItrDone());
}

}
