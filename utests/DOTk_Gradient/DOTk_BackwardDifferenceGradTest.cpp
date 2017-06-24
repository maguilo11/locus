/*
 * DOTk_BackwardDifferenceGradTest.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_BackwardDifferenceGrad.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"

namespace DOTkBackwardDifferenceGradTest
{

TEST(BackwardDifferenceGrad, getFiniteDiffPerturbationVec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_BackwardDifferenceGrad grad(mng.getNewGradient());

    EXPECT_EQ(2, grad.getFiniteDiffPerturbationVec()->size());
    primal->control()->fill(1e-6);
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
}

TEST(BackwardDifferenceGrad, setFiniteDiffPerturbationVec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_BackwardDifferenceGrad grad(mng.getNewGradient());

    primal->control()->fill(1e-6);
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
    primal->control()->fill(1234);
    grad.setFiniteDiffPerturbationVec(*primal->control());
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
    primal->control()->fill(8976);
    grad.setFiniteDiffPerturbationVec(*primal->control());
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
}

TEST(BackwardDifferenceGrad, gradient)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_BackwardDifferenceGrad grad(mng.getNewGradient());
    EXPECT_EQ(dotk::types::BACKWARD_DIFF_GRAD, grad.type());

    mng.setNewPrimal(*primal->control());
    Real objective_function_value = mng.getRoutinesMng()->objective(mng.getNewPrimal());
    mng.setNewObjectiveFunctionValue(objective_function_value);
    grad.gradient(&mng);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold, 1e-2);
}

TEST(BackwardDifferenceGrad, getGradient)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_BackwardDifferenceGrad grad(mng.getNewGradient());
    EXPECT_EQ(dotk::types::BACKWARD_DIFF_GRAD, grad.type());

    mng.setNewPrimal(*primal->control());
    Real objective_function_value = mng.getRoutinesMng()->objective(mng.getNewPrimal());
    mng.setNewObjectiveFunctionValue(objective_function_value);
    grad.getGradient(objective_function_value, mng.getRoutinesMng(), mng.getNewPrimal(), mng.getNewGradient());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold, 1e-2);
}

}
