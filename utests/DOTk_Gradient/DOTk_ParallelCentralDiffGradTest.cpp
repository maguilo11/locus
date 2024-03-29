/*
 * DOTk_ParallelCentralDiffGradTest.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ParallelCentralDiffGrad.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkParallelCentralDiffGradTest
{

TEST(ParallelCentralDiffGrad, getFiniteDiffPerturbationVec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelCentralDiffGrad grad(mng.getNewGradient());

    EXPECT_EQ(2, grad.getFiniteDiffPerturbationVec()->size());
    primal->control()->fill(1e-6);
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
}

TEST(ParallelCentralDiffGrad, setFiniteDiffPerturbationVec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelCentralDiffGrad grad(mng.getNewGradient());

    primal->control()->fill(1e-6);
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
    primal->control()->fill(1234);
    grad.setFiniteDiffPerturbationVec(*primal->control());
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
    primal->control()->fill(8976);
    grad.setFiniteDiffPerturbationVec(*primal->control());
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
}

TEST(ParallelCentralDiffGrad, gradient)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelCentralDiffGrad grad(mng.getNewGradient());
    EXPECT_EQ(dotk::types::PARALLEL_CENTRAL_DIFF_GRAD, grad.type());

    mng.setNewPrimal(*primal->control());
    Real objective_function_value = mng.getRoutinesMng()->objective(mng.getNewPrimal());
    mng.setNewObjectiveFunctionValue(objective_function_value);
    grad.gradient(&mng);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold, 1e-6);
}

TEST(ParallelCentralDiffGrad, getGradient)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelCentralDiffGrad grad(mng.getNewGradient());
    EXPECT_EQ(dotk::types::PARALLEL_CENTRAL_DIFF_GRAD, grad.type());

    mng.setNewPrimal(*primal->control());
    Real fval = mng.getRoutinesMng()->objective(mng.getNewPrimal());
    grad.getGradient(fval, mng.getRoutinesMng(), mng.getNewPrimal(), mng.getNewGradient());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold, 1e-6);
}

}
