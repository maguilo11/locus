/*
 * DOTk_ParallelBackwardDiffGradTest.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ParallelBackwardDiffGrad.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace
{

TEST(ParallelBackwardDiffGrad, getFiniteDiffPerturbationVec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelBackwardDiffGrad grad(mng.getNewGradient());

    EXPECT_EQ(2, grad.getFiniteDiffPerturbationVec()->size());
    primal->control()->fill(1e-6);
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
}

TEST(ParallelBackwardDiffGrad, setFiniteDiffPerturbationVec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelBackwardDiffGrad grad(mng.getNewGradient());

    primal->control()->fill(1e-6);
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
    primal->control()->fill(1234);
    grad.setFiniteDiffPerturbationVec(*primal->control());
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
    primal->control()->fill(8976);
    grad.setFiniteDiffPerturbationVec(*primal->control());
    dotk::gtest::checkResults(*grad.getFiniteDiffPerturbationVec(), *primal->control());
}

TEST(ParallelBackwardDiffGrad, gradient)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelBackwardDiffGrad grad(mng.getNewGradient());
    EXPECT_EQ(dotk::types::PARALLEL_BACKWARD_DIFF_GRAD, grad.type());

    mng.setNewPrimal(*primal->control());
    Real objective_function_value = mng.getRoutinesMng()->objective(mng.getNewPrimal());
    mng.setNewObjectiveFunctionValue(objective_function_value);
    grad.gradient(&mng);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold, 1e-2);
}

TEST(ParallelBackwardDiffGrad, getGradient)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);
    dotk::DOTk_ParallelBackwardDiffGrad grad(mng.getNewGradient());
    EXPECT_EQ(dotk::types::PARALLEL_BACKWARD_DIFF_GRAD, grad.type());

    mng.setNewPrimal(*primal->control());
    Real fval = mng.getRoutinesMng()->objective(mng.getNewPrimal());
    grad.getGradient(fval, mng.getRoutinesMng(), mng.getNewPrimal(), mng.getNewGradient());

    std::tr1::shared_ptr<dotk::vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = 1602.;
    (*gold)[1] = -400.;
    dotk::gtest::checkResults(*mng.getNewGradient(), *gold, 1e-2);
}

}
