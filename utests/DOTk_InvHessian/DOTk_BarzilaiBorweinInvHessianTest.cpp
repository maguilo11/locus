/*
 * DOTk_BarzilaiBorweinInvHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_BarzilaiBorweinInvHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkBarzilaiBorweinInvHessianTest
{

TEST(DOTk_BarzilaiBorweinInvHessian, apply)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));


    mng->getOldPrimal()->fill(2);
    mng->getNewPrimal()->fill(1.5);
    (*mng->getTrialStep())[0] = -3;
    (*mng->getTrialStep())[1] = -1;
    std::tr1::shared_ptr<dotk::vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);

    // EVEN CASE
    std::tr1::shared_ptr<dotk::vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_BarzilaiBorweinInvHessian invhess(primal->control());
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, invhess.getInvHessianType());

    invhess.apply(mng, mng->getTrialStep(), vec);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -0.000974193776260;
    (*gold)[1] = -0.000324731258753;
    dotk::gtest::checkResults(*vec, *gold);

    // ODD CASE
    invhess.setNumOptimizationItrDone(1);
    invhess.apply(mng, mng->getTrialStep(), vec);
    (*gold)[0] = -0.003329633740289;
    (*gold)[1] = -0.001109877913429;
    dotk::gtest::checkResults(*vec, *gold);
}

TEST(DOTk_BarzilaiBorweinInvHessian, getInvHessian)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    std::tr1::shared_ptr<dotk::vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_BarzilaiBorweinInvHessian invhess(vec);
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, invhess.getInvHessianType());

    (*invhess.getDeltaPrimal())[0] = -0.5;
    (*invhess.getDeltaPrimal())[1] = -0.5;
    (*invhess.getDeltaGrad())[0] = -1151.;
    (*invhess.getDeltaGrad())[1] = 250.;
    (*mng->getTrialStep())[0] = -3;
    (*mng->getTrialStep())[1] = -1;

    // EVEN CASE
    invhess.getInvHessian(mng->getTrialStep(), vec);
    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -0.000974193776260;
    (*gold)[1] = -0.000324731258753;
    dotk::gtest::checkResults(*vec, *gold);

    // ODD CASE
    invhess.setNumOptimizationItrDone(1);
    invhess.getInvHessian(mng->getTrialStep(), vec);
    (*gold)[0] = -0.003329633740289;
    (*gold)[1] = -0.001109877913429;
    dotk::gtest::checkResults(*vec, *gold);
}

}
