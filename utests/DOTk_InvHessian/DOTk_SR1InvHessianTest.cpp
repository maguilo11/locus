/*
 * DOTk_SR1InvHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_SR1InvHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSR1InvHessianTest
{

TEST(DOTk_SR1InvHessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_SR1InvHessian invhess(vec);
    EXPECT_EQ(dotk::types::SR1_INV_HESS, invhess.getInvHessianType());

    vec->fill(2);
    mng->setOldPrimal(*vec);
    (*vec)[0] = 1.;
    (*vec)[1] = 3.;
    mng->setNewPrimal(*vec);

    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), vec);
    mng->setOldGradient(*vec);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), vec);
    mng->setNewGradient(*vec);

    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);

    // EVEN CASE
    invhess.apply(mng, mng->getTrialStep(), vec);
    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.3996502748063;
    (*gold)[1] = -0.1998251374032;
    dotk::gtest::checkResults(*vec, *gold);

    // ODD CASE
    invhess.setNumOptimizationItrDone(1);
    invhess.apply(mng, mng->getTrialStep(), vec);
    (*gold)[0] = 0.4996876951905;
    (*gold)[1] = -0.249843847595;
    dotk::gtest::checkResults(*vec, *gold);
}

TEST(DOTk_SR1InvHessian, getInvHessian)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_SR1InvHessian invhess(vec);
    EXPECT_EQ(dotk::types::SR1_INV_HESS, invhess.getInvHessianType());

    (*invhess.getDeltaPrimal())[0] = -1.;
    (*invhess.getDeltaPrimal())[1] = 1.;
    (*invhess.getDeltaGrad())[0] = -2402;
    (*invhess.getDeltaGrad())[1] = 800.;
    (*vec)[0] = 800.;
    (*vec)[1] = -400.;
    mng->setTrialStep(*vec);

    // EVEN CASE
    invhess.getInvHessian(mng->getTrialStep(), vec);
    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.3996502748063;
    (*gold)[1] = -0.1998251374032;
    dotk::gtest::checkResults(*vec, *gold);

    // ODD CASE
    invhess.setNumOptimizationItrDone(1);
    invhess.getInvHessian(mng->getTrialStep(), vec);
    (*gold)[0] = 0.4996876951905;
    (*gold)[1] = -0.249843847595;
    dotk::gtest::checkResults(*vec, *gold);
}

}
