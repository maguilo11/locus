/*
 * DOTk_LSR1InvHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LSR1InvHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLSR1InvHessianTest
{

TEST(DOTk_LSR1InvHessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    size_t secant_storage = 2;
    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_LSR1InvHessian invhess(vec, secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, invhess.getInvHessianType());

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

    invhess.apply(mng, mng->getTrialStep(), vec);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -39.66013996624;
    (*gold)[1] = -120.57957024863;
    dotk::gtest::checkResults(*vec, *gold);
}

TEST(DOTk_LSR1InvHessian, getInvHessian)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    size_t secant_storage = 2;
    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_LSR1InvHessian invhess(vec, secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, invhess.getInvHessianType());

    invhess.setNumUpdatesStored(1);
    (*invhess.getDeltaPrimalStorage(0))[0] = -1.;
    (*invhess.getDeltaPrimalStorage(0))[1] = 1.;
    (*invhess.getDeltaGradStorage(0))[0] = -2402;
    (*invhess.getDeltaGradStorage(0))[1] = 800.;
    (*invhess.getDeltaGradPrimalInnerProductStorage())[0] = static_cast<Real>(1.) /
            invhess.getDeltaPrimalStorage(0)->dot(*invhess.getDeltaGradStorage(0));
    (*vec)[0] = 800.;
    (*vec)[1] = -400.;
    mng->setTrialStep(*vec);

    invhess.getInvHessian(mng->getTrialStep(), vec);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -39.66013996624;
    (*gold)[1] = -120.57957024863;
    dotk::gtest::checkResults(*vec, *gold);
}

}
