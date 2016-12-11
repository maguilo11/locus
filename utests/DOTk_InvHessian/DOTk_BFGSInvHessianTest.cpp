/*
 * DOTk_BFGSInvHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_BFGSInvHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkBFGSInvHessianTest
{

TEST(DOTk_BFGSInvHessian, apply)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    dotk::DOTk_BFGSInvHessian invhess(mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::BFGS_INV_HESS, invhess.getInvHessianType());

    primal->control()->fill(2.);
    mng->setOldPrimal(*primal->control());
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 3.;
    mng->setNewPrimal(*primal->control());

    std::tr1::shared_ptr<dotk::Vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);

    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);

    std::tr1::shared_ptr<dotk::Vector<Real> > direction = primal->control()->clone();
    invhess.apply(mng, mng->getTrialStep(), direction);

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -49.68763666992951;
    (*gold)[1] = -150.68712910146334;
    dotk::gtest::checkResults(*direction, *gold);
}

TEST(DOTk_BFGSInvHessian, getInvHessian)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    std::tr1::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_BFGSInvHessian invhess(vec);
    EXPECT_EQ(dotk::types::BFGS_INV_HESS, invhess.getInvHessianType());

    (*invhess.getDeltaPrimal())[0] = -1.;
    (*invhess.getDeltaPrimal())[1] = 1.;
    (*invhess.getDeltaGrad())[0] = -2402;
    (*invhess.getDeltaGrad())[1] = 800.;
    (*vec)[0] = 800.;
    (*vec)[1] = -400.;
    mng->setTrialStep(*vec);

    invhess.getInvHessian(mng->getTrialStep(), vec);

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -49.68763666992951;
    (*gold)[1] = -150.68712910146334;
    dotk::gtest::checkResults(*vec, *gold);
}

}
