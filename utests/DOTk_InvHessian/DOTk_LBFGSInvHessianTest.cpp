/*
 * DOTk_LBFGSInvHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LBFGSInvHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLBFGSInvHessianTest
{

TEST(DOTk_LBFGSInvHessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setOldPrimal(*primal->control());
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 3.;
    mng->setNewPrimal(*primal->control());
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), mng->getNewGradient());
    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1);

    size_t secant_storage = 3;
    dotk::DOTk_LBFGSInvHessian invhess(mng->getMatrixTimesVector(), secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, invhess.getInvHessianType());
    invhess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -49.68763666992951;
    (*gold)[1] = -150.68712910146334;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

TEST(DOTk_LBFGSInvHessian, getInvHessian)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    size_t secant_storage = 3;
    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::DOTk_LBFGSInvHessian invhess(vec, secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, invhess.getInvHessianType());

    invhess.setNumUpdatesStored(1);
    EXPECT_EQ(1, invhess.getNumUpdatesStored());
    // Set Secant Storage
    (*invhess.getDeltaPrimalStorage(0))[0] = -1.;
    (*invhess.getDeltaPrimalStorage(0))[1] = 1.;
    // Set Delta Gradient Storage
    (*invhess.getDeltaGradStorage(0))[0] = -2402.;
    (*invhess.getDeltaGradStorage(0))[1] = 800.;
    for(size_t j = 0; j < invhess.getMaxNumSecantStorage(); ++j)
    {
        (*invhess.getDeltaGradPrimalInnerProductStorage())[j] = static_cast<Real>(1.) /
                invhess.getDeltaGradStorage(j)->dot(*invhess.getDeltaPrimalStorage(j));
    }
    // Set Trial Step
    (*vec)[0] = 800.;
    (*vec)[1] = -400.;
    mng->setTrialStep(*vec);
    // Test
    invhess.getInvHessian(mng->getTrialStep(), vec);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -49.68763666992951;
    (*gold)[1] = -150.68712910146334;
    dotk::gtest::checkResults(*vec, *gold);
}

}
