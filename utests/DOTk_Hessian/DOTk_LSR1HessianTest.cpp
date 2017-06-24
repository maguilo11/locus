/*
 * DOTk_LSR1HessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LSR1Hessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLSR1HessianTest
{

TEST(DOTk_LSR1Hessian, getHessian)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    size_t secant_storage = 3;
    dotk::DOTk_LSR1Hessian hess(*mng->getMatrixTimesVector(), secant_storage);
    EXPECT_EQ(dotk::types::LSR1_HESS, hess.getHessianType());

    hess.setNumUpdatesStored(3);
    EXPECT_EQ(3, hess.getNumUpdatesStored());
    // Set Delta Primal Storage
    (*hess.getDeltaPrimalStorage(0))[0] = 0.980021092332167;
    (*hess.getDeltaPrimalStorage(0))[1] = -1.191750027971445;
    (*hess.getDeltaPrimalStorage(1))[0] = -0.437981859241434;
    (*hess.getDeltaPrimalStorage(1))[1] = 1.366459262299115;
    (*hess.getDeltaPrimalStorage(2))[0] = -0.872652252646403;
    (*hess.getDeltaPrimalStorage(2))[1] = 0.089327034058346;
    // Set Delta Gradient Storage
    (*hess.getDeltaGradStorage(0))[0] = -0.548372720057146;
    (*hess.getDeltaGradStorage(0))[1] = -0.096267955052387;
    (*hess.getDeltaGradStorage(1))[0] = -1.3806708657954;
    (*hess.getDeltaGradStorage(1))[1] = -0.728371038269661;
    (*hess.getDeltaGradStorage(2))[0] = 1.88600200593293;
    (*hess.getDeltaGradStorage(2))[1] = -2.941385892824783;

    std::shared_ptr<dotk::Vector<Real> > trial_step = primal->control()->clone();
    (*trial_step)[0] = 0.979448879095330;
    (*trial_step)[1] = -0.265611268123836;
    std::shared_ptr<dotk::Vector<Real> > Hess_times_vec = primal->control()->clone();
    hess.getHessian(trial_step, Hess_times_vec);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -2.889509684974680;
    (*gold)[1] = 1.197517114591315;
    dotk::gtest::checkResults(*Hess_times_vec, *gold);
}

}
