/*
 * DOTk_BarzilaiBorweinHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_BarzilaiBorweinHessian.hpp"

namespace DOTkBarzilaiBorweinHessianTest
{

TEST(DOTk_BarzilaiBorweinHessian, getHessian)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    std::tr1::shared_ptr<dotk::vector<Real> > grad = primal->control()->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > control = primal->control()->clone();
    control->fill(2);

    mng->getRoutinesMng()->gradient(control, grad);
    mng->setNewGradient(*grad);

    grad->fill(0.);
    control->fill(0.);
    mng->setOldPrimal(*control);
    mng->setOldGradient(*grad);

    std::tr1::shared_ptr<dotk::vector<Real> > trial_step = primal->control()->clone();
    trial_step->fill(1);
    std::tr1::shared_ptr<dotk::vector<Real> > hess_times_vec = primal->control()->clone();

    dotk::DOTk_BarzilaiBorweinHessian hess(hess_times_vec);
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_HESS, hess.getHessianType());
    hess.computeDeltaPrimal(mng->getNewPrimal(), mng->getOldPrimal());
    hess.computeDeltaGradient(mng->getNewGradient(), mng->getOldGradient());

    // TEST 1: EVEN
    hess.getHessian(trial_step, hess_times_vec);
    std::tr1::shared_ptr<dotk::vector<Real> > gold = trial_step->clone();
    gold->fill(static_cast<Real>(1134.111480865));
    dotk::gtest::checkResults(*hess_times_vec, *gold);

    // TEST 2: ODD
    hess.setNumOptimizationItrDone(1);
    hess.getHessian(trial_step, hess_times_vec);
    gold->fill(static_cast<Real>(300.5));
    dotk::gtest::checkResults(*hess_times_vec, *gold);
}

}
