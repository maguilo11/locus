/*
 * DOTk_LDFPHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LDFPHessian.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLDFPHessianTest
{

TEST(DOTk_LDFPHessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    size_t secant_storage = 2;
    dotk::DOTk_LDFPHessian hess(*mng->getMatrixTimesVector(), secant_storage);
    EXPECT_EQ(dotk::types::LDFP_HESS, hess.getHessianType());

    dotk::StdVector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);

    std::shared_ptr<dotk::Vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);
    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1681800.0937109494;
    (*gold)[1] = -559799.90628905024;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

}
