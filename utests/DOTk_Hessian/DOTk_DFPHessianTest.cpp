/*
 * DOTk_DFPHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_DFPHessian.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_SerialVector.cpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTk_DFPHessianTest
{

TEST(DOTk_DFPHessian, getHessian)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    dotk::DOTk_DFPHessian hess(mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::DFP_HESS, hess.getHessianType());

    // Primal Information
    dotk::serial::vector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);
    hess.getDeltaPrimal()->copy(*mng->getNewPrimal());
    hess.getDeltaPrimal()->axpy(-1.0, *mng->getOldPrimal());

    // Gradient Information
    std::tr1::shared_ptr<dotk::vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);
    hess.getDeltaGrad()->copy(*mng->getNewGradient());
    hess.getDeltaGrad()->axpy(-1.0, *mng->getOldGradient());

    // Trial step information
    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);
    std::tr1::shared_ptr<dotk::vector<Real> > solution = primal->control()->clone();
    hess.getHessian(mng->getTrialStep(), solution);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1681800.0937109496;
    (*gold)[1] = -559799.90628905024;
    dotk::gtest::checkResults(*solution, *gold);
}

TEST(DOTk_DFPHessian, apply)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::DOTk_DFPHessian hess(mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::DFP_HESS, hess.getHessianType());

    // Primal Information
    dotk::serial::vector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);

    // Gradient Information
    std::tr1::shared_ptr<dotk::vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);

    // Trial step information
    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1681800.0937109496;
    (*gold)[1] = -559799.90628905024;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

}
