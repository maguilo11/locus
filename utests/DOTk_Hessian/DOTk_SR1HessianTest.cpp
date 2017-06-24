/*
 * DOTk_SR1HessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_SR1Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSR1HessianTest
{

TEST(SR1Hessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    dotk::DOTk_SR1Hessian hess(*mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::SR1_HESS, hess.getHessianType());
    // vec Information
    dotk::StdVector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);
    // Gradient Information
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), mng->getNewGradient());
    // Trial step information
    mng->getTrialStep()->update(1., *mng->getNewGradient(), 0.);
    mng->getTrialStep()->scale(-1.);
    // ODD CASE
    hess.setNumOptimizationItrDone(3);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());
    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1280800;
    (*gold)[1] = -640400;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
    // EVEN CASE
    hess.setNumOptimizationItrDone(4);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());
    (*gold)[0] = 1601400.1249219237;
    (*gold)[1] = -800700.06246096187;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

TEST(SR1Hessian, getDeltaPrimal)
{

    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    dotk::StdVector<Real> control(2, 2.);
    mng.setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng.setNewPrimal(control);

    dotk::DOTk_SR1Hessian hess(*mng.getMatrixTimesVector());

    hess.getDeltaPrimal()->update(1., *mng.getNewPrimal(), 0.);
    hess.getDeltaPrimal()->update(-1.0, *mng.getOldPrimal(), 1.);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -1.;
    (*gold)[1] = 1.;
    dotk::gtest::checkResults(*(hess.getDeltaPrimal()), *(gold));
}

TEST(SR1Hessian, getDeltaGrad)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    // vec Information
    dotk::StdVector<Real> control(2, 2.);
    mng.setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng.setNewPrimal(control);

    // Gradient Information
    mng.getRoutinesMng()->gradient(mng.getOldPrimal(), mng.getOldGradient());
    mng.getRoutinesMng()->gradient(mng.getNewPrimal(), mng.getNewGradient());

    dotk::DOTk_SR1Hessian hess(*mng.getMatrixTimesVector());
    EXPECT_EQ(dotk::types::SR1_HESS, hess.getHessianType());
    hess.getDeltaGrad()->update(1., *mng.getNewGradient(), 0.);
    hess.getDeltaGrad()->update(-1.0, *mng.getOldGradient(), 1.);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -2402.;
    (*gold)[1] = 800.;

    dotk::gtest::checkResults(*(hess.getDeltaGrad()), *(gold));
}

TEST(SR1Hessian, getHessian)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    // vec Information
    dotk::StdVector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);

    dotk::DOTk_SR1Hessian hess(*mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::SR1_HESS, hess.getHessianType());
    hess.getDeltaPrimal()->update(1., *mng->getNewPrimal(), 0.);
    hess.getDeltaPrimal()->update(-1.0, *mng->getOldPrimal(), 1.);

    // Gradient Information
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), mng->getNewGradient());
    hess.getDeltaGrad()->update(1., *mng->getNewGradient(), 0.);
    hess.getDeltaGrad()->update(-1.0, *mng->getOldGradient(), 1.);

    // Trial step information
    mng->getTrialStep()->update(-1., *mng->getNewGradient(), 0.);

    // EVEN CASE
    hess.getHessian(mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1601400.1249219237;
    (*gold)[1] = -800700.06246096187;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);

    // ODD CASE
    hess.setNumOptimizationItrDone(1);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());
    (*gold)[0] = 1280800;
    (*gold)[1] = -640400;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

}
