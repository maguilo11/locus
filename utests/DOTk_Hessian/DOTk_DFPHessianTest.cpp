/*
 * DOTk_DFPHessianTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_DFPHessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTk_DFPHessianTest
{

TEST(DOTk_DFPHessian, getHessian)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    dotk::DOTk_DFPHessian hess(*mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::DFP_HESS, hess.getHessianType());

    // Primal Information
    dotk::StdVector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);
    hess.getDeltaPrimal()->update(1., *mng->getNewPrimal(), 0.);
    hess.getDeltaPrimal()->update(-1.0, *mng->getOldPrimal(), 1.);

    // Gradient Information
    std::shared_ptr<dotk::Vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);
    hess.getDeltaGrad()->update(1., *mng->getNewGradient(), 0.);
    hess.getDeltaGrad()->update(-1.0, *mng->getOldGradient(), 1.);

    // Trial step information
    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);
    std::shared_ptr<dotk::Vector<Real> > solution = primal->control()->clone();
    hess.getHessian(mng->getTrialStep(), solution);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1681800.0937109496;
    (*gold)[1] = -559799.90628905024;
    dotk::gtest::checkResults(*solution, *gold);
}

TEST(DOTk_DFPHessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    dotk::DOTk_DFPHessian hess(*mng->getMatrixTimesVector());
    EXPECT_EQ(dotk::types::DFP_HESS, hess.getHessianType());

    // Primal Information
    dotk::StdVector<Real> control(2, 2.);
    mng->setOldPrimal(control);
    control[0] = 1.;
    control[1] = 3.;
    mng->setNewPrimal(control);

    // Gradient Information
    std::shared_ptr<dotk::Vector<Real> > grad = primal->control()->clone();
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), grad);
    mng->setOldGradient(*grad);
    mng->getRoutinesMng()->gradient(mng->getNewPrimal(), grad);
    mng->setNewGradient(*grad);

    // Trial step information
    mng->setTrialStep(*mng->getNewGradient());
    mng->getTrialStep()->scale(-1.);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1681800.0937109496;
    (*gold)[1] = -559799.90628905024;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

}
