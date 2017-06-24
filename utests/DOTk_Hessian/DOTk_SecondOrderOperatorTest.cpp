/*
 * DOTk_SecondOrderOperatorTest.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_UserDefinedHessian.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSecondOrderOperatorTest
{

TEST(SecondOrderOperator, getAndSetNumUpdatesStored)
{
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_EQ(0, hess.getNumUpdatesStored());
    hess.setNumUpdatesStored(31);
    EXPECT_EQ(31, hess.getNumUpdatesStored());
}

TEST(SecondOrderOperator, getAndSetNumOptimizationItrDone)
{
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_EQ(0, hess.getNumOptimizationItrDone());
    hess.setNumOptimizationItrDone(3);
    EXPECT_EQ(3, hess.getNumOptimizationItrDone());
}

TEST(SecondOrderOperator, getAndSetMaxNumSecantStorages)
{
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_EQ(0, hess.getMaxNumSecantStorage());
    hess.setMaxNumSecantStorages(5);
    EXPECT_EQ(5, hess.getMaxNumSecantStorage());
}

TEST(SecondOrderOperator, getAndSetDiagonalScaleFactor)
{
    Real tol = 1e-8;
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_NEAR(0.1, hess.getDiagonalScaleFactor(), tol);
    hess.setDiagonalScaleFactor(0.5);
    EXPECT_NEAR(0.5, hess.getDiagonalScaleFactor(), tol);
}

TEST(SecondOrderOperator, setAndGetUpdateSecantStorageFlag)
{
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_TRUE(hess.updateSecondOrderOperator());
    hess.setUpdateSecondOrderOperator(false);
    EXPECT_FALSE(hess.updateSecondOrderOperator());
}

TEST(SecondOrderOperator, IsSecantStorageFull)
{
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_FALSE(hess.IsSecantStorageFull());
    hess.setSecantStorageFullFlag(true);
    EXPECT_TRUE(hess.IsSecantStorageFull());
}

TEST(SecondOrderOperator, getAndSetInvHessianType)
{
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_EQ(dotk::types::INV_HESS_DISABLED, hess.getInvHessianType());
    hess.setInvHessianType(dotk::types::LBFGS_INV_HESS);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, hess.getInvHessianType());
}

TEST(SecondOrderOperator, updateSecantStorage)
{
    std::shared_ptr<dotk::Vector<Real> > delta_grad = dotk::gtest::allocateData(2);
    (*delta_grad)[0] = -1151;
    (*delta_grad)[1] = 250;

    size_t storage = 2;
    std::vector<Real> rho(storage, 0.);
    std::shared_ptr<dotk::matrix<Real> > dgrad_storage = std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*delta_grad, storage);
    std::shared_ptr<dotk::matrix<Real> > dprimal_storage = std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*delta_grad, storage);

    // Test 1: SECANT STORAGE IS NOT FULL, STORE DATA INTO FIRST ENTRY
    dotk::DOTk_SecondOrderOperator hess(storage);
    hess.setUpdateSecondOrderOperator(true);
    std::shared_ptr<dotk::Vector<Real> > delta_primal = delta_grad->clone();
    delta_primal->fill(-0.5);
    hess.updateSecantStorage(delta_primal, delta_grad, rho, dprimal_storage, dgrad_storage);
    EXPECT_EQ(1, hess.getNumUpdatesStored());

    std::vector<Real> rho_gold(rho.size(), 0.);
    std::shared_ptr<dotk::matrix<Real> > dgrad_sotrage_gold = dgrad_storage->clone();
    std::shared_ptr<dotk::matrix<Real> > dprimal_storage_gold = dprimal_storage->clone();
    rho_gold[0] = 0.00221975582686;
    (*dgrad_sotrage_gold)(0,0) = -1151.;
    (*dgrad_sotrage_gold)(0,1) = 250.;
    dprimal_storage_gold->basis(0)->fill(-0.5);
    dotk::gtest::checkResults(rho, rho_gold);
    dotk::gtest::checkResults(*dgrad_storage->basis(0), *dgrad_sotrage_gold->basis(0));
    dotk::gtest::checkResults(*dgrad_storage->basis(1), *dgrad_sotrage_gold->basis(1));
    dotk::gtest::checkResults(*dprimal_storage->basis(0), *dprimal_storage_gold->basis(0));
    dotk::gtest::checkResults(*dprimal_storage->basis(1), *dprimal_storage_gold->basis(1));

    // Test 2: SECANT STORAGE IS NOT FULL, STORE DATA INTO SECOND ENTRY
    (*delta_grad)[0] = -1152.;
    (*delta_grad)[1] = 251.;
    delta_primal->fill(-0.51);
    hess.setUpdateSecondOrderOperator(true);
    hess.updateSecantStorage(delta_primal, delta_grad, rho, dprimal_storage, dgrad_storage);
    EXPECT_EQ(2, hess.getNumUpdatesStored());

    rho_gold[1] = 0.00217623;
    (*dgrad_sotrage_gold)(1,0) = -1152.;
    (*dgrad_sotrage_gold)(1,1) = 251.;
    dprimal_storage_gold->basis(1)->fill(-0.51);
    dotk::gtest::checkResults(rho, rho_gold);
    dotk::gtest::checkResults(*dgrad_storage->basis(0), *dgrad_sotrage_gold->basis(0));
    dotk::gtest::checkResults(*dgrad_storage->basis(1), *dgrad_sotrage_gold->basis(1));
    dotk::gtest::checkResults(*dprimal_storage->basis(0), *dprimal_storage_gold->basis(0));
    dotk::gtest::checkResults(*dprimal_storage->basis(1), *dprimal_storage_gold->basis(1));

    // Test 3: SECANT STORAGE IS FULL, CULL HALF AND STORE DATA MOST RECENT DATA
    (*delta_grad)[0] = -1153.;
    (*delta_grad)[1] = 252.;
    delta_primal->fill(-0.52);
    hess.setUpdateSecondOrderOperator(true);
    hess.updateSecantStorage(delta_primal, delta_grad, rho, dprimal_storage, dgrad_storage);
    EXPECT_EQ(2, hess.getNumUpdatesStored());

    rho_gold[0] = 0.0021762312028;
    rho_gold[1] = 0.00213438060275;
    (*dgrad_sotrage_gold)(0,0) = -1152.;
    (*dgrad_sotrage_gold)(0,1) = 251.;
    dprimal_storage_gold->basis(0)->fill(-0.51);
    (*dgrad_sotrage_gold)(1,0) = -1153.;
    (*dgrad_sotrage_gold)(1,1) = 252.;
    dprimal_storage_gold->basis(1)->fill(-0.52);
    dotk::gtest::checkResults(rho, rho_gold);
    dotk::gtest::checkResults(*dgrad_storage->basis(0), *dgrad_sotrage_gold->basis(0));
    dotk::gtest::checkResults(*dgrad_storage->basis(1), *dgrad_sotrage_gold->basis(1));
    dotk::gtest::checkResults(*dprimal_storage->basis(0), *dprimal_storage_gold->basis(0));
    dotk::gtest::checkResults(*dprimal_storage->basis(1), *dprimal_storage_gold->basis(1));
}

TEST(SecondOrderOperator, computeStatePerturbation)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > grad = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();

    mng->getRoutinesMng()->gradient(vector, grad);

    mng->getNewPrimal()->update(1., *vector, 0.);
    mng->getNewGradient()->update(1., *grad, 0.);

    dotk::DOTk_SecondOrderOperator hess;
    std::shared_ptr<dotk::Vector<Real> > dgrad = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > dprimal = primal->control()->clone();
    hess.computeDeltaPrimal(mng->getNewPrimal(), mng->getOldPrimal(), dprimal);
    hess.computeDeltaGradient(mng->getNewGradient(), mng->getOldGradient(), dgrad);

    dotk::gtest::checkResults(*dgrad, *grad);
    dotk::gtest::checkResults(*dprimal, *vector);
}

TEST(SecondOrderOperator, getBarzilaiBorweinStep)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    dotk::DOTk_SecondOrderOperator hess;
    EXPECT_EQ(dotk::types::HESSIAN_DISABLED, hess.getHessianType());

    std::shared_ptr<dotk::Vector<Real> > dgrad = primal->control()->clone();
    dgrad->fill(2);
    std::shared_ptr<dotk::Vector<Real> > dprimal = primal->control()->clone();
    dprimal->fill(2);
    mng->getRoutinesMng()->gradient(dprimal, dgrad);

    Real step = hess.getBarzilaiBorweinStep(dprimal, dgrad);
    Real tol = 1e-8;
    EXPECT_NEAR(0.0008817475, step, tol);

    hess.setNumOptimizationItrDone(1);
    step = hess.getBarzilaiBorweinStep(dprimal, dgrad);
    EXPECT_NEAR(0.003327787, step, tol);
}

TEST(UserDefinedHessian, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    dotk::DOTk_UserDefinedHessian hess;
    EXPECT_EQ(dotk::types::USER_DEFINED_HESS, hess.getHessianType());

    mng->getTrialStep()->fill(1);
    hess.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 3202.;
    (*gold)[1] = -600.;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

}
