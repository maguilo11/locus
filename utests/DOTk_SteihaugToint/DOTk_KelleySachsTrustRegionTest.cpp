/*
 * DOTk_KelleySachsTrustRegionTest.cpp
 *
 *  Created on: Apr 9, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_KelleySachsStepMng.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_SteihaugTointKelleySachs.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkTrustRegionKelleySachsTest
{

TEST(SteihaugTointKelleySachs, setAndGetFunctions)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal, objective));

    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);

    // TEST DEFAULT VALUES
    EXPECT_EQ(10u, alg.getMaxNumUpdates());
    // TEST SET FUNCTION
    alg.setMaxNumUpdates(2);
    EXPECT_EQ(2u, alg.getMaxNumUpdates());
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        mng(new dotk::DOTk_SteihaugTointDataMng(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setReducedSpaceHessian();
    EXPECT_EQ(dotk::types::USER_DEFINED_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessTBD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessTFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderForwardDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessSFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setSecondOrderForwardDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessBD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setBackwardDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setForwardDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_GradFD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setForwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-3);
}

TEST(SteihaugTointKelleySachs, getMin_GradCD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setCentralFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_GradBD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-3);
}

}
