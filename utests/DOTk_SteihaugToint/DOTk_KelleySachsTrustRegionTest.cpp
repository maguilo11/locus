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
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal, objective));

    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);

    // TEST DEFAULT VALUES
    EXPECT_EQ(10, alg.getMaxNumUpdates());
    // TEST SET FUNCTION
    alg.setMaxNumUpdates(2);
    EXPECT_EQ(2, alg.getMaxNumUpdates());
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        mng(new dotk::DOTk_SteihaugTointDataMng(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setReducedSpaceHessian();
    EXPECT_EQ(dotk::types::USER_DEFINED_HESS, hessian->hessianType());
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));
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
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setCentralDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessTBD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessTFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderForwardDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessSFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setSecondOrderForwardDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessBD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setBackwardDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_UsrDefGrad_NumDiffHessFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setForwardDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal, hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_GradFD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setForwardFiniteDiffGradient(primal);
    hessian->setCentralDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-3);
}

TEST(SteihaugTointKelleySachs, getMin_GradCD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setCentralFiniteDiffGradient(primal);
    hessian->setCentralDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointKelleySachs, getMin_GradBD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setBackwardFiniteDiffGradient(primal);
    hessian->setCentralDifference(primal);
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step_mng(new dotk::DOTk_KelleySachsStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointKelleySachs alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::ACTUAL_REDUCTION_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-3);
}

}