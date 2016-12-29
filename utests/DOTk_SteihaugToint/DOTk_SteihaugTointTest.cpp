/*
 * DOTk_SteihaugTointTest.cpp
 *
 *  Created on: Aug 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_SteihaugTointLinMore.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_SteihaugTointStepMng.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace DOTkSteinhaugTointTest
{

TEST(SteihaugTointDataMng, getAndSetFunctions)
{
    size_t ncontrol = 8;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    primal->allocateSerialControlVector(ncontrol, 0.5);

    dotk::DOTk_SteihaugTointDataMng mng(primal, objective);

    EXPECT_EQ(ncontrol, mng.getTrialStep()->size());
    EXPECT_EQ(ncontrol, mng.getNewPrimal()->size());
}

TEST(SteihaugTointStepMng, updateAdaptiveGradientInexactnessTolerance)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::DOTk_SteihaugTointStepMng mng(primal, hessian);

    // TEST 1: MIN Value = norm of the gradient
    Real norm_gradient = 0.1;
    mng.updateAdaptiveGradientInexactnessTolerance(norm_gradient);
    Real tolerance = 1e-8;
    EXPECT_NEAR(0.1, mng.getAdaptiveGradientInexactnessTolerance(), tolerance);

    // TEST 2: MIN Value = current trust region radius
    mng.setTrustRegionRadius(0.01);
    mng.updateAdaptiveGradientInexactnessTolerance(norm_gradient);
    EXPECT_NEAR(0.01, mng.getAdaptiveGradientInexactnessTolerance(), tolerance);
}

TEST(SteihaugTointStepMng, updateAdaptiveObjectiveInexactnessTolerance)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::DOTk_SteihaugTointStepMng mng(primal, hessian);

    mng.setActualOverPredictedReduction(0.1);
    mng.setPredictedReduction(0.1);
    mng.updateAdaptiveObjectiveInexactnessTolerance();
    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-2, mng.getAdaptiveObjectiveInexactnessTolerance(), tolerance);
}

TEST(SteihaugTointStepMng, getAndSetFunctions)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::DOTk_SteihaugTointStepMng mng(primal, hessian);

    // TEST DEFAULT VALUES
    Real tolerance = 1e-8;
    EXPECT_NEAR(1e4, mng.getTrustRegionRadius(), tolerance);
    EXPECT_NEAR(2., mng.getTrustRegionExpansion(), tolerance);
    EXPECT_NEAR(0.5, mng.getTrustRegionReduction(), tolerance);
    EXPECT_NEAR(0.75, mng.getActualOverPredictedReductionUpperBound(), tolerance);
    EXPECT_NEAR(0.1, mng.getActualOverPredictedReductionLowerBound(), tolerance);
    EXPECT_NEAR(0., mng.getActualReduction(), tolerance);
    EXPECT_NEAR(0., mng.getPredictedReduction(), tolerance);
    EXPECT_NEAR(0., mng.getActualOverPredictedReduction(), tolerance);
    EXPECT_NEAR(1e-2, mng.getMinCosineAngleTolerance(), tolerance);
    EXPECT_EQ(0u, mng.getNumTrustRegionSubProblemItrDone());
    EXPECT_EQ(30u, mng.getMaxNumTrustRegionSubProblemItr());

    // TEST SET FUNCTIONS VALUES
    mng.setTrustRegionRadius(1);
    EXPECT_NEAR(1., mng.getTrustRegionRadius(), tolerance);
    mng.setTrustRegionExpansion(3);
    EXPECT_NEAR(3., mng.getTrustRegionExpansion(), tolerance);
    mng.setTrustRegionReduction(0.1);
    EXPECT_NEAR(0.1, mng.getTrustRegionReduction(), tolerance);
    mng.setActualOverPredictedReductionUpperBound(0.9);
    EXPECT_NEAR(0.9, mng.getActualOverPredictedReductionUpperBound(), tolerance);
    mng.setActualOverPredictedReductionLowerBound(0.15);
    EXPECT_NEAR(0.15, mng.getActualOverPredictedReductionLowerBound(), tolerance);
    mng.setActualReduction(0.25);
    EXPECT_NEAR(0.25, mng.getActualReduction(), tolerance);
    mng.setPredictedReduction(0.2);
    EXPECT_NEAR(0.2, mng.getPredictedReduction(), tolerance);
    mng.setActualOverPredictedReduction(0.98);
    EXPECT_NEAR(0.98, mng.getActualOverPredictedReduction(), tolerance);
    mng.setMinCosineAngleTolerance(1e-1);
    EXPECT_NEAR(1e-1, mng.getMinCosineAngleTolerance(), tolerance);
    mng.setNumTrustRegionSubProblemItrDone(2);
    EXPECT_EQ(2u, mng.getNumTrustRegionSubProblemItrDone());
    mng.setMaxNumTrustRegionSubProblemItr(23);
    EXPECT_EQ(23u, mng.getMaxNumTrustRegionSubProblemItr());
}

TEST(Hessian, apply)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    mng->getTrialStep()->fill(1.);
    hessian->apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    Real tolerance = 1e-8;
    (*primal->control())[0] = 3202.;
    (*primal->control())[1] = -600.;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *primal->control(), tolerance);
}

TEST(SteihaugTointLinMore, getAndSetFunctions)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);

    // TEST DEFAULT VALUES
    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-10, alg.getGradientTolerance(), tolerance);
    EXPECT_NEAR(1e-10, alg.getObjectiveTolerance(), tolerance);
    EXPECT_NEAR(1e-10, alg.getTrialStepTolerance(), tolerance);
    EXPECT_EQ(0u, alg.getNumOptimizationItrDone());
    EXPECT_EQ(100u, alg.getMaxNumOptimizationItr());
    EXPECT_EQ(dotk::types::OPT_ALG_HAS_NOT_CONVERGED, alg.getStoppingCriterion());

    // TEST SET FUNCTIONS VALUES
    alg.setGradientTolerance(0.54);
    EXPECT_NEAR(0.54, alg.getGradientTolerance(), tolerance);
    alg.setObjectiveTolerance(0.22);
    EXPECT_NEAR(0.22, alg.getObjectiveTolerance(), tolerance);
    alg.setTrialStepTolerance(0.32);
    EXPECT_NEAR(0.32, alg.getTrialStepTolerance(), tolerance);
    alg.setNumOptimizationItrDone(1);
    EXPECT_EQ(1u, alg.getNumOptimizationItrDone());
    alg.setMaxNumOptimizationItr(23);
    EXPECT_EQ(23u, alg.getMaxNumOptimizationItr());
    alg.setStoppingCriterion(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessLDFP_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setLdfpHessian(*primal->control(), 2);
    EXPECT_EQ(dotk::types::LDFP_HESS, hessian->hessianType());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(74u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessLSR1_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setLsr1Hessian(*primal->control(), 2);
    EXPECT_EQ(dotk::types::LSR1_HESS, hessian->hessianType());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(77u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointLinMore, getMin_GradFD_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-7);
    mng->setForwardFiniteDiffGradient(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
}

TEST(SteihaugTointLinMore, getMin_GradBD_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-8);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradCD_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-9);
    mng->setCentralFiniteDiffGradient(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_GradPFD_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-8);
    mng->setParallelForwardFiniteDiffGradient(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(37u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradPBD_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-8);
    mng->setParallelBackwardFiniteDiffGradient(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradPCD_UsrDefHess_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-9);
    mng->setParallelCentralFiniteDiffGradient(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessBD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setBackwardDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));

    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 2e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setForwardDifference(*primal->control(), 1e-8);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(33u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 2e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessSFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(*primal->control(), 1e-8);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(26u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessTFD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessTBD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(*primal->control(), 1e-6);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(34u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradFD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-7);
    mng->setForwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
}

TEST(SteihaugTointLinMore, getMin_GradBD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-8);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(24u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradCD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setCentralFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradPFD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-7);
    mng->setParallelForwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
}

TEST(SteihaugTointLinMore, getMin_GradPBD_NumDiffHessCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-8);
    mng->setParallelBackwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(24u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradCD_NumDiffHessPCD_Rosenbrock)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setParallelCentralFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng> step(new dotk::DOTk_SteihaugTointStepMng(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

}
