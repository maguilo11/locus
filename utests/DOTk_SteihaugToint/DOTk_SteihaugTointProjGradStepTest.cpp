/*
 * DOTk_SteihaugTointProjGradStepTest.cpp
 *
 *  Created on: Sep 9, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_SteihaugTointStepMng.hpp"
#include "DOTk_SteihaugTointLinMore.hpp"
#include "DOTk_SteihaugTointProjGradStep.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace DOTkSteihaugTointProjGradStepTest
{

TEST(SteihaugTointProjGradStep, getAndSetFunctions)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::DOTk_SteihaugTointProjGradStep mng(primal, hessian);

    // TEST BASE CLASS DEFAULT VALUES
    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-4, mng.getMinTrustRegionRadius(), tolerance);
    EXPECT_NEAR(1e4, mng.getMaxTrustRegionRadius(), tolerance);
    EXPECT_NEAR(1e4, mng.getTrustRegionRadius(), tolerance);
    EXPECT_NEAR(2., mng.getTrustRegionExpansion(), tolerance);
    EXPECT_NEAR(0.5, mng.getTrustRegionReduction(), tolerance);
    EXPECT_NEAR(0.25, mng.getActualOverPredictedReductionMidBound(), tolerance);
    EXPECT_NEAR(0.75, mng.getActualOverPredictedReductionUpperBound(), tolerance);
    EXPECT_NEAR(0.1, mng.getActualOverPredictedReductionLowerBound(), tolerance);
    EXPECT_NEAR(1, mng.getAdaptiveGradientInexactnessConstant(), tolerance);
    EXPECT_NEAR(0, mng.getAdaptiveGradientInexactnessTolerance(), tolerance);
    EXPECT_NEAR(1, mng.getAdaptiveObjectiveInexactnessConstant(), tolerance);
    EXPECT_NEAR(0, mng.getAdaptiveObjectiveInexactnessTolerance(), tolerance);
    EXPECT_NEAR(0., mng.getActualReduction(), tolerance);
    EXPECT_NEAR(0., mng.getPredictedReduction(), tolerance);
    EXPECT_NEAR(0., mng.getActualOverPredictedReduction(), tolerance);
    EXPECT_NEAR(1e-2, mng.getMinCosineAngleTolerance(), tolerance);
    EXPECT_EQ(0u, mng.getNumTrustRegionSubProblemItrDone());
    EXPECT_EQ(30u, mng.getMaxNumTrustRegionSubProblemItr());

    // TEST BASE CLASS SET FUNCTIONS VALUES
    mng.setMinTrustRegionRadius(11);
    EXPECT_NEAR(11., mng.getMinTrustRegionRadius(), tolerance);
    mng.setMaxTrustRegionRadius(10);
    EXPECT_NEAR(10., mng.getMaxTrustRegionRadius(), tolerance);
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
    mng.setAdaptiveGradientInexactnessConstant(0.14);
    EXPECT_NEAR(0.14, mng.getAdaptiveGradientInexactnessConstant(), tolerance);
    mng.setAdaptiveObjectiveInexactnessConstant(0.13);
    EXPECT_NEAR(0.13, mng.getAdaptiveObjectiveInexactnessConstant(), tolerance);
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

    // TEST CLASS DEFAULT VALUES
    EXPECT_EQ(10u, mng.getMaxNumProjections());
    EXPECT_NEAR(0.5, mng.getLineSearchContraction(), tolerance);
    EXPECT_NEAR(1e-2, mng.getControlUpdateRoutineConstant(), tolerance);

    // TEST CLASS SET FUNCTIONS
    mng.setMaxNumProjections(3);
    EXPECT_EQ(3u, mng.getMaxNumProjections());
    mng.setLineSearchContraction(0.25);
    EXPECT_NEAR(0.25, mng.getLineSearchContraction(), tolerance);
    mng.setControlUpdateRoutineConstant(1e-1);
    EXPECT_NEAR(1e-1, mng.getControlUpdateRoutineConstant(), tolerance);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    EXPECT_EQ(dotk::types::USER_DEFINED_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessDFP_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setDfpHessian(*primal->control());
    EXPECT_EQ(dotk::types::DFP_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(41u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessLDFP_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setLdfpHessian(*primal->control(), 2);
    EXPECT_EQ(dotk::types::LDFP_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessLSR1_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setLsr1Hessian(*primal->control(), 2);
    EXPECT_EQ(dotk::types::LSR1_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessSR1_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(*primal->control());
    EXPECT_EQ(dotk::types::SR1_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.setMaxNumOptimizationItr(500);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(231u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-5);
}

TEST(SteihaugTointLinMore, getMin_UsrDefGrad_HessBB_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(*primal->control());
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_HESS, hessian->hessianType());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.setMaxNumOptimizationItr(500);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(231u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-5);
}

TEST(SteihaugTointLinMore, getMin_GradFD_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-7);
    mng->setForwardFiniteDiffGradient(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(24u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
}

TEST(SteihaugTointLinMore, getMin_GradBD_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-8);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_GradCD_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-9);
    mng->setCentralFiniteDiffGradient(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_GradPFD_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-8);
    mng->setParallelForwardFiniteDiffGradient(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradPBD_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-8);
    mng->setParallelBackwardFiniteDiffGradient(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_GradPCD_UsrDefHess_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));

    primal->control()->fill(1e-9);
    mng->setParallelCentralFiniteDiffGradient(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessSFD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e3);
    primal->setControlUpperBound(1e3);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(*primal->control(), 5e0);
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(57u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessTFD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(*primal->control(), 1e0);
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_UserDefGrad_NumDiffHessTBD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(*primal->control(), 1e0);
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradFD_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-8);
    mng->setForwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
}

TEST(SteihaugTointLinMore, getMin_GradBD_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-8);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradCD_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setCentralFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradPFD_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-8);
    mng->setParallelForwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
}

TEST(SteihaugTointLinMore, getMin_GradPBD_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-8);
    mng->setParallelBackwardFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(SteihaugTointLinMore, getMin_GradPCD_NumDiffHessCD_Rosenbrock_Bounds)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0.);
    primal->setControlUpperBound(5.);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng> mng(new dotk::DOTk_SteihaugTointDataMng(primal,objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    primal->control()->fill(1e-6);
    mng->setParallelCentralFiniteDiffGradient(*primal->control());
    hessian->setCentralDifference(*primal->control());
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step_mng(new dotk::DOTk_SteihaugTointProjGradStep(primal,hessian));
    dotk::DOTk_SteihaugTointLinMore alg(mng, step_mng);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21u, alg.getNumOptimizationItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}
}
