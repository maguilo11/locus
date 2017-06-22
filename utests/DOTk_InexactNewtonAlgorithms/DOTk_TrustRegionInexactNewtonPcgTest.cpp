/*
 * DOTk_TrustRegionInexactNewtonPcgTest.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_BealeObjective.hpp"
#include "DOTk_TrustRegionMngTypeULP.hpp"
#include "DOTk_TrustRegionInexactNewton.hpp"
#include "DOTk_FreudensteinRothObjective.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkTrustRegionInexactNewtonTest
{

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_DoubleDoglegTR_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setUserDefinedGradient();
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8u, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_DoglegTR_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(10u, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_GradFD_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    primal->control()->fill(5e-9);
    mng->setForwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_GradPrllFD_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    primal->control()->fill(5e-9);
    mng->setParallelForwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_GradBD_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    primal->control()->fill(5e-9);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_GradPrllBD_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    primal->control()->fill(5e-9);
    mng->setParallelBackwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_GradCD_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    primal->control()->fill(5e-9);
    mng->setCentralFiniteDiffGradient(*primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_GradPrllCD_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    primal->control()->fill(5e-9);
    mng->setParallelCentralFiniteDiffGradient(*primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(42u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_BBHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(1863u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_BBHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2030u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(1863u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2030u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_LDFPHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(18u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_LDFPHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-8);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setForwardDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessBD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setBackwardDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessCD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessSoFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToBD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setForwardDifference(*primal->control(), 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessBD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setBackwardDifference(*primal->control(), 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessCD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(*primal->control(), 5e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessSoFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(*primal->control(), 1e-5);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(*primal->control());
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToBD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(*primal->control());
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

}
