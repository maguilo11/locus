/*
 * DOTk_TrustRegionInexactNewtonCgnrTest.cpp
 *
 *  Created on: Nov 5, 2014
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

namespace DOTkTrustRegionInexactNewtonCgnrTest
{

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_DoubleDoglegTR_DOTk_FreudensteinRothObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;

    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective(new dotk::DOTk_FreudensteinRothObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setUserDefinedGradient();
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(12u, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_UsrDefHess_DoubleDoglegTR_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1.);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setUserDefinedGradient();
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(11u, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_UsrDefHess_DoglegTR_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1.);

    std::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_UsrDefHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-8);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_DFPHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(477u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_DFPHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(253u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_BBHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2134u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_BBHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2030u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_SR1Hess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2134u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_SR1Hess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(*primal->control());;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2030u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_LDFPHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(35u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 7e-8);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_LDFPHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(35u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 7e-8);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setForwardDifference(*primal->control());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(26u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessBD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setBackwardDifference(*primal->control(), 5e-8);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessCD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(*primal->control(), 1e-7);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessSoFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(*primal->control(), 1e-7);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(34u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessToBD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(*primal->control(), 1e-5);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessToFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(*primal->control(), 1e-5);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setForwardDifference(*primal->control(), 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessBD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setBackwardDifference(*primal->control(), 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessCD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(*primal->control(), 5e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessSoFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(*primal->control(), 1e-5);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessToFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(*primal->control(), 1e-5);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNR, getMin_UsrDefGrad_NumIntgHessToBD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(*primal->control(), 1e-5);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

}
