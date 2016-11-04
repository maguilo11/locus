/*
 * DOTk_TrustRegionInexactNewtonCgneTest.cpp
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

namespace DOTkTrustRegionInexactNewtonCgneTest
{

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_DoubleDoglegTR_DOTk_FreudensteinRothObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;

    std::tr1::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective(new dotk::DOTk_FreudensteinRothObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setUserDefinedGradient();
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(10, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_DoubleDoglegTR_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1.);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setUserDefinedGradient();
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8 , alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_DoglegTR_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1.);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(10 , alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-8);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_DFPHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_BBHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2214, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_BBHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2030, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_SR1Hess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2214, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_SR1Hess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(2030, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_LDFPHess_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoglegTrustRegionMethod();
    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(primal->control(), secant_storage);;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(18, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_LDFPHess_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(primal->control(), secant_storage);;
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-8);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessCD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(primal, 1e-8);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessSoFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(primal, 1e-7);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessToBD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(primal, 1e-8);
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessToFD_DoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderForwardDifference(primal, 5e-9);
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setForwardDifference(primal, 7.5e-8);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(12, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessBD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setBackwardDifference(primal, 5e-8);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(56, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessCD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setCentralDifference(primal, 1e-8);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessSoFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setSecondOrderForwardDifference(primal, 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessToFD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(primal, 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_TrustRegionInexactNewtonCGNE, getMin_UsrDefGrad_NumDiffHessToBD_DoubleDoglegTR_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2.);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP> mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(primal, 1e-7);
    mng->setDoubleDoglegTrustRegionMethod(mng->getTrialStep());
    mng->setUserDefinedGradient();
    dotk::DOTk_TrustRegionInexactNewton alg(hessian, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

}
