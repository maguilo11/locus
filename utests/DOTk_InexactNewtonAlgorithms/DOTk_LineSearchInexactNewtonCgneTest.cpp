/*
 * DOTk_LineSearchInexactNewtonCgneTest.cpp
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
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_FreudensteinRothObjective.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLineSearchInexactNewtonCgneTest
{

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_CubicItrpLS_FreudensteinRothObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;

    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setStagnationTolerance(1e-2);
    std::tr1::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective(new dotk::DOTk_FreudensteinRothObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(10, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UserHess_LDFPHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setLdfpHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_LDFPHess_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_NumIntg_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderBackwardDifference(*primal->control(), 1e-5);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 2e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_UsrDefHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-9);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_DFPHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setDfpHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(31, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_DFPHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setDfpHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(30, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_DFPHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setDfpHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_BBHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setBarzilaiBorweinHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-9);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_BBHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setBarzilaiBorweinHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(129, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_BBHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setBarzilaiBorweinHessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(50u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_SR1Hess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setSr1Hessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-9);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_SR1Hess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setSr1Hessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(129u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_SR1Hess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setSr1Hessian(*primal->control());

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(50u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 4e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_LDFPHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    const size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(46u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 7e-8);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_LDFPHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    const size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(32u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_LDFPHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    const size_t secant_storage = 6;
    hessian->setLdfpHessian(*primal->control(), secant_storage);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(13u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 2e-6);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessFD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(*primal->control(), 1e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(11u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessFD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(*primal->control(), 1e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(11u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessFD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(*primal->control(), 1e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessBD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setBackwardDifference(*primal->control(), 2.5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessBD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setBackwardDifference(*primal->control(), 7.5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessBD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setBackwardDifference(*primal->control(), 2.5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessCD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setCentralDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessCD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setCentralDifference(*primal->control(), 1e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessCD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setCentralDifference(*primal->control(), 1e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessSoFD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessSoFD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(14u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessSoFD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessToBD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessToBD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessToBD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessToFD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderForwardDifference(*primal->control(), 1e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessToFD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setArmijoLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonCGNE, getMin_UsrDefGrad_NumIntgHessToFD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->setGoldsteinLineSearch(primal);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderForwardDifference(*primal->control(), 1e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgneKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

}
