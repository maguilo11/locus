/*
 * DOTk_LineSearchInexactNewtonPcgTest.cpp
 *
 *  Created on: Sep 29, 2014
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

namespace DOTkInexactNewtonLineSearchPcgTest
{

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_CubicItrpLS_FreudensteinRothObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;

    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective(new dotk::DOTk_FreudensteinRothObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(8, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_BBHess_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(38, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_DFPHess_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_SR1Hess_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(38, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_LSR1Hess_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setLsr1Hessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(13, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_LDFPHess_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setLdfpHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_NumIntg_UsrDefHess_ArmijoLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);

    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(primal, 1e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(8, alg.getNumItrDone());
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(32, alg.getNumItrDone());
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(30, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_BBHess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_BBHess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(101, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_BBHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(50, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(101, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(50, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-5);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_LBFGSHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLbfgsHessian(primal->control(), secant_storage);;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(30, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 5e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_LDFPHess_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();

    mng->setUserDefinedGradient();
    size_t secant_storage = 6;
    hessian->setLdfpHessian(primal->control(), secant_storage);;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(13, alg.getNumItrDone());
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 5e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessFD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setForwardDifference(primal, 5e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessFD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setForwardDifference(primal, 5e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessFD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setForwardDifference(primal, 1e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessBD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setBackwardDifference(primal, 5e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessBD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBackwardDifference(primal, 5e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessBD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBackwardDifference(primal, 1e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessCD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setCentralDifference(primal, 5e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessCD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setCentralDifference(primal, 5e-9);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessCD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setCentralDifference(primal, 1e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessSoFD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setSecondOrderForwardDifference(primal, 5e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessSoFD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSecondOrderForwardDifference(primal, 1e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessSoFD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSecondOrderForwardDifference(primal, 5e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToBD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(primal, 5e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToBD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(primal, 5e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToBD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setThirdOrderBackwardDifference(primal, 1e-8);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToFD_CubicLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setThirdOrderForwardDifference(primal, 5e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToFD_ArmijoLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    hessian->setThirdOrderBackwardDifference(primal, 5e-7);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_LineSearchInexactNewtonPCG, getMin_UsrDefGrad_NumIntgHessToFD_GoldsteinLS_NoPrec)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    step->setGoldsteinLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setThirdOrderForwardDifference(primal, 5e-7);
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_LineSearchInexactNewtonPCG, checkConvergence)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldPrimal(*primal->control());
    (*primal->control())[0] = 2.;
    (*primal->control())[1] = 3.;
    mng->setNewPrimal(*primal->control());
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    Real tol = 1e-8;

    // TEST 1: ALG. HAS NOT CONVERGED
    EXPECT_FALSE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::OPT_ALG_HAS_NOT_CONVERGED, alg.getStoppingCriterion());

    // TEST 2: NaN TRIAL STEP NORM
    alg.setNumItrDone(1);
    mng->getTrialStep()->fill(std::numeric_limits<Real>::quiet_NaN());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::NaN_TRIAL_STEP_NORM, alg.getStoppingCriterion());
    EXPECT_NEAR(mng->getNewObjectiveFunctionValue(), mng->getOldObjectiveFunctionValue(), tol);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());
    dotk::gtest::checkResults(*mng->getNewGradient(), *mng->getOldGradient());

    // TEST 3: NaN GRADIENT NORM
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2.;
    mng->setTrialStep(*primal->control());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    mng->getNewGradient()->fill(std::numeric_limits<Real>::quiet_NaN());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::NaN_GRADIENT_NORM, alg.getStoppingCriterion());
    EXPECT_NEAR(mng->getNewObjectiveFunctionValue(), mng->getOldObjectiveFunctionValue(), tol);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());
    dotk::gtest::checkResults(*mng->getNewGradient(), *mng->getOldGradient());

    // TEST 4: TRIAL STEP NORM IS LESS THAN TOLERANCE
    mng->getTrialStep()->fill(std::numeric_limits<Real>::min());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());

    // TEST 5: GRADIENT NORM IS LESS THAN TOLERANCE
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2.;
    mng->setTrialStep(*primal->control());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    mng->getNewGradient()->fill(std::numeric_limits<Real>::min());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());

    // TEST 6: OBJECTIVE FUNCTION VALUE IS LESS THAN TOLERANCE
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    mng->setNewObjectiveFunctionValue(std::numeric_limits<Real>::min());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());

    // TEST 7: MAXIMUM NUMBER OF ITERATIONS REACHED
    alg.setNumItrDone(5000);
    mng->setNewObjectiveFunctionValue(1.);
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::MAX_NUM_ITR_REACHED, alg.getStoppingCriterion());
}

}
