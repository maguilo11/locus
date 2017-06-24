/*
 * DOTk_LineSearchQuasiNewtonTest.cpp
 *
 *  Created on: Sep 19, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_BealeObjective.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLineSearchQuasiNewtonTest
{

TEST(GradientBasedOptimizationTools, steepestDescent)
{
    std::shared_ptr<dotk::Vector<Real> > grad = dotk::gtest::allocateControl();
    std::shared_ptr<dotk::Vector<Real> > step = grad->clone();
    grad->fill(3);

    dotk::gtools::getSteepestDescent(grad, step);

    std::shared_ptr<dotk::Vector<Real> > gold = grad->clone();
    gold->fill(-3);
    dotk::gtest::checkResults(*step, *gold);
}

TEST(GradientBasedOptimizationTools, computeCosineAngle)
{
    std::shared_ptr<dotk::Vector<Real> > grad = dotk::gtest::allocateControl();
    std::shared_ptr<dotk::Vector<Real> > step = grad->clone();
    (*grad)[0] = 1.;
    (*grad)[1] = 2.;
    (*step)[0] = 11.;
    (*step)[1] = -22.;

    Real value = dotk::gtools::computeCosineAngle(grad, step);
    Real tol = 1e-8;
    EXPECT_NEAR(0.6, value, tol);

    (*step)[0] = -1.;
    (*step)[1] = -2.;
    value = dotk::gtools::computeCosineAngle(grad, step);
    EXPECT_NEAR(1., value, tol);
}

TEST(GradientBasedOptimizationTools, checkDescentDirection)
{
    std::shared_ptr<dotk::Vector<Real> > grad = dotk::gtest::allocateControl();
    (*grad)[0] = 1.;
    (*grad)[1] = 2.;
    std::shared_ptr<dotk::Vector<Real> > step = grad->clone();
    (*step)[0] = 11.;
    (*step)[1] = -22.;

    // TEST 1: DESCENT DIRECTION IS VALID, DO NOT TAKE STEEPEST DESCENT
    dotk::gtools::checkDescentDirection(grad, step);
    std::shared_ptr<dotk::Vector<Real> > gold = grad->clone();
    gold->update(1., *step, 0.);
    dotk::gtest::checkResults(*step, *gold);

    // TEST 2: INVALID DESCENT DIRECTION, TAKE STEEPEST DESCENT
    step->fill(std::numeric_limits<Real>::min());
    dotk::gtools::checkDescentDirection(grad, step);
    gold->update(-1., *grad, 0.);
    dotk::gtest::checkResults(*step, *gold);

    // TEST 3: INVALID DESCENT DIRECTION, TAKE STEEPEST DESCENT
    step->fill(std::numeric_limits<Real>::quiet_NaN());
    dotk::gtools::checkDescentDirection(grad, step);
    gold->update(-1., *grad, 0.);
    dotk::gtest::checkResults(*step, *gold);

    // TEST 4: INVALID DESCENT DIRECTION, TAKE STEEPEST DESCENT
    step->fill(std::numeric_limits<Real>::infinity());
    dotk::gtools::checkDescentDirection(grad, step);
    gold->update(-1., *grad, 0.);
    dotk::gtest::checkResults(*step, *gold);
}

TEST(LineSearchQuasiNewton, BarzilaiBorweinCubicIntrpLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 1);
    std::shared_ptr<dotk::DOTk_BealeObjective> objective = std::make_shared<dotk::DOTk_BealeObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    alg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LSR1CubicIntrpLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 1);
    std::shared_ptr<dotk::DOTk_BealeObjective> objective = std::make_shared<dotk::DOTk_BealeObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod();
    alg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(16, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BFGSCubicIntrpLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 1);
    std::shared_ptr<dotk::DOTk_BealeObjective> objective = std::make_shared<dotk::DOTk_BealeObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBfgsSecantMethod();
    alg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(20, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, SR1CubicIntrpLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 1);
    std::shared_ptr<dotk::DOTk_BealeObjective> objective = std::make_shared<dotk::DOTk_BealeObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    alg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LBFGSCubicIntrpLS_BealeObjective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 1);
    std::shared_ptr<dotk::DOTk_BealeObjective> objective = std::make_shared<dotk::DOTk_BealeObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLbfgsSecantMethod();
    alg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BarzilaiBorweinArmijoLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(59, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LSR1ArmijoLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);

    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(24, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LBFGSArmijoLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLbfgsSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(27, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LDFPArmijoLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    size_t secant_storage = 5;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLdfpSecantMethod(secant_storage);
    alg.getMin();
    EXPECT_EQ(dotk::types::LDFP_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-3);
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(34, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BFGSArmijoLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    EXPECT_EQ(dotk::types::BFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(27, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, SR1ArmijoLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    EXPECT_EQ(dotk::types::SR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(59, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BarzilaiBorweinGoldsteinLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    step->setMaxNumIterations(0);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(30, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LSR1GoldsteinLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(28, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LBFGSGoldsteinLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLbfgsSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(26, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LDFPGoldsteinLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    step->setContractionFactor(0.25);
    size_t secant_storage = 5;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLdfpSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LDFP_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-4);
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(135, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BFGSGoldsteinLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    EXPECT_EQ(dotk::types::BFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, SR1GoldsteinLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    step->setMaxNumIterations(0);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    EXPECT_EQ(dotk::types::SR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(30, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BarzilaiBorweinCubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LSR1CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(27, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LBFGSCubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLbfgsSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, LDFPCubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    size_t secant_storage = 4;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLdfpSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LDFP_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-3);
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(34, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, BFGSCubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    EXPECT_EQ(dotk::types::BFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(23, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewton, SR1CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    EXPECT_EQ(dotk::types::SR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
}

}
