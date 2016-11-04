/*
 * DOTk_BoundLineSearchInexactNewtonPcgTest.cpp
 *
 *  Created on: Nov 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkBoundLineSearchInexactNewtonPcgTest
{

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_ArmijoLS_NoPrec_FeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setFeasibleDirectionConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_UsrDefHess_ArmijoLS_NoPrec_ScaledProjectionAlongFeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(15, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-8);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_LDFPHess_ArmijoLS_NoPrec_FeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setFeasibleDirectionConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    size_t max_secant_storage = 4;
    hessian->setLdfpHessian(primal->control(), max_secant_storage);;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_LDFPHess_ArmijoLS_NoPrec_ScaledProjectionAlongFeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.2);
    mng->setUserDefinedGradient();
    size_t max_secant_storage = 4;
    hessian->setLdfpHessian(primal->control(), max_secant_storage);;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(22, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-8);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_LSR1Hess_ArmijoLS_NoPrec_FeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setFeasibleDirectionConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    size_t max_secant_storage = 4;
    hessian->setLsr1Hessian(primal->control(), max_secant_storage);;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(19, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_LSR1Hess_ArmijoLS_NoPrec_ScaledProjectionAlongFeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    size_t max_secant_storage = 4;
    hessian->setLsr1Hessian(primal->control(), max_secant_storage);;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(24, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_ArmijoLS_NoPrec_FeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setFeasibleDirectionConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_DFPHess_ArmijoLS_NoPrec_ScaledProjectionAlongFeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.2);
    mng->setUserDefinedGradient();
    hessian->setDfpHessian(primal->control());;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(21, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_BBHess_ArmijoLS_NoPrec_FeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setFeasibleDirectionConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-9);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_BBHess_ArmijoLS_NoPrec_ScaledProjectionAlongFeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setBarzilaiBorweinHessian(primal->control());;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(52, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-6);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_ArmijoLS_NoPrec_FeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setFeasibleDirectionConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(43, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-9);
}

TEST(DOTk_BoundLineSearchInexactNewtonPCG, getMin_UsrDefGrad_SR1Hess_ArmijoLS_NoPrec_ScaledProjectionAlongFeasDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoLineSearch(primal);
    mng->setUserDefinedGradient();
    hessian->setSr1Hessian(primal->control());;
    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(52, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 3e-6);
}

}
