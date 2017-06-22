/*
 * DOTk_BoundLineSearchIxNewtonCgnrNumIntgHessTest.cpp
 *
 *  Created on: Feb 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkBoundLineSearchIxNewtonCgnrNumIntgHessTest
{

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessFD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(19u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessBD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setBackwardDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(39u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessCD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setCentralDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(44u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessSoFD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(52u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessToFD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessToBD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessFD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(19u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessBD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setBackwardDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(39u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-8);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessCD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setCentralDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(32u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessSoFD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(32u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessToFD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderForwardDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

TEST(DOTk_BoundLineSearchIxNewtonCgnrNumIntgHess, getMin_UsrDefGrad_NumIntgHessToBD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-7);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgnrKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(29u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
}

}

