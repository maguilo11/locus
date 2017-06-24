/*
 * DOTk_BoundLineSearchIxNewtonPcgNumIntgHessTest.cpp
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
#include "DOTk_ProjectedLineSearchStep.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkBoundLineSearchIxNewtonPcgNumIntgHessTest
{

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessFD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessBD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setBackwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessCD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setCentralDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessSoFD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessToFD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setThirdOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessToBD_CubicLS_NoPrec_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessFD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setForwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessBD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setBackwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessCD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setCentralDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessSoFD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setSecondOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(18u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessToFD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setThirdOrderForwardDifference(*primal->control(), 5e-9);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
}

TEST(DOTk_BoundLineSearchIxNewtonPcgNumIntgHess, getMin_UsrDefGrad_NumIntgHessToBD_CubicLS_NoPrec_ProjFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step = std::make_shared<dotk::DOTk_ProjectedLineSearchStep>(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    mng->setUserDefinedGradient();
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    hessian->setThirdOrderBackwardDifference(*primal->control(), 5e-8);

    dotk::DOTk_LineSearchInexactNewton alg(hessian, step, mng);
    alg.setLeftPrecCgKrylovSolver(primal);
    alg.getMin();

    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(17u, alg.getNumItrDone());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
}

}
