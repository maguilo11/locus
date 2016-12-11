/*
 * DOTk_LeftPrecCGTest.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_LeftPrecCG.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_RelativeCriterion.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_LeftPrecConjGradDataMng.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLeftPrecCGTest
{

TEST(DOTk_LeftPrecCG, setAndGetNumSolverItrDone)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    EXPECT_EQ(0, solver.getNumSolverItrDone());
    solver.setNumSolverItrDone(32);
    EXPECT_EQ(32, solver.getNumSolverItrDone());
}

TEST(DOTk_LeftPrecCG, setAndGetTrustRegionRadius)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::max(), solver.getTrustRegionRadius(), tol);
    solver.setTrustRegionRadius(1);
    EXPECT_NEAR(1, solver.getTrustRegionRadius(), tol);
}

TEST(DOTk_LeftPrecCG, setAndGetSolverResidualNorm)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), solver.getSolverResidualNorm(), tol);
    solver.setSolverResidualNorm(1e-4);
    EXPECT_NEAR(1e-4, solver.getSolverResidualNorm(), tol);
}

TEST(DOTk_LeftPrecCG, setAndGetInitialStoppingTolerance)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), solver.getInitialStoppingTolerance(), tol);
    solver.setInitialStoppingTolerance(4e-1);
    EXPECT_NEAR(4e-1, solver.getInitialStoppingTolerance(), tol);
}

TEST(DOTk_LeftPrecCG, setAndGetSolverStopCriterion)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    EXPECT_EQ(dotk::types::SOLVER_DID_NOT_CONVERGED, solver.getSolverStopCriterion());
    solver.setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
    EXPECT_EQ(dotk::types::MAX_SOLVER_ITR_REACHED, solver.getSolverStopCriterion());
}

TEST(DOTk_LeftPrecCG, isCurvatureInvalid)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    EXPECT_FALSE(solver.invalidCurvatureWasDetected());
    solver.invalidCurvatureDetected(true);
    EXPECT_TRUE(solver.invalidCurvatureWasDetected());
}

TEST(DOTk_LeftPrecCG, checkCurvature)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    solver.checkCurvature(-1);
    EXPECT_EQ(dotk::types::NEGATIVE_CURVATURE_DETECTED, solver.getSolverStopCriterion());
    solver.checkCurvature(std::numeric_limits<Real>::quiet_NaN());
    EXPECT_EQ(dotk::types::NaN_CURVATURE_DETECTED, solver.getSolverStopCriterion());
    solver.checkCurvature(std::numeric_limits<Real>::min());
    EXPECT_EQ(dotk::types::ZERO_CURVATURE_DETECTED, solver.getSolverStopCriterion());
    solver.checkCurvature(std::numeric_limits<Real>::infinity());
    EXPECT_EQ(dotk::types::INF_CURVATURE_DETECTED, solver.getSolverStopCriterion());
}

TEST(DOTk_LeftPrecCG, checkResidualNorm)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    Real stopping_tolerance = 1e-1;
    solver.checkResidualNorm(std::numeric_limits<Real>::min(), stopping_tolerance);
    EXPECT_EQ(dotk::types::SOLVER_TOLERANCE_SATISFIED, solver.getSolverStopCriterion());
    solver.checkResidualNorm(std::numeric_limits<Real>::quiet_NaN(), stopping_tolerance);
    EXPECT_EQ(dotk::types::NaN_RESIDUAL_NORM, solver.getSolverStopCriterion());
    solver.checkResidualNorm(std::numeric_limits<Real>::infinity(), stopping_tolerance);
    EXPECT_EQ(dotk::types::INF_RESIDUAL_NORM, solver.getSolverStopCriterion());
}

TEST(DOTk_LeftPrecCG, initialize)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    mng->computeGradient();

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);
    std::tr1::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    vec->copy(*mng->getNewGradient());
    vec->scale(-1);

    Real relative_tolerance = 1e-2;
    std::tr1::shared_ptr<dotk::DOTk_RelativeCriterion> criterion(new dotk::DOTk_RelativeCriterion(relative_tolerance));
    solver.initialize(vec, criterion, mng);

    dotk::gtest::checkResults(*solver.getDataMng()->getResidual(), *vec);
    dotk::gtest::checkResults(*solver.getDataMng()->getLeftPrecTimesVector(), *solver.getDataMng()->getResidual());

    Real tol = 1e-8;
    EXPECT_NEAR(1651.182606497, solver.getSolverResidualNorm(), tol);
    EXPECT_NEAR(16.51182606497, solver.getInitialStoppingTolerance(), tol);
}

TEST(DOTk_LeftPrecCG, pcg)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    mng->computeGradient();
    std::tr1::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::gtools::getSteepestDescent(mng->getNewGradient(), vec);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));
    dotk::DOTk_LeftPrecCG solver(solver_mng);

    Real relative_tolerance = 1e-2;
    std::tr1::shared_ptr<dotk::DOTk_RelativeCriterion> criterion(new dotk::DOTk_RelativeCriterion(relative_tolerance));
    solver.pcg(vec, criterion, mng);

    EXPECT_EQ(dotk::types::SOLVER_TOLERANCE_SATISFIED, solver.getSolverStopCriterion());
    (*vec)[0] = -0.002493765586;
    (*vec)[1] = 1.990024937656;
    dotk::gtest::checkResults(*solver.getDataMng()->getSolution(), *vec);
}

TEST(DOTk_LeftPrecCG, solve)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    std::tr1::shared_ptr<dotk::DOTk_LeftPrecConjGradDataMng> solver_mng(new dotk::DOTk_LeftPrecConjGradDataMng(primal, hessian));

    mng->computeGradient();
    std::tr1::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::gtools::getSteepestDescent(mng->getNewGradient(), vec);

    dotk::DOTk_LeftPrecCG solver(solver_mng);

    Real relative_tolerance = 1e-2;
    std::tr1::shared_ptr<dotk::DOTk_RelativeCriterion> criterion(new dotk::DOTk_RelativeCriterion(relative_tolerance));
    solver.solve(vec, criterion, mng);

    EXPECT_EQ(dotk::types::SOLVER_TOLERANCE_SATISFIED, solver.getSolverStopCriterion());
    (*vec)[0] = -0.002493765586;
    (*vec)[1] = 1.990024937656;
    dotk::gtest::checkResults(*solver.getDataMng()->getSolution(), *vec);
}

}
