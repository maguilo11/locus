/*
 * DOTk_LeftPrecCgneTest.cpp
 *
 *  Created on: Nov 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LeftPrecCGNE.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_RelativeCriterion.hpp"
#include "DOTk_LeftPrecCGNEqDataMng.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_DescentDirectionTools.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLeftPrecCgneTest
{

TEST(DOTk_LeftPrecCGNE, initialize)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    std::shared_ptr<dotk::DOTk_Hessian> hessian = std::make_shared<dotk::DOTk_Hessian>();

    mng->computeGradient();
    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    vec->update(-1., *mng->getNewGradient(), 0.);

    std::shared_ptr<dotk::DOTk_LeftPrecCGNEqDataMng> solver_mng = std::make_shared<dotk::DOTk_LeftPrecCGNEqDataMng>(primal, hessian);
    dotk::DOTk_LeftPrecCGNE solver(solver_mng);

    Real relative_tolerance = 1e-2;
    std::shared_ptr<dotk::DOTk_RelativeCriterion> criterion = std::make_shared<dotk::DOTk_RelativeCriterion>(relative_tolerance);
    solver.initialize(vec, criterion, mng);

    dotk::gtest::checkResults(*solver.getDataMng()->getResidual(), *vec);
    (*vec)[0] = -6731204.;
    (*vec)[1] = 1361600.;
    dotk::gtest::checkResults(*vec, *solver.getDescentDirection());

    dotk::gtest::checkResults(*solver.getDataMng()->getLeftPrecTimesVector(), *solver.getDataMng()->getResidual());

    Real tol = 1e-8;
    EXPECT_NEAR(1651.182606497, solver.getSolverResidualNorm(), tol);
    EXPECT_NEAR(16.51182606497, solver.getInitialStoppingTolerance(), tol);
}

TEST(DOTk_LeftPrecCGNE, cgne)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    std::shared_ptr<dotk::DOTk_Hessian> hessian = std::make_shared<dotk::DOTk_Hessian>();

    mng->computeGradient();
    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::gtools::getSteepestDescent(mng->getNewGradient(), vec);
    std::shared_ptr<dotk::DOTk_LeftPrecCGNEqDataMng> solver_mng = std::make_shared<dotk::DOTk_LeftPrecCGNEqDataMng>(primal, hessian);
    dotk::DOTk_LeftPrecCGNE solver(solver_mng);

    Real relative_tolerance = 1e-2;
    std::shared_ptr<dotk::DOTk_RelativeCriterion> criterion = std::make_shared<dotk::DOTk_RelativeCriterion>(relative_tolerance);
    solver.cgne(vec, criterion, mng);

    EXPECT_EQ(dotk::types::SOLVER_TOLERANCE_SATISFIED, solver.getSolverStopCriterion());
    (*vec)[0] = -0.002493765586;
    (*vec)[1] = 1.990024937656;
    dotk::gtest::checkResults(*solver.getDataMng()->getSolution(), *vec);
}

TEST(DOTk_LeftPrecCGNE, solve)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Hessian> hessian = std::make_shared<dotk::DOTk_Hessian>();
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->computeGradient();
    std::shared_ptr<dotk::Vector<Real> > vec = primal->control()->clone();
    dotk::gtools::getSteepestDescent(mng->getNewGradient(), vec);

    std::shared_ptr<dotk::DOTk_LeftPrecCGNEqDataMng> solver_mng = std::make_shared<dotk::DOTk_LeftPrecCGNEqDataMng>(primal, hessian);
    dotk::DOTk_LeftPrecCGNE solver(solver_mng);

    Real relative_tolerance = 1e-2;
    std::shared_ptr<dotk::DOTk_RelativeCriterion> criterion = std::make_shared<dotk::DOTk_RelativeCriterion>(relative_tolerance);
    solver.solve(vec, criterion, mng);

    EXPECT_EQ(dotk::types::SOLVER_TOLERANCE_SATISFIED, solver.getSolverStopCriterion());
    (*vec)[0] = -0.002493765586;
    (*vec)[1] = 1.990024937656;
    dotk::gtest::checkResults(*solver.getDataMng()->getSolution(), *vec);
}

}
