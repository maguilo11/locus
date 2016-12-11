/*
 * DOTk_BoundConstraintTest.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_PrimalVector.hpp"
#include "DOTk_PrimalVector.cpp"
#include "DOTk_BoundConstraint.hpp"

#include "DOTk_Rosenbrock.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_ArmijoLineSearch.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_BoundConstraintFactory.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkBoundConstraintTest
{

TEST(BoundConstraint, setAndGetStepSize)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);
    Real tol = 1e-8;
    EXPECT_NEAR(1., bound.getStepSize(), tol);
    bound.setStepSize(0.1);
    EXPECT_NEAR(0.1, bound.getStepSize(), tol);
}

TEST(BoundConstraint, setAndGetStagnationTol)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);
    Real tol = 1e-8;
    EXPECT_NEAR(1e-8, bound.getStagnationTolerance(), tol);
    bound.setStagnationTolerance(0.12);
    EXPECT_NEAR(0.12, bound.getStagnationTolerance(), tol);
}

TEST(BoundConstraint, setAndGetBoundStepType)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);
    EXPECT_EQ(dotk::types::ARMIJO_STEP, bound.getStepType());
    bound.setStepType(dotk::types::TRUST_REGION_STEP);
    EXPECT_EQ(dotk::types::TRUST_REGION_STEP, bound.getStepType());
}

TEST(BoundConstraint, getMinReductionStep)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setOldPrimal(*primal->control());
    mng->setOldObjectiveFunctionValue(mng->evaluateObjective(mng->getNewPrimal()));
    mng->computeGradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getTrialStep()->update(-1., *mng->getOldGradient(), 0.);
    mng->setUserDefinedGradient();

    dotk::DOTk_BoundConstraint bound(primal);
    Real step = bound.getMinReductionStep(mng);

    Real tol = 1e-8;
    EXPECT_NEAR(0.001953125, step, tol);
    EXPECT_EQ(10, bound.getNumFeasibleItr());
    (*primal->control())[0] = -1.12890625;
    (*primal->control())[1] = 2.78125;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    EXPECT_NEAR(231.5830976004, bound.getNewObjectiveFunctionValue(), tol);
}

TEST(BoundConstraint, getStep_Armijo)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ArmijoLineSearch> line_search(new dotk::DOTk_ArmijoLineSearch(primal->control()));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setOldPrimal(*primal->control());
    mng->setOldObjectiveFunctionValue(mng->getRoutinesMng()->objective(mng->getNewPrimal()));
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getTrialStep()->update(-1., *mng->getOldGradient(), 0.);
    mng->setUserDefinedGradient();

    dotk::DOTk_BoundConstraint bound(primal);
    bound.setStepType(dotk::types::ARMIJO_STEP);
    bound.setNumFeasibleItr(0);
    Real step = bound.getStep(line_search, mng);

    Real tol = 1e-8;
    EXPECT_NEAR(0.001953125, step, tol);
    EXPECT_EQ(0, bound.getNumFeasibleItr());
    EXPECT_EQ(10, line_search->getNumLineSearchItrDone());
    (*primal->control())[0] = -1.12890625;
    (*primal->control())[1] = 2.78125;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
}

TEST(BoundConstraint, getStep_MinReduction)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ArmijoLineSearch> line_search(new dotk::DOTk_ArmijoLineSearch(primal->control()));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setOldPrimal(*primal->control());
    mng->setOldObjectiveFunctionValue(mng->getRoutinesMng()->objective(mng->getNewPrimal()));
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getTrialStep()->update(-1., *mng->getOldGradient(), 0.);
    mng->setUserDefinedGradient();

    dotk::DOTk_BoundConstraint bound(primal);
    bound.setStepType(dotk::types::MIN_REDUCTION_STEP);
    bound.setNumFeasibleItr(0);
    Real step = bound.getStep(line_search, mng);

    Real tol = 1e-8;
    EXPECT_NEAR(0.001953125, step, tol);
    EXPECT_EQ(10, bound.getNumFeasibleItr());
    EXPECT_EQ(0, line_search->getNumLineSearchItrDone());
    (*primal->control())[0] = -1.12890625;
    (*primal->control())[1] = 2.78125;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    EXPECT_NEAR(231.5830976004, bound.getNewObjectiveFunctionValue(), tol);
}

TEST(BoundConstraint, getStep_Constant)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ArmijoLineSearch> line_search(new dotk::DOTk_ArmijoLineSearch(primal->control()));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setOldPrimal(*primal->control());
    mng->setOldObjectiveFunctionValue(mng->getRoutinesMng()->objective(mng->getNewPrimal()));
    mng->getRoutinesMng()->gradient(mng->getOldPrimal(), mng->getOldGradient());
    mng->getTrialStep()->update(-1., *mng->getOldGradient(), 0.);
    mng->setUserDefinedGradient();

    dotk::DOTk_BoundConstraint bound(primal);
    bound.setStepType(dotk::types::CONSTANT_STEP);
    bound.setNumFeasibleItr(0);
    Real step = bound.getStep(line_search, mng);

    Real tol = 1e-8;
    EXPECT_NEAR(1, step, tol);
    EXPECT_EQ(0, bound.getNumFeasibleItr());
    EXPECT_EQ(0, line_search->getNumLineSearchItrDone());
    (*primal->control())[0] = 2.;
    (*primal->control())[1] = 2.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    (*primal->control())[0] = 0.;
    (*primal->control())[1] = 0.;
    dotk::gtest::checkResults(*mng->getNewGradient(), *primal->control());
    EXPECT_NEAR(0., mng->getNewObjectiveFunctionValue(), tol);
}

TEST(BoundConstraint, setAndGetContractionStep)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);
    Real tol = 1e-8;
    EXPECT_NEAR(0.5, bound.getContractionStep(), tol);
    bound.setContractionStep(0.25);
    EXPECT_NEAR(0.25, bound.getContractionStep(), tol);
}

TEST(BoundConstraint, setAndGetNumFeasibleItr)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);
    EXPECT_EQ(0, bound.getNumFeasibleItr());
    bound.setNumFeasibleItr(1);
    EXPECT_EQ(1, bound.getNumFeasibleItr());
}

TEST(BoundConstraint, active)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);
    EXPECT_TRUE(bound.active());
    bound.activate(false);
    EXPECT_FALSE(bound.active());
}

TEST(BoundConstraint, isPrimalStationary1)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    dotk::DOTk_BoundConstraint bound(primal);

    // TEST 1: CONTROL IS STATIONARY
    bool is_vector_stationary = bound.isFeasible(primal->getControlLowerBound(),
                                                 primal->getControlUpperBound(),
                                                 primal->control());
    EXPECT_TRUE(is_vector_stationary);
    // TEST 2: LESS THAN LOWER BOUND
    primal->control()->fill(-1.);
    is_vector_stationary = bound.isFeasible(primal->getControlLowerBound(),
                                            primal->getControlUpperBound(),
                                            primal->control());
    EXPECT_FALSE(is_vector_stationary);
    // TEST 3: GREATER THAN LOWER BOUND
    primal->control()->fill(6.);
    is_vector_stationary = bound.isFeasible(primal->getControlLowerBound(),
                                            primal->getControlUpperBound(),
                                            primal->control());
    EXPECT_FALSE(is_vector_stationary);
}

TEST(BoundConstraint, computeActiveSet)
{
    size_t ncontrols = 10;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 3);
    primal->setControlLowerBound(1);
    primal->setControlUpperBound(5);
    (*primal->control())[0] = -1;
    (*primal->control())[4] = 6;
    (*primal->control())[8] = 7;
    (*primal->control())[9] = -2;

    dotk::DOTk_BoundConstraint bounds(primal);
    bounds.computeActiveSet(primal->getControlLowerBound(), primal->getControlUpperBound(), primal->control());

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 1;
    (*gold)[4] = 1;
    (*gold)[8] = 1;
    (*gold)[9] = 1;
    dotk::gtest::checkResults(*bounds.activeSet(), *gold);
}

TEST(BoundConstraint, pruneActive)
{
    size_t nvars = 10;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(nvars);
    dotk::DOTk_BoundConstraint bounds(primal);

    (*bounds.activeSet())[0] = 1;
    (*bounds.activeSet())[4] = 1;
    (*bounds.activeSet())[8] = 1;
    (*bounds.activeSet())[9] = 1;
    std::tr1::shared_ptr<dotk::Vector<Real> > gradient = dotk::gtest::allocateData(nvars, 3);
    bounds.pruneActive(gradient);

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = gradient->clone();
    gold->fill(3);
    (*gold)[0] = 0;
    (*gold)[4] = 0;
    (*gold)[8] = 0;
    (*gold)[9] = 0;
    dotk::gtest::checkResults(*gradient, *gold);
}

TEST(BoundConstraint, project)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(1);
    primal->setControlUpperBound(4);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    // TEST 1: PRIMAL INSIDE FEASIBLE REGION
    dotk::DOTk_BoundConstraint bound(primal);
    bound.project(primal->getControlLowerBound(), primal->getControlUpperBound(), mng->getNewPrimal());
    primal->control()->fill(2.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    primal->control()->fill(0.);
    dotk::gtest::checkResults(*bound.activeSet(), *primal->control());

    // TEST 2: PRIMAL OUTSIDE FEASIBLE REGION (UPPER BOUND VIOLATED)
    primal->control()->fill(5.);
    mng->setNewPrimal(*primal->control());
    bound.project(primal->getControlLowerBound(), primal->getControlUpperBound(), mng->getNewPrimal());
    primal->control()->fill(4.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*bound.activeSet(), *primal->control());

    // TEST 3: PRIMAL OUTSIDE FEASIBLE REGION (LOWER BOUND VIOLATED)
    primal->control()->fill(-1.);
    mng->setNewPrimal(*primal->control());
    bound.project(primal->getControlLowerBound(), primal->getControlUpperBound(), mng->getNewPrimal());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    primal->control()->fill(1.);
    dotk::gtest::checkResults(*bound.activeSet(), *primal->control());

    // TEST 4: PRIMAL OUTSIDE FEASIBLE REGION (LOWER AND UPPER BOUND VIOLATED)
    (*mng->getNewPrimal())[0] = 5;
    (*mng->getNewPrimal())[1] = -1;
    bound.project(primal->getControlLowerBound(), primal->getControlUpperBound(), mng->getNewPrimal());
    (*primal->control())[0] = 4;
    (*primal->control())[1] = 1;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control());
    primal->control()->fill(0.);
    (*primal->control())[0] = 1;
    (*primal->control())[1] = 1;
    dotk::gtest::checkResults(*bound.activeSet(), *primal->control());
}

}
