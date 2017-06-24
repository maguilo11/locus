/*
 * DOTk_OptimalityCriteriaTest.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_InequalityConstraint.hpp"
#include "DOTk_OptimalityCriteriaDataMng.hpp"

namespace DOTkOptimalityCriteriaTest
{

TEST(DOTkPrimalTest, ConstructorOne)
{
    size_t ndual = 3;
    size_t nstate = 5;
    size_t ncontrol = 8;
    dotk::DOTk_Primal primal;

    primal.allocateSerialDualVector(ndual, 1.);
    primal.allocateSerialStateVector(nstate, 2.);
    primal.allocateSerialControlVector(ncontrol, 3.);

    EXPECT_EQ(dotk::types::PRIMAL, primal.type());
    EXPECT_EQ(ndual, primal.dual()->size());
    EXPECT_EQ(nstate, primal.state()->size());
    EXPECT_EQ(ncontrol, primal.control()->size());

    std::shared_ptr<dotk::Vector<Real> > dual = primal.dual()->clone();
    dual->fill(1.);
    dotk::gtest::checkResults(*dual, *primal.dual());
    std::shared_ptr<dotk::Vector<Real> > state = primal.state()->clone();
    state->fill(2.);
    dotk::gtest::checkResults(*state, *primal.state());
    std::shared_ptr<dotk::Vector<Real> > control = primal.control()->clone();
    control->fill(3.);
    dotk::gtest::checkResults(*control, *primal.control());

    primal.setDualLowerBound(-1);
    primal.setStateLowerBound(-2);
    primal.setControlLowerBound(-3);

    dual->fill(-1);
    dotk::gtest::checkResults(*dual, *primal.getDualLowerBound());
    state->fill(-2);
    dotk::gtest::checkResults(*state, *primal.getStateLowerBound());
    control->fill(-3);
    dotk::gtest::checkResults(*control, *primal.getControlLowerBound());

    primal.setDualUpperBound(1);
    primal.setStateUpperBound(2);
    primal.setControlUpperBound(3);

    dual->fill(1);
    dotk::gtest::checkResults(*dual, *primal.getDualUpperBound());
    state->fill(2);
    dotk::gtest::checkResults(*state, *primal.getStateUpperBound());
    control->fill(3);
    dotk::gtest::checkResults(*control, *primal.getControlUpperBound());

    dual->fill(-4);
    primal.setDualLowerBound(*dual);
    state->fill(-5);
    primal.setStateLowerBound(*state);
    control->fill(-6);
    primal.setControlLowerBound(*control);

    dotk::gtest::checkResults(*dual, *primal.getDualLowerBound());
    dotk::gtest::checkResults(*state, *primal.getStateLowerBound());
    dotk::gtest::checkResults(*control, *primal.getControlLowerBound());

    dual->fill(4);
    primal.setDualUpperBound(*dual);
    state->fill(5);
    primal.setStateUpperBound(*state);
    control->fill(6);
    primal.setControlUpperBound(*control);

    dotk::gtest::checkResults(*dual, *primal.getDualUpperBound());
    dotk::gtest::checkResults(*state, *primal.getStateUpperBound());
    dotk::gtest::checkResults(*control, *primal.getControlUpperBound());
}

TEST(DOTkPrimalTest, ConstructorTwo)
{
    size_t ndual = 3;
    size_t nstate = 5;
    size_t ncontrol = 8;
    dotk::StdArray<Real> dual(ndual, 1);
    dotk::StdArray<Real> state(nstate, 2);
    dotk::StdArray<Real> control(ncontrol, 3);
    dotk::DOTk_Primal primal;
    primal.allocateUserDefinedDual(dual);
    primal.allocateUserDefinedState(state);
    primal.allocateUserDefinedControl(control);

    EXPECT_EQ(dotk::types::PRIMAL, primal.type());
    EXPECT_EQ(ndual, primal.dual()->size());
    EXPECT_EQ(nstate, primal.state()->size());
    EXPECT_EQ(ncontrol, primal.control()->size());

    dotk::gtest::checkResults(dual, *primal.dual());
    dotk::gtest::checkResults(state, *primal.state());
    dotk::gtest::checkResults(control, *primal.control());

    primal.setDualLowerBound(-1);
    primal.setStateLowerBound(-2);
    primal.setControlLowerBound(-3);

    dual.fill(-1);
    dotk::gtest::checkResults(dual, *primal.getDualLowerBound());
    state.fill(-2);
    dotk::gtest::checkResults(state, *primal.getStateLowerBound());
    control.fill(-3);
    dotk::gtest::checkResults(control, *primal.getControlLowerBound());

    primal.setDualUpperBound(1);
    primal.setStateUpperBound(2);
    primal.setControlUpperBound(3);

    dual.fill(1);
    dotk::gtest::checkResults(dual, *primal.getDualUpperBound());
    state.fill(2);
    dotk::gtest::checkResults(state, *primal.getStateUpperBound());
    control.fill(3);
    dotk::gtest::checkResults(control, *primal.getControlUpperBound());

    dual.fill(-4);
    primal.setDualLowerBound(dual);
    state.fill(-5);
    primal.setStateLowerBound(state);
    control.fill(-6);
    primal.setControlLowerBound(control);

    dotk::gtest::checkResults(dual, *primal.getDualLowerBound());
    dotk::gtest::checkResults(state, *primal.getStateLowerBound());
    dotk::gtest::checkResults(control, *primal.getControlLowerBound());

    dual.fill(4);
    primal.setDualUpperBound(dual);
    state.fill(5);
    primal.setStateUpperBound(state);
    control.fill(6);
    primal.setControlUpperBound(control);

    dotk::gtest::checkResults(dual, *primal.getDualUpperBound());
    dotk::gtest::checkResults(state, *primal.getStateUpperBound());
    dotk::gtest::checkResults(control, *primal.getControlUpperBound());
}

TEST(DOTkOptimalityCriteriaDataMngTest, DefaultValues)
{
    size_t ndual = 3;
    size_t nstate = 5;
    size_t ncontrol = 8;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();

    primal->allocateSerialDualVector(ndual);
    primal->allocateSerialStateVector(nstate);
    primal->allocateSerialControlVector(ncontrol, 0.5);

    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-3);

    dotk::DOTk_OptimalityCriteriaDataMng mng(primal);

    EXPECT_EQ(nstate, mng.getState().size());
    EXPECT_EQ(ncontrol, mng.getOldControl().size());
    EXPECT_EQ(ncontrol, mng.getNewControl().size());
    EXPECT_EQ(ncontrol, mng.getControlLowerBound().size());
    EXPECT_EQ(ncontrol, mng.getControlUpperBound().size());
    EXPECT_EQ(ncontrol, mng.getObjectiveGradient().size());
    EXPECT_EQ(ncontrol, mng.getInequalityGradient().size());
    EXPECT_EQ(100, mng.getMaxNumOptimizationItr());

    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-2, mng.getMoveLimit(), tolerance);
    EXPECT_NEAR(0., mng.getInequalityDual(), tolerance);
    EXPECT_NEAR(1e-8, mng.getGradientTolerance(), tolerance);
    EXPECT_NEAR(1e-4, mng.getBisectionTolerance(), tolerance);
    EXPECT_NEAR(1e-8, mng.getFeasibilityTolerance(), tolerance);
    EXPECT_NEAR(0., mng.getOldObjectiveFunctionValue(), tolerance);
    EXPECT_NEAR(0., mng.getNewObjectiveFunctionValue(), tolerance);
    EXPECT_NEAR(1e-3, mng.getControlStagnationTolerance(), tolerance);
    EXPECT_NEAR(std::numeric_limits<Real>::max(), mng.getInequalityConstraintResidual(), tolerance);
    EXPECT_NEAR(std::numeric_limits<Real>::max(), mng.getMaxControlRelativeDifference(), tolerance);
    EXPECT_NEAR(std::numeric_limits<Real>::max(), mng.getNormObjectiveFunctionGradient(), tolerance);
    EXPECT_NEAR(0., mng.getInequalityConstraintDualLowerBound(), tolerance);
    EXPECT_NEAR(1e4, mng.getInequalityConstraintDualUpperBound(), tolerance);

    std::shared_ptr<dotk::Vector<Real> > state = mng.getState().clone();
    state->fill(0.);
    dotk::gtest::checkResults(*state, mng.getState());
    std::shared_ptr<dotk::Vector<Real> > control = mng.getNewControl().clone();
    control->fill(0.5);
    dotk::gtest::checkResults(*control, mng.getNewControl());

    control->fill(1.);
    dotk::gtest::checkResults(*control, mng.getControlUpperBound());
    control->fill(1e-3);
    dotk::gtest::checkResults(*control, mng.getControlLowerBound());

    control->fill(0.);
    dotk::gtest::checkResults(*control, mng.getOldControl());
    dotk::gtest::checkResults(*control, mng.getObjectiveGradient());
    dotk::gtest::checkResults(*control, mng.getInequalityGradient());
}

TEST(DOTkOptimalityCriteriaDataMngTest, SetFunctions)
{
    size_t ndual = 3;
    size_t nstate = 5;
    size_t ncontrol = 8;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();

    primal->allocateSerialDualVector(ndual);
    primal->allocateSerialStateVector(nstate);
    primal->allocateSerialControlVector(ncontrol, 0.5);

    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-3);

    dotk::DOTk_OptimalityCriteriaDataMng mng(primal);

    mng.setMaxNumOptimizationItr(32);
    EXPECT_EQ(32, mng.getMaxNumOptimizationItr());

    Real tolerance = 1e-8;
    mng.setMoveLimit(0.1);
    EXPECT_NEAR(0.1, mng.getMoveLimit(), tolerance);
    mng.setInequalityDual(0.11);
    EXPECT_NEAR(0.11, mng.getInequalityDual(), tolerance);
    mng.setGradientTolerance(1e-3);
    EXPECT_NEAR(1e-3, mng.getGradientTolerance(), tolerance);
    mng.setBisectionTolerance(1e-2);
    EXPECT_NEAR(1e-2, mng.getBisectionTolerance(), tolerance);
    mng.setFeasibilityTolerance(1e-5);
    EXPECT_NEAR(1e-5, mng.getFeasibilityTolerance(), tolerance);
    mng.setOldObjectiveFunctionValue(1.);
    EXPECT_NEAR(1., mng.getOldObjectiveFunctionValue(), tolerance);
    mng.setNewObjectiveFunctionValue(0.3);
    EXPECT_NEAR(0.3, mng.getNewObjectiveFunctionValue(), tolerance);
    mng.setControlStagnationTolerance(2e-6);
    EXPECT_NEAR(2e-6, mng.getControlStagnationTolerance(), tolerance);
    mng.setInequalityConstraintResidual(0.33);
    EXPECT_NEAR(0.33, mng.getInequalityConstraintResidual(), tolerance);
    mng.setMaxControlRelativeDifference(0.38);
    EXPECT_NEAR(0.38, mng.getMaxControlRelativeDifference(), tolerance);
    mng.setNormObjectiveFunctionGradient(0.038);
    EXPECT_NEAR(0.038, mng.getNormObjectiveFunctionGradient(), tolerance);
    mng.setInequalityConstraintDualLowerBound(1e-4);
    EXPECT_NEAR(1e-4, mng.getInequalityConstraintDualLowerBound(), tolerance);
    mng.setInequalityConstraintDualUpperBound(1e2);
    EXPECT_NEAR(1e2, mng.getInequalityConstraintDualUpperBound(), tolerance);
}

}
