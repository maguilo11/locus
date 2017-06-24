/*
 * DOTk_MethodCCSA_Test.cpp
 *
 *  Created on: Nov 17, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "matrix.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_SubProblemMMA.hpp"
#include "DOTk_RoutinesTypeNP.hpp"
#include "DOTk_RoutinesTypeLP.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_SubProblemGCMMA.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_DataMngNonlinearCG.hpp"
#include "DOTk_ScaleParametersNLCG.hpp"
#include "DOTk_DualObjectiveFunctionMMA.hpp"
#include "DOTk_GcmmaTestObjectiveFunction.hpp"
#include "DOTk_GcmmaTestInequalityConstraint.hpp"

namespace DOTkMethodCCSATest
{

TEST(Bounds, setAndGetFunctions)
{
    dotk::DOTk_BoundConstraints bounds;

    EXPECT_EQ(10, bounds.getMaxNumFeasibleIterations());
    bounds.setMaxNumFeasibleIterations(2);
    EXPECT_EQ(2, bounds.getMaxNumFeasibleIterations());

    Real tolerance = 1e-8;
    EXPECT_NEAR(0.5, bounds.getContractionFactor(), tolerance);
    bounds.setContractionFactor(0.1);
    EXPECT_NEAR(0.1, bounds.getContractionFactor(), tolerance);
}

TEST(Bounds, isDirectionFeasible)
{
    size_t nvars = 10;
    dotk::StdArray<Real> x(nvars, 3);
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);

    // CASE 1: Feasible
    dotk::DOTk_BoundConstraints bounds;
    EXPECT_TRUE(bounds.isDirectionFeasible(lower_bound, upper_bound, x));

    // CASE 2: Not Feasible
    x[4] = 6;
    EXPECT_FALSE(bounds.isDirectionFeasible(lower_bound, upper_bound, x));
}

TEST(Bounds, computeFeasibleDirection)
{
    size_t nvars = 10;
    dotk::StdArray<Real> x(nvars, 3);
    x[0] = -1;
    x[4] = 6;
    x[8] = 7;
    x[9] = -2;
    dotk::StdArray<Real> trial_step(nvars, -2.);
    trial_step[0] = 10;
    trial_step[4] = -10;
    trial_step[8] = -10;
    trial_step[9] = 10;
    dotk::StdArray<Real> trial_variable(nvars);
    dotk::StdArray<Real> feasible_direction(nvars);
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);

    dotk::DOTk_BoundConstraints bounds;
    bounds.computeFeasibleDirection(lower_bound, upper_bound, x, trial_step, trial_variable, feasible_direction);

    dotk::StdArray<Real> gold(nvars, 2);
    gold[0] = 4;
    gold[4] = 1;
    gold[9] = 3;
    dotk::gtest::checkResults(trial_variable, gold);
    gold.fill(-1);
    gold[0] = 5;
    gold[4] = -5;
    gold[8] = -5;
    gold[9] = 5;
    dotk::gtest::checkResults(feasible_direction, gold);
}

TEST(Bounds, computeFeasibleDirectionWithProjectionActive)
{
    size_t nvars = 10;
    dotk::StdArray<Real> x(nvars, 3);
    x[0] = -1;
    x[4] = 6;
    x[8] = 7;
    x[9] = -2;
    dotk::StdArray<Real> trial_step(nvars, -5.);
    trial_step[0] = 10;
    trial_step[4] = -10;
    trial_step[8] = -10;
    trial_step[9] = 10;
    dotk::StdArray<Real> trial_variable(nvars);
    dotk::StdArray<Real> feasible_direction(nvars);
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);

    dotk::DOTk_BoundConstraints bounds;
    bounds.computeFeasibleDirection(lower_bound, upper_bound, x, trial_step, trial_variable, feasible_direction);

    dotk::StdArray<Real> gold(nvars, 2.9951171875);
    gold[0] = 1;
    gold[4] = 5;
    gold[8] = 5;
    gold[9] = 1;
    dotk::gtest::checkResults(trial_variable, gold);
    gold.fill(-0.0048828125);
    gold[0] = 2;
    gold[4] = -1;
    gold[8] = -2;
    gold[9] = 3;
    dotk::gtest::checkResults(feasible_direction, gold);
}

TEST(Bounds, project)
{
    size_t nvars = 10;
    dotk::StdArray<Real> x(nvars, 3);
    x[0] = -1;
    x[4] = 6;
    x[8] = 7;
    x[9] = -2;
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);

    dotk::DOTk_BoundConstraints bounds;
    EXPECT_TRUE(bounds.active());
    bounds.project(lower_bound, upper_bound, x);

    dotk::StdArray<Real> gold(nvars, 3);
    gold[0] = 1;
    gold[4] = 5;
    gold[8] = 5;
    gold[9] = 1;
    dotk::gtest::checkResults(x, gold);
}

TEST(Bounds, pruneActive)
{
    size_t nvars = 10;
    dotk::StdArray<Real> gradient(nvars, 3);
    dotk::StdArray<Real> active_set(nvars, 0.);
    active_set[0] = 1;
    active_set[4] = 1;
    active_set[8] = 1;
    active_set[9] = 1;

    dotk::DOTk_BoundConstraints bounds;
    bounds.pruneActive(active_set, gradient);

    dotk::StdArray<Real> gold(nvars, 3);
    gold[0] = 0;
    gold[4] = 0;
    gold[8] = 0;
    gold[9] = 0;
    dotk::gtest::checkResults(gradient, gold);
}

TEST(Bounds, computeProjectedStep)
{
    size_t nvars = 10;
    dotk::StdArray<Real> trial_x(nvars, 3);
    dotk::StdArray<Real> current_x(nvars, 1.);
    dotk::StdArray<Real> projected_step(nvars);

    dotk::DOTk_BoundConstraints bounds;
    bounds.computeProjectedStep(trial_x, current_x, projected_step);

    dotk::StdArray<Real> gold(nvars, 2);
    dotk::gtest::checkResults(projected_step, gold);
}

TEST(Bounds, computeProjectedGradient)
{
    size_t nvars = 10;
    dotk::StdArray<Real> x(nvars, 3);
    x[0] = -1;
    x[4] = 6;
    x[8] = 7;
    x[9] = -2;
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);
    dotk::StdArray<Real> projected_gradient(nvars, -2.);

    dotk::DOTk_BoundConstraints bounds;
    bounds.computeProjectedGradient(x, lower_bound, upper_bound, projected_gradient);

    dotk::StdArray<Real> gold(nvars, -2);
    gold[0] = 0;
    gold[4] = 0;
    gold[8] = 0;
    gold[9] = 0;
    dotk::gtest::checkResults(projected_gradient, gold);
}

TEST(Bounds, computeActiveAndInactiveSets)
{
    size_t nvars = 10;
    dotk::StdArray<Real> input(nvars, 3);
    dotk::StdArray<Real> active(nvars, 0);
    dotk::StdArray<Real> inactive(nvars, 0);
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);

    input[0] = -1;
    input[3] = -1;
    input[7] = 6;
    input[8] = 6;
    dotk::DOTk_BoundConstraints bounds;
    bounds.computeActiveAndInactiveSets(input, lower_bound, upper_bound, active, inactive);

    // TEST 1: inactive set
    dotk::StdArray<Real> gold(nvars, 1);
    gold[0] = 0;
    gold[3] = 0;
    gold[7] = 0;
    gold[8] = 0;
    dotk::gtest::checkResults(inactive, gold);
    // TEST 2: active set
    gold.fill(0);
    gold[0] = 1;
    gold[3] = 1;
    gold[7] = 1;
    gold[8] = 1;
    dotk::gtest::checkResults(active, gold);
}

TEST(Bounds, projectActive)
{
    size_t nvars = 10;
    dotk::StdArray<Real> x(nvars, 3);
    x[0] = -1;
    x[4] = 6;
    x[8] = 7;
    x[9] = -2;
    dotk::StdArray<Real> active_set(nvars);
    dotk::StdArray<Real> lower_bound(nvars, 1.);
    dotk::StdArray<Real> upper_bound(nvars, 5.);

    dotk::DOTk_BoundConstraints bounds;
    bounds.projectActive(lower_bound, upper_bound, x, active_set);

    dotk::StdArray<Real> gold(nvars, 3);
    gold[0] = 1;
    gold[4] = 5;
    gold[8] = 5;
    gold[9] = 1;
    dotk::gtest::checkResults(x, gold);

    gold.fill(0.);
    gold[0] = 1;
    gold[4] = 1;
    gold[8] = 1;
    gold[9] = 1;
    dotk::gtest::checkResults(active_set, gold);
}

TEST(DOTk_NLCG, fletcherReeves)
{
    size_t nvars = 10;
    dotk::StdArray<Real> new_steepest_descent(nvars, 1.);
    dotk::StdArray<Real> old_steepest_descent(nvars, 5.);

    Real tolerance = 1e-8;
    Real gold = dotk::nlcg::fletcherReeves(new_steepest_descent, old_steepest_descent);
    EXPECT_NEAR(0.04, gold, tolerance);
}

TEST(DOTk_NLCG, polakRibiere)
{
    size_t nvars = 10;
    dotk::StdArray<Real> new_steepest_descent(nvars, 1.);
    dotk::StdArray<Real> old_steepest_descent(nvars, 5.);

    Real tolerance = 1e-8;
    Real gold = dotk::nlcg::polakRibiere(new_steepest_descent, old_steepest_descent);
    EXPECT_NEAR(-0.16, gold, tolerance);
}

TEST(DOTk_NLCG, hestenesStiefel)
{
    size_t nvars = 10;
    dotk::StdArray<Real> old_trial_step(nvars, 3.);
    dotk::StdArray<Real> new_steepest_descent(nvars, 1.);
    dotk::StdArray<Real> old_steepest_descent(nvars, 5.);

    Real tolerance = 1e-8;
    Real gold = dotk::nlcg::hestenesStiefel(new_steepest_descent, old_steepest_descent, old_trial_step);
    EXPECT_NEAR(0.333333333333333, gold, tolerance);
}

TEST(DOTk_NLCG, daiYuan)
{
    size_t nvars = 10;
    dotk::StdArray<Real> old_trial_step(nvars, 3.);
    dotk::StdArray<Real> new_steepest_descent(nvars, 1.);
    dotk::StdArray<Real> old_steepest_descent(nvars, 5.);

    Real tolerance = 1e-8;
    Real gold = dotk::nlcg::daiYuan(new_steepest_descent, old_steepest_descent, old_trial_step);
    EXPECT_NEAR(-0.083333333333333, gold, tolerance);
}

TEST(DOTk_NLCG, liuStorey)
{
    size_t nvars = 10;
    dotk::StdArray<Real> old_trial_step(nvars, 3.);
    dotk::StdArray<Real> new_steepest_descent(nvars, 1.);
    dotk::StdArray<Real> old_steepest_descent(nvars, 5.);

    Real tolerance = 1e-8;
    Real gold = dotk::nlcg::liuStorey(new_steepest_descent, old_steepest_descent, old_trial_step);
    EXPECT_NEAR(-0.266666666666666, gold, tolerance);
}

TEST(DOTk_NLCG, conjugateDescent)
{
    size_t nvars = 10;
    dotk::StdArray<Real> old_trial_step(nvars, 3.);
    dotk::StdArray<Real> new_steepest_descent(nvars, 1.);
    dotk::StdArray<Real> old_steepest_descent(nvars, 5.);

    Real tolerance = 1e-8;
    Real gold = dotk::nlcg::conjugateDescent(new_steepest_descent, old_steepest_descent, old_trial_step);
    EXPECT_NEAR(0.021081851067789197, gold, tolerance);
}

TEST(DOTk_GcmmaTestOperators, inequality)
{
    dotk::DOTk_GcmmaTestInequalityConstraint operators;
    EXPECT_EQ(1, operators.bound());

    size_t nvars = 5;
    Real tolerance = 1e-8;
    dotk::StdArray<Real> control(nvars, 1.);
    EXPECT_NEAR(125, operators.value(control), tolerance);
    EXPECT_NEAR(124, operators.residual(control), tolerance);

    dotk::StdArray<Real> gradient(nvars);
    operators.gradient(control, gradient);

    dotk::StdArray<Real> gold(nvars);
    gold[0] = -183;
    gold[1] = -111;
    gold[2] = -57;
    gold[3] = -21;
    gold[4] = -3;
    dotk::gtest::checkResults(gradient, gold);
}

TEST(DOTk_GcmmaTestOperators, objective)
{
    size_t nvars = 5;
    Real tolerance = 1e-8;
    dotk::StdArray<Real> control(nvars, 1.);
    dotk::DOTk_GcmmaTestObjectiveFunction operators;
    EXPECT_NEAR(0.312, operators.value(control), tolerance);

    dotk::StdArray<Real> gradient(nvars);
    operators.gradient(control, gradient);

    dotk::StdArray<Real> gold(nvars);
    gold[0] = 0.0624;
    gold[1] = 0.0624;
    gold[2] = 0.0624;
    gold[3] = 0.0624;
    gold[4] = 0.0624;
    dotk::gtest::checkResults(gradient, gold);
}

TEST(DOTk_RoutinesTypeLP, gcmma)
{
    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    dotk::DOTk_RoutinesTypeLP assembly(objective, inequality);

    size_t nvars = 5;
    Real tolerance = 1e-8;
    dotk::DOTk_Primal primal;
    primal.allocateSerialControlArray(nvars, 1.);
    EXPECT_NEAR(0.312, assembly.objective(primal.control()), tolerance);

    std::shared_ptr<dotk::Vector<Real> > gradient = primal.control()->clone();
    assembly.gradient(primal.control(), gradient);

    std::shared_ptr<dotk::Vector<Real> > gold = primal.control()->clone();
    (*gold)[0] = 0.0624;
    (*gold)[1] = 0.0624;
    (*gold)[2] = 0.0624;
    (*gold)[3] = 0.0624;
    (*gold)[4] = 0.0624;
    dotk::gtest::checkResults(*gradient, *gold);

    const size_t INEQUALITY_INDEX = 0;
    EXPECT_EQ(1, assembly.inequalityBound(INEQUALITY_INDEX));
    EXPECT_NEAR(125, assembly.inequalityValue(INEQUALITY_INDEX, primal.control()), tolerance);

    gradient->fill(0);
    assembly.inequalityGradient(INEQUALITY_INDEX, primal.control(), gradient);
    (*gold)[0] = -183;
    (*gold)[1] = -111;
    (*gold)[2] = -57;
    (*gold)[3] = -21;
    (*gold)[4] = -3;
    dotk::gtest::checkResults(*gradient, *gold);
}

TEST(DOTk_DataMngCCSA, computeFunctionGradients)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    dotk::DOTk_DataMngCCSA mng(primal, objective, inequality);

    mng.computeFunctionGradients();
    EXPECT_EQ(1, mng.getGradientEvaluationCounter());
    EXPECT_EQ(1, mng.getInequalityConstraintGradientCounter());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.0624;
    (*gold)[1] = 0.0624;
    (*gold)[2] = 0.0624;
    (*gold)[3] = 0.0624;
    (*gold)[4] = 0.0624;
    dotk::gtest::checkResults(*mng.m_CurrentObjectiveGradient, *gold);

    (*gold)[0] = -183;
    (*gold)[1] = -111;
    (*gold)[2] = -57;
    (*gold)[3] = -21;
    (*gold)[4] = -3;
    dotk::gtest::checkResults(*mng.m_CurrentInequalityGradients->basis(0), *gold);
}

TEST(DOTk_DataMngCCSA, evaluateFunctionValues)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    dotk::DOTk_DataMngCCSA mng(primal, objective, inequality);

    mng.evaluateFunctionValues();
    EXPECT_EQ(1, mng.getObjectiveFunctionEvaluationCounter());

    Real tolerance = 1e-8;
    EXPECT_NEAR(124, (*mng.m_CurrentFeasibilityMeasures)[0], tolerance);
    EXPECT_NEAR(0.312, mng.m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(124, (*mng.m_CurrentInequalityResiduals)[0], tolerance);

    EXPECT_NEAR(0, mng.m_InitialAuxiliaryVariableZ, tolerance);
    EXPECT_NEAR(124, (*mng.m_InputAuxiliaryVariablesY)[0], tolerance);

    EXPECT_NEAR(0.312, mng.evaluateObjectiveFunction(primal->control()), tolerance);
}

TEST(DOTk_DataMngCCSA, evaluateInequalityConstraint)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    dotk::DOTk_DataMngCCSA mng(primal, objective, inequality);

    std::shared_ptr<dotk::Vector<Real> > residual = primal->dual()->clone();
    std::shared_ptr<dotk::Vector<Real> > feasibility_measure = primal->dual()->clone();
    mng.evaluateInequalityConstraints(primal->control(), residual, feasibility_measure);

    Real tolerance = 1e-8;
    EXPECT_NEAR(124, (*residual)[0], tolerance);
    EXPECT_NEAR(124, (*feasibility_measure)[0], tolerance);
}

TEST(DOTk_DualObjectiveFunctionMMA, updateMovingAsymptotes)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    dotk::DOTk_DualObjectiveFunctionMMA dual_objective(mng);

    mng->m_CurrentSigma->fill(0.1);
    dual_objective.updateMovingAsymptotes(mng->m_CurrentControl, mng->m_CurrentSigma);
    std::shared_ptr<dotk::Vector<Real> > lower_asymptote = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > upper_asymptote = primal->control()->clone();
    dual_objective.gatherMovingAsymptotes(lower_asymptote, upper_asymptote);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(0.9);
    dotk::gtest::checkResults(*lower_asymptote, *gold);
    gold->fill(1.1);
    dotk::gtest::checkResults(*upper_asymptote, *gold);
}

TEST(DOTk_DualObjectiveFunctionMMA, updateTrialControlBounds)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    dotk::DOTk_DualObjectiveFunctionMMA dual_objective(mng);

    mng->m_CurrentSigma->fill(0.1);
    dual_objective.updateTrialControlBounds(0.5, mng->m_CurrentControl, mng->m_CurrentSigma);
    std::shared_ptr<dotk::Vector<Real> > trial_control_lower_bound = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > trial_control_upper_bound = primal->control()->clone();
    dual_objective.gatherTrialControlBounds(trial_control_lower_bound, trial_control_upper_bound);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(0.95);
    dotk::gtest::checkResults(*trial_control_lower_bound, *gold);
    gold->fill(1.05);
    dotk::gtest::checkResults(*trial_control_upper_bound, *gold);
}

TEST(DOTk_DualObjectiveFunctionMMA, updateObjectiveCoefficientVectors)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    Real globalization_scale = 0.5;
    mng->m_CurrentSigma->fill(0.1);
    mng->m_CurrentObjectiveGradient->fill(1.);
    dotk::DOTk_DualObjectiveFunctionMMA dual_objective(mng);
    dual_objective.updateObjectiveCoefficientVectors(globalization_scale,
                                                     mng->m_CurrentSigma,
                                                     mng->m_CurrentObjectiveGradient);

    Real r_coefficient = 0;
    std::shared_ptr<dotk::Vector<Real> > p_coefficients = mng->m_CurrentSigma->clone();
    std::shared_ptr<dotk::Vector<Real> > q_coefficients = p_coefficients->clone();
    dual_objective.gatherObjectiveCoefficients(p_coefficients, q_coefficients, r_coefficient);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(0.0225);
    dotk::gtest::checkResults(*gold, *p_coefficients);
    gold->fill(0.0125);
    dotk::gtest::checkResults(*gold, *q_coefficients);

    Real tolerance = 1e-8;
    EXPECT_NEAR(-1.75, r_coefficient, tolerance);
}

TEST(DOTk_DualObjectiveFunctionMMA, updateInequalityCoefficientVectors)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 1);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    mng->m_CurrentSigma->fill(0.1);
    mng->m_CurrentInequalityGradients->basis(0)->fill(1);
    (*mng->m_CurrentInequalityGradients->basis(0))[0] = 2.;
    dotk::DOTk_DualObjectiveFunctionMMA dual_objective(mng);

    std::shared_ptr<dotk::Vector<Real> > globalization_scale = primal->dual()->clone();
    globalization_scale->fill(0.5);
    dual_objective.updateInequalityCoefficientVectors(globalization_scale,
                                                      mng->m_CurrentSigma,
                                                      mng->m_CurrentInequalityGradients);

    std::shared_ptr<dotk::matrix<Real> > p_coefficients = mng->m_CurrentInequalityGradients->clone();
    std::shared_ptr<dotk::matrix<Real> > q_coefficients = p_coefficients->clone();
    std::shared_ptr<dotk::Vector<Real> > r_coefficients = primal->dual()->clone();
    dual_objective.gatherInequalityCoefficients(p_coefficients, q_coefficients, r_coefficients);

    std::shared_ptr<dotk::Vector<Real> > gold1 = primal->control()->clone();
    gold1->fill(0.0225);
    (*gold1)[0] = 0.0325;
    dotk::gtest::checkResults(*gold1, *p_coefficients->basis(0));
    gold1->fill(0.0125);
    dotk::gtest::checkResults(*gold1, *q_coefficients->basis(0));

    std::shared_ptr<dotk::Vector<Real> > gold2 = primal->dual()->clone();
    gold2->fill(-1.85);
    dotk::gtest::checkResults(*gold2, *r_coefficients);
}

TEST(DOTk_DualObjectiveFunctionMMA, evaluate)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    mng->m_CurrentSigma->fill(0.1);
    mng->m_CurrentObjectiveGradient->fill(1.);
    mng->m_CurrentInequalityGradients->basis(0)->fill(1);
    (*mng->m_CurrentInequalityGradients)(0, 0) = 2.;
    dotk::DOTk_DualObjectiveFunctionMMA dual_objective(mng);

    std::shared_ptr<dotk::Vector<Real> > residual = primal->dual()->clone();
    residual->fill(0.2);
    dual_objective.setCurrentObjectiveFunctionValue(0.1);
    dual_objective.setCurrentInequalityConstraintResiduals(residual);
    dual_objective.updateMovingAsymptotes(mng->m_CurrentControl, mng->m_CurrentSigma);
    dual_objective.updateTrialControlBounds(0.5, mng->m_CurrentControl, mng->m_CurrentSigma);

    Real objective_globalization_scale = 0.5;
    dual_objective.updateObjectiveCoefficientVectors(objective_globalization_scale,
                                                     mng->m_CurrentSigma,
                                                     mng->m_CurrentObjectiveGradient);

    std::shared_ptr<dotk::Vector<Real> > inequality_globalization_scale = primal->dual()->clone();
    inequality_globalization_scale->fill(0.5);
    dual_objective.updateInequalityCoefficientVectors(inequality_globalization_scale,
                                                      mng->m_CurrentSigma,
                                                      mng->m_CurrentInequalityGradients);

    Real tolerance = 1e-6;
    Real value = dual_objective.value(*mng->m_Dual);
    EXPECT_NEAR(-0.21245071085, value, tolerance);

    dual_objective.gatherTrialControl(mng->m_WorkVector);
    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.980539949569856;
    (*gold)[1] = 0.985410196624969;
    (*gold)[2] = 0.985410196624969;
    (*gold)[3] = 0.985410196624969;
    (*gold)[4] = 0.985410196624969;
    dotk::gtest::checkResults(*gold, *mng->m_WorkVector);
}

TEST(DOTk_DualObjectiveFunctionMMA, gradient)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    mng->m_CurrentSigma->fill(0.1);
    mng->m_CurrentObjectiveGradient->fill(1.);
    mng->m_CurrentInequalityGradients->basis(0)->fill(1);
    (*mng->m_CurrentInequalityGradients)(0, 0) = 2.;
    dotk::DOTk_DualObjectiveFunctionMMA dual_objective(mng);

    std::shared_ptr<dotk::Vector<Real> > residual = primal->dual()->clone();
    residual->fill(0.2);
    dual_objective.setCurrentObjectiveFunctionValue(0.1);
    dual_objective.setCurrentInequalityConstraintResiduals(residual);
    dual_objective.updateMovingAsymptotes(mng->m_CurrentControl, mng->m_CurrentSigma);
    dual_objective.updateTrialControlBounds(0.5, mng->m_CurrentControl, mng->m_CurrentSigma);

    Real objective_globalization_scale = 0.5;
    dual_objective.updateObjectiveCoefficientVectors(objective_globalization_scale,
                                                     mng->m_CurrentSigma,
                                                     mng->m_CurrentObjectiveGradient);

    std::shared_ptr<dotk::Vector<Real> > work = primal->dual()->clone();
    work->fill(0.5);
    dual_objective.updateInequalityCoefficientVectors(work, mng->m_CurrentSigma, mng->m_CurrentInequalityGradients);

    std::shared_ptr<dotk::Vector<Real> > trial_control = mng->m_CurrentSigma->clone();
    (*trial_control)[0] = 0.980539949569856;
    (*trial_control)[1] = 0.985410196624969;
    (*trial_control)[2] = 0.985410196624969;
    (*trial_control)[3] = 0.985410196624969;
    (*trial_control)[4] = 0.985410196624969;
    dual_objective.setTrialControl(trial_control);

    std::shared_ptr<dotk::Vector<Real> > gradient = primal->dual()->clone();
    dual_objective.gradient(*mng->m_Dual, *gradient);

    std::shared_ptr<dotk::Vector<Real> > gold = gradient->clone();
    (*gold)[0] = -0.14808035198;

    dotk::gtest::checkResults(*gradient, *gold);
}

TEST(CCSA, computeResidualNorm)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0.1);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    mng->m_CurrentObjectiveGradient->fill(1.);
    mng->m_CurrentInequalityGradients->basis(0)->fill(1);
    (*mng->m_CurrentInequalityGradients)(0, 0) = 2.;
    mng->m_CurrentObjectiveFunctionValue = 0.1;
    mng->m_CurrentInequalityResiduals->fill(1.);

    Real tolerance = 1e-6;
    Real value = dotk::ccsa::computeResidualNorm(mng->m_CurrentControl, mng->m_Dual, mng);
    EXPECT_NEAR(1.02215458713, value, tolerance);
}

TEST(DOTk_DualSolverNLCG, setFunctions)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0.1);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    dotk::DOTk_DualSolverNLCG solver(primal);

    Real tolerance = 1e-6;
    EXPECT_NEAR(1e-8, solver.getGradientTolerance(), tolerance);
    solver.setGradientTolerance(0.1);
    EXPECT_NEAR(0.1, solver.getGradientTolerance(), tolerance);

    EXPECT_NEAR(1e-8, solver.getObjectiveStagnationTolerance(), tolerance);
    solver.setObjectiveStagnationTolerance(0.2);
    EXPECT_NEAR(0.2, solver.getObjectiveStagnationTolerance(), tolerance);

    EXPECT_NEAR(1e-8, solver.getTrialStepTolerance(), tolerance);
    solver.setTrialStepTolerance(0.3);
    EXPECT_NEAR(0.3, solver.getTrialStepTolerance(), tolerance);
}

TEST(DOTk_DualSolverNLCG, solve)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0.1);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    mng->m_CurrentSigma->fill(0.1);
    mng->m_CurrentObjectiveGradient->fill(1.);
    mng->m_CurrentInequalityGradients->fill(1);
    (*mng->m_CurrentInequalityGradients)(0, 0) = 2.;
    mng->m_CurrentObjectiveFunctionValue = 0.1;
    mng->m_CurrentInequalityResiduals->fill(0.2);
    std::shared_ptr<dotk::DOTk_DualObjectiveFunctionMMA> dual_objective(new dotk::DOTk_DualObjectiveFunctionMMA(mng));

    std::shared_ptr<dotk::Vector<Real> > work = mng->m_Dual->clone();
    work->fill(0.2);
    dual_objective->setCurrentObjectiveFunctionValue(0.1);
    dual_objective->setCurrentInequalityConstraintResiduals(work);
    dual_objective->updateMovingAsymptotes(mng->m_CurrentControl, mng->m_CurrentSigma);
    dual_objective->updateTrialControlBounds(0.5, mng->m_CurrentControl, mng->m_CurrentSigma);

    Real value = 0.5;
    dual_objective->updateObjectiveCoefficientVectors(value, mng->m_CurrentSigma, mng->m_CurrentObjectiveGradient);

    work->fill(0.5);
    dual_objective->updateInequalityCoefficientVectors(work, mng->m_CurrentSigma, mng->m_CurrentInequalityGradients);

    dotk::DOTk_DualSolverNLCG solver(primal);
    EXPECT_EQ(5u, solver.getMaxNumLineSearchIterations());
    EXPECT_TRUE(dotk::ccsa::dual_solver_t::NONLINEAR_CG == solver.getDualSolverType());
    EXPECT_TRUE(dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG == solver.getNonlinearCgType());
    solver.setMaxNumIterations(2);
    solver.solve(dual_objective, mng->m_Dual);

    Real tolerance = 1e-6;
    EXPECT_NEAR(-0.14530537595, solver.getNewObjectiveFunctionValue(), tolerance);
    EXPECT_NEAR(-0.12321665321, solver.getOldObjectiveFunctionValue(), tolerance);

    work->fill(0.54726572433);
    dotk::gtest::checkResults(*mng->m_Dual, *work);
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::MAX_NUMBER_ITERATIONS, solver.getStoppingCriterion());
}

TEST(DOTk_SubProblemGCMMA, setFunctions)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 0.1);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlUpperBound(1.);
    primal->setControlLowerBound(1e-2);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    dotk::DOTk_SubProblemGCMMA sub_problem(mng, solver);
    EXPECT_TRUE(dotk::ccsa::subproblem_t::GCMMA == sub_problem.type());

    EXPECT_EQ(10u, sub_problem.getMaxNumIterations());
    sub_problem.setMaxNumIterations(11);
    EXPECT_EQ(11u, sub_problem.getMaxNumIterations());

    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-6, sub_problem.getResidualTolerance(), tolerance);
    sub_problem.setResidualTolerance(0.2);
    EXPECT_NEAR(0.2, sub_problem.getResidualTolerance(), tolerance);
    EXPECT_NEAR(1e-6, sub_problem.getStagnationTolerance(), tolerance);
    sub_problem.setStagnationTolerance(0.1);
    EXPECT_NEAR(0.1, sub_problem.getStagnationTolerance(), tolerance);
}

TEST(DOTk_SubProblemGCMMA, solve)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals, 1);
    primal->allocateSerialControlArray(nvars, 1.);
    primal->setControlLowerBound(1e-2);
    primal->setControlUpperBound(1.);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);
    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    const size_t INEQUALITY_INDEX = 0;
    mng->m_CurrentSigma->fill(0.1);
    objective->gradient(*mng->m_CurrentControl, *mng->m_CurrentObjectiveGradient);
    inequality[INEQUALITY_INDEX]->gradient(*mng->m_CurrentControl, *mng->m_CurrentInequalityGradients->basis(0));
    mng->m_CurrentObjectiveFunctionValue = objective->value(*mng->m_CurrentControl);
    (*mng->m_CurrentInequalityResiduals)[0] = shared_ptr->residual(*mng->m_CurrentControl);

    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    dotk::DOTk_SubProblemGCMMA sub_problem(mng, solver);
    EXPECT_TRUE(dotk::ccsa::subproblem_t::GCMMA == sub_problem.type());

    sub_problem.solve(mng);

    std::shared_ptr<dotk::Vector<Real> > gold1 = mng->m_Dual->clone();
    gold1->fill(1112.2374766);
    dotk::gtest::checkResults(*mng->m_Dual, *gold1);

    std::shared_ptr<dotk::Vector<Real> > gold2 = mng->m_CurrentControl->clone();
    gold2->fill(1.);
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold2);
}

TEST(DOTk_AlgorithmCCSA, solve_GCMMA_POLAK_RIBIERE)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    std::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA gcmma(mng, sub_problem);

    gcmma.getMin();
    EXPECT_EQ(12u, gcmma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, gcmma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.4468784912, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.3399567957, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(-6.866958023e-7, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.016287053522432;
    (*gold)[1] = 5.3096539954232265;
    (*gold)[2] = 4.4954886342217941;
    (*gold)[3] = 3.5000846183148426;
    (*gold)[4] = 2.1521522971543781;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_GCMMA_FLETCHER_REEVES)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setFletcherReevesNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA gcmma(mng, sub_problem);

    gcmma.getMin();
    EXPECT_EQ(12u, gcmma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, gcmma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.4467803129, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.3399556592, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(1.72710027657e-6, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.015693551244798;
    (*gold)[1] = 5.30958525991438;
    (*gold)[2] = 4.4952182677445149;
    (*gold)[3] = 3.5003956317338818;
    (*gold)[4] = 2.1527556749379109;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_GCMMA_DAI_YUAN)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setDaiYuanNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA gcmma(mng, sub_problem);

    gcmma.getMin();
    EXPECT_EQ(12u, gcmma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, gcmma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.4467783954, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.3399564044, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(2.29338057433e-8, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.015752285;
    (*gold)[1] = 5.309502472;
    (*gold)[2] = 4.495089604;
    (*gold)[3] = 3.500513276;
    (*gold)[4] = 2.152802689;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_GCMMA_HESTENES_STIEFEL)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setHestenesStiefelNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA gcmma(mng, sub_problem);

    gcmma.getMin();
    EXPECT_EQ(12u, gcmma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, gcmma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.44678010273, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.33995640349, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(2.86183319264e-8, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.015743553;
    (*gold)[1] = 5.309507776;
    (*gold)[2] = 4.495102284;
    (*gold)[3] = 3.500501301;
    (*gold)[4] = 2.152805395;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_GCMMA_CONJUGATE_DESCENT)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setConjugateDescentNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA gcmma(mng, sub_problem);

    gcmma.getMin();
    EXPECT_EQ(13u, gcmma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::CONTROL_STAGNATION, gcmma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.44689903491, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.33995660855, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(-4.26152497734e-8, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.0158710431;
    (*gold)[1] = 5.3099054573;
    (*gold)[2] = 4.4959970968;
    (*gold)[3] = 3.4995459167;
    (*gold)[4] = 2.1523440844;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_GCMMA_LIU_STOREY)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setLiuStoreyNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA gcmma(mng, sub_problem);

    gcmma.getMin();
    EXPECT_EQ(12u, gcmma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, gcmma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.446878505283, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.3399567959, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(-6.870411412e-7, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.0162869366;
    (*gold)[1] = 5.3096540580;
    (*gold)[2] = 4.4954887341;
    (*gold)[3] = 3.5000843624;
    (*gold)[4] = 2.1521525100;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_MMA_ConjugateDescent)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setConjugateDescentNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA mma(mng, sub_problem);

    mma.getMin();
    EXPECT_EQ(19u, mma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, mma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.44667073112, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.33997913608, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(-5.08261684745e-5, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.01532392933;
    (*gold)[1] = 5.30900262735;
    (*gold)[2] = 4.49447525851;
    (*gold)[3] = 3.50277254484;
    (*gold)[4] = 2.15245025669;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_MMA_DaiYuan)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setDaiYuanNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA mma(mng, sub_problem);

    mma.getMin();
    EXPECT_EQ(85u, mma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, mma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.44684628351, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.3399732863, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(-3.773019762e-5, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.0151966393;
    (*gold)[1] = 5.3096300483;
    (*gold)[2] = 4.4952059704;
    (*gold)[3] = 3.5019096029;
    (*gold)[4] = 2.1519886091;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_MMA_LIU_STOREY)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setLiuStoreyNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA mma(mng, sub_problem);

    mma.getMin();
    EXPECT_EQ(19u, mma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, mma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.446641485, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.339956430, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(2.5781232704e-9, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.01523880535;
    (*gold)[1] = 5.30876116579;
    (*gold)[2] = 4.49426749121;
    (*gold)[3] = 3.50270529890;
    (*gold)[4] = 2.15268797651;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_MMA_FletcherReeves)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setFletcherReevesNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA mma(mng, sub_problem);

    mma.getMin();
    EXPECT_EQ(19u, mma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, mma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.4466415497, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.339956432, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(-1.234440649e-8, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.0152634849;
    (*gold)[1] = 5.3087762636;
    (*gold)[2] = 4.4942687299;
    (*gold)[3] = 3.5026652062;
    (*gold)[4] = 2.1526870871;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_MMA_HestenesStiefel)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    solver->setHestenesStiefelNLCG();
    std::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA mma(mng, sub_problem);

    mma.getMin();
    EXPECT_EQ(19u, mma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, mma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.4466403107, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.3399558642, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(1.2592072382e-6, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.0152623266;
    (*gold)[1] = 5.3087750620;
    (*gold)[2] = 4.4942658653;
    (*gold)[3] = 3.5026642039;
    (*gold)[4] = 2.1526842120;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

TEST(DOTk_AlgorithmCCSA, solve_MMA_PolakRibiere)
{
    size_t nvars = 5;
    size_t nduals = 1;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(nvars, 5.);
    primal->setControlLowerBound(1e-3);
    primal->setControlUpperBound(10);

    std::shared_ptr<dotk::DOTk_GcmmaTestObjectiveFunction> objective(new dotk::DOTk_GcmmaTestObjectiveFunction);
    std::shared_ptr<dotk::DOTk_GcmmaTestInequalityConstraint> shared_ptr(new dotk::DOTk_GcmmaTestInequalityConstraint);
    std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > inequality(1, shared_ptr);

    std::shared_ptr<dotk::DOTk_DataMngCCSA> mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));
    std::shared_ptr<dotk::DOTk_DualSolverNLCG> solver(new dotk::DOTk_DualSolverNLCG(primal));
    std::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(mng, solver));
    dotk::DOTk_AlgorithmCCSA mma(mng, sub_problem);

    mma.getMin();

    EXPECT_EQ(19u, mma.getIterationCount());
    EXPECT_EQ(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE, mma.getStoppingCriterion());

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.44664137457, (*mng->m_Dual)[0], tolerance);
    EXPECT_NEAR(1.33995639388, mng->m_CurrentObjectiveFunctionValue, tolerance);
    EXPECT_NEAR(8.3386932692e-8, (*mng->m_CurrentInequalityResiduals)[0], tolerance);

    std::shared_ptr<dotk::Vector<Real> > gold = mng->m_CurrentControl->clone();
    (*gold)[0] = 6.0152391513;
    (*gold)[1] = 5.3087612741;
    (*gold)[2] = 4.4942672453;
    (*gold)[3] = 3.5027047699;
    (*gold)[4] = 2.1526877176;
    dotk::gtest::checkResults(*mng->m_CurrentControl, *gold);
}

}
