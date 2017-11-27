/*
 * Locus_OptimalityCriteriaTest.cpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "Locus_UnitTestUtils.hpp"

#include "Locus_DataFactory.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_OptimalityCriteria.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_OptimalityCriteriaDataMng.hpp"
#include "Locus_OptimalityCriteriaStageMng.hpp"
#include "Locus_SynthesisOptimizationSubProblem.hpp"
#include "Locus_SingleConstraintSubProblemTypeLP.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"
#include "Locus_OptimalityCriteriaTestObjectiveOne.hpp"
#include "Locus_OptimalityCriteriaTestObjectiveTwo.hpp"
#include "Locus_OptimalityCriteriaTestInequalityOne.hpp"
#include "Locus_OptimalityCriteriaTestInequalityTwo.hpp"

namespace LocusTest
{

TEST(LocusTest, OptimalityCriteriaObjectiveTest)
{
    locus::StandardVectorReductionOperations<double,size_t> tInterface;
    locus::OptimalityCriteriaTestObjectiveOne<double,size_t> tObjective(tInterface);

    size_t tNumVectors = 1;
    size_t tNumElements = 5;
    std::vector<double> tData(tNumElements, 0.);
    locus::StandardMultiVector<double,size_t> tControl(tNumVectors, tData);

    // ********* Set Control Data For Test *********
    const size_t tVectorIndex = 0;
    tData = { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        tControl(tVectorIndex, tIndex) = tData[tIndex];
    }

    // ********* Test Objective Value *********
    double tObjectiveValue = tObjective.value(tControl);

    double tTolerance = 1e-6;
    double tGoldValue = 1.3401885069;
    EXPECT_NEAR(tObjectiveValue, tGoldValue, tTolerance);

    // ********* Test Objective Gradient *********
    locus::StandardMultiVector<double,size_t> tGradient(tNumVectors, tData);
    tObjective.gradient(tControl, tGradient);

    std::vector<double> tGoldGradient(tNumElements, 0.);
    std::fill(tGoldGradient.begin(), tGoldGradient.end(), 0.0624);
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        EXPECT_NEAR(tGradient(tVectorIndex, tIndex), tGoldGradient[tIndex], tTolerance);
    }
}

TEST(LocusTest, OptimalityCriteriaInequalityTestOne)
{
    locus::OptimalityCriteriaTestInequalityOne<double,size_t> tInequality;

    size_t tNumVectors = 1;
    size_t tNumElements = 5;
    std::vector<double> tData(tNumElements, 0.);
    locus::StandardMultiVector<double,size_t> tControl(tNumVectors, tData);

    // ********* Set Control Data For Test *********
    const size_t tVectorIndex = 0;
    tData = { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        tControl(tVectorIndex, tIndex) = tData[tIndex];
    }

    double tInequalityValue = tInequality.value(tControl);

    double tTolerance = 1e-6;
    double tGoldValue = -5.07057774290498e-6;
    EXPECT_NEAR(tInequalityValue, tGoldValue, tTolerance);

    // ********* Test Inequality Gradient *********
    locus::StandardMultiVector<double,size_t> tGradient(tNumVectors, tData);
    tInequality.gradient(tControl, tGradient);

    std::vector<double> tGoldGradient =
            { -0.13778646890793422, -0.14864537557631985, -0.13565219858574704, -0.1351771199123859, -0.13908690613190111 };
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        EXPECT_NEAR(tGradient(tVectorIndex, tIndex), tGoldGradient[tIndex], tTolerance);
    }
}

TEST(LocusTest, OptimalityCriteriaDataMng)
{
    // ********* Test Factories for Dual Data *********
    size_t tNumVectors = 1;
    locus::DataFactory<double,size_t> tFactory;

    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 10;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 5;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double,size_t> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    locus::OptimalityCriteriaDataMng<double,size_t> tDataMng(tFactory);
    double tValue = 23;
    tDataMng.setCurrentObjectiveValue(tValue);

    double tGold = tDataMng.getCurrentObjectiveValue();
    double tTolerance = 1e-6;
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tValue = 24;
    tDataMng.setPreviousObjectiveValue(tValue);
    tGold = tDataMng.getPreviousObjectiveValue();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // ********* Test Dual Functions *********
    tValue = 0.23;
    size_t tIndex = 0;
    tDataMng.setCurrentDual(tIndex, tValue);
    tGold = 0.23;
    EXPECT_NEAR(tDataMng.getCurrentDual()[tIndex], tGold, tTolerance);

    tValue = 0.345;
    tDataMng.setCurrentConstraintValue(tIndex, tValue);
    tGold = 0.345;
    EXPECT_NEAR(tDataMng.getCurrentConstraintValues()[tIndex], tGold, tTolerance);

    // ********* Test Initial Guess Functions *********
    tValue = 0.18;
    locus::StandardMultiVector<double,size_t> tInitialGuess(tNumVectors, tNumControls, tValue);
    tDataMng.setInitialGuess(tInitialGuess);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tInitialGuess, tTolerance);

    tValue = 0.44;
    size_t tVectorIndex = 0;
    tInitialGuess[tVectorIndex].fill(tValue);
    tDataMng.setInitialGuess(tVectorIndex, tInitialGuess[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tInitialGuess.operator [](tVectorIndex));

    tValue = 0.07081982;
    tInitialGuess[tVectorIndex].fill(tValue);
    tDataMng.setInitialGuess(tVectorIndex, tValue);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tInitialGuess.operator [](tVectorIndex));

    tValue = 0.10111983;
    tInitialGuess[tVectorIndex].fill(tValue);
    tDataMng.setInitialGuess(tValue);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tInitialGuess.operator [](tVectorIndex));

    // ********* Test Control Functions *********
    tValue = 0.08;
    locus::StandardMultiVector<double,size_t> tCurrentControl(tNumVectors, tNumControls, tValue);
    tDataMng.setCurrentControl(tCurrentControl);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tCurrentControl, tTolerance);

    tValue = 0.11;
    tCurrentControl[tVectorIndex].fill(tValue);
    tDataMng.setCurrentControl(tVectorIndex, tCurrentControl[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tCurrentControl.operator [](tVectorIndex));

    tValue = 0.09;
    locus::StandardMultiVector<double,size_t> tPreviousControl(tNumVectors, tNumControls, tValue);
    tDataMng.setPreviousControl(tPreviousControl);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tPreviousControl, tTolerance);

    tValue = 0.21;
    tPreviousControl[tVectorIndex].fill(tValue);
    tDataMng.setPreviousControl(tVectorIndex, tPreviousControl[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tPreviousControl.operator [](tVectorIndex));

    // ********* Test Objective Gradient Functions *********
    tValue = 0.88;
    locus::StandardMultiVector<double,size_t> tObjectiveGradient(tNumVectors, tNumControls, tValue);
    tDataMng.setObjectiveGradient(tObjectiveGradient);
    LocusTest::checkMultiVectorData(tDataMng.getObjectiveGradient(), tObjectiveGradient, tTolerance);

    tValue = 0.91;
    tObjectiveGradient[tVectorIndex].fill(tValue);
    tDataMng.setObjectiveGradient(tVectorIndex, tObjectiveGradient[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getObjectiveGradient(tVectorIndex), tObjectiveGradient.operator [](tVectorIndex));

    // ********* Test Inequality Gradient Functions *********
    tValue = 0.68;
    locus::StandardMultiVector<double,size_t> tInequalityGradient(tNumVectors, tNumControls, tValue);
    tDataMng.setInequalityGradient(tInequalityGradient);
    LocusTest::checkMultiVectorData(tDataMng.getInequalityGradient(), tInequalityGradient, tTolerance);

    tValue = 0.61;
    tInequalityGradient[tVectorIndex].fill(tValue);
    tDataMng.setInequalityGradient(tVectorIndex, tInequalityGradient[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getInequalityGradient(tVectorIndex), tInequalityGradient.operator [](tVectorIndex));

    // ********* Test Control Lower Bounds Functions *********
    tValue = 1e-3;
    locus::StandardMultiVector<double,size_t> tLowerBounds(tNumVectors, tNumControls, tValue);
    tDataMng.setControlLowerBounds(tLowerBounds);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tLowerBounds, tTolerance);

    tValue = 1e-2;
    tLowerBounds[tVectorIndex].fill(tValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tLowerBounds[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tLowerBounds.operator [](tVectorIndex));

    tValue = -1;
    tDataMng.setControlLowerBounds(tVectorIndex, tValue);
    tLowerBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tLowerBounds.operator [](tVectorIndex));

    tValue = 0.5;
    tDataMng.setControlLowerBounds(tVectorIndex, tValue);
    tLowerBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tLowerBounds.operator [](tVectorIndex));

    // ********* Test Control Upper Bounds Functions *********
    tValue = 1;
    locus::StandardMultiVector<double,size_t> tUpperBounds(tNumVectors, tNumControls, tValue);
    tDataMng.setControlUpperBounds(tUpperBounds);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tUpperBounds, tTolerance);

    tValue = 0.99;
    tUpperBounds[tVectorIndex].fill(tValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tUpperBounds[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tUpperBounds.operator [](tVectorIndex));

    tValue = 10;
    tDataMng.setControlUpperBounds(tVectorIndex, tValue);
    tUpperBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tUpperBounds.operator [](tVectorIndex));

    tValue = 8;
    tDataMng.setControlUpperBounds(tVectorIndex, tValue);
    tUpperBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tUpperBounds.operator [](tVectorIndex));

    // ********* Test Compute Stagnation Measure Functions *********
    tCurrentControl[tVectorIndex].fill(1.5);
    tDataMng.setCurrentControl(tCurrentControl);
    tPreviousControl[tVectorIndex].fill(4.0);
    tDataMng.setPreviousControl(tPreviousControl);
    tDataMng.computeStagnationMeasure();

    tGold = 2.5;
    tValue = tDataMng.getStagnationMeasure();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // ********* Test Compute Max Inequality Value Functions *********
    tDataMng.computeMaxInequalityValue();
    tGold = 0.345;
    tValue = tDataMng.getMaxInequalityValue();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // ********* Test Compute Max Inequality Value Functions *********
    tDataMng.computeNormObjectiveGradient();
    tGold = 2.0348218595248;
    tValue = tDataMng.getNormObjectiveGradient();
    EXPECT_NEAR(tValue, tGold, tTolerance);
}

TEST(LocusTest, OptimalityCriteriaStageMngSimpleTest)
{
    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumVectors = 1;
    locus::DataFactory<double,size_t> tFactory;
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 1;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 5;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double,size_t> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Optimality Criteria Data Manager *********
    locus::OptimalityCriteriaDataMng<double,size_t> tDataMng(tFactory);

    // ********* Allocate Stage Manager *********
    locus::CriterionList<double, size_t> tInequalityList;
    locus::OptimalityCriteriaTestInequalityOne<double,size_t> tInequality;
    tInequalityList.add(tInequality);
    locus::OptimalityCriteriaTestObjectiveOne<double,size_t> tObjective(tReductionOperations);
    locus::OptimalityCriteriaStageMng<double,size_t> tStageMng(tFactory, tObjective, tInequalityList);

    // ********* Test Update Function *********
    std::vector<double> tData =
        { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    locus::StandardVector<double,size_t> tControl(tData);

    size_t tVectorIndex = 0;
    tDataMng.setCurrentControl(tVectorIndex, tControl);
    tStageMng.update(tDataMng);

    double tTolerance = 1e-6;
    double tGoldValue = 1.3401885069;
    double tObjectiveValue = tDataMng.getCurrentObjectiveValue();
    EXPECT_NEAR(tObjectiveValue, tGoldValue, tTolerance);

    std::fill(tData.begin(), tData.end(), 0.0624);
    locus::StandardVector<double,size_t> tGoldObjectiveGradient(tData);
    LocusTest::checkVectorData(tDataMng.getObjectiveGradient(tVectorIndex), tGoldObjectiveGradient);

    const size_t tConstraintIndex = 0;
    tGoldValue = -5.07057774290498e-6;
    EXPECT_NEAR(tDataMng.getCurrentConstraintValues(tConstraintIndex), tGoldValue, tTolerance);

    tData = { -0.13778646890793422, -0.14864537557631985, -0.13565219858574704, -0.1351771199123859, -0.13908690613190111 };
    locus::StandardVector<double,size_t> tGoldInequalityGradient(tData);
    LocusTest::checkVectorData(tDataMng.getInequalityGradient(tVectorIndex), tGoldInequalityGradient);
}

TEST(LocusTest, SynthesisOptimizationSubProblem)
{
    // ********* NOTE: Default OrdinalType = size_t *********
    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumVectors = 1;
    locus::DataFactory<double> tFactory;
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 1;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 2;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Optimality Criteria Data Manager *********
    locus::OptimalityCriteriaDataMng<double> tDataMng(tFactory);

    // ********* Allocate Synthesis Optimization Sub-Problem  *********
    locus::SynthesisOptimizationSubProblem<double> tSubProblem(tFactory);

    double tGold = 1e-4;
    double tValue = tSubProblem.getBisectionTolerance();
    double tTolerance = 1e-6;
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setBisectionTolerance(1e-1);
    tGold = 0.1;
    tValue = tSubProblem.getBisectionTolerance();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 0.2;
    tValue = tSubProblem.getMoveLimit();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setMoveLimit(0.15);
    tGold = 0.15;
    tValue = tSubProblem.getMoveLimit();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 0.5;
    tValue = tSubProblem.getDampingPower();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setDampingPower(0.25);
    tGold = 0.25;
    tValue = tSubProblem.getDampingPower();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 0;
    tValue = tSubProblem.getDualLowerBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setDualLowerBound(0.35);
    tGold = 0.35;
    tValue = tSubProblem.getDualLowerBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 1e7;
    tValue = tSubProblem.getDualUpperBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setDualUpperBound(0.635);
    tGold = 0.635;
    tValue = tSubProblem.getDualUpperBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // NOTE: I NEED TO UNIT TEST SUBPROBLEM WITH PHYSICS-BASED CRITERIA
}

TEST(LocusTest, OptimalityCriteria)
{
    // ********* NOTE: Default OrdinalType = size_t *********

    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumVectors = 1;
    locus::DataFactory<double> tFactory;
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 1;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 2;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Optimality Criteria Data Manager *********
    std::shared_ptr<locus::OptimalityCriteriaDataMng<double>> tDataMng =
            std::make_shared<locus::OptimalityCriteriaDataMng<double>>(tFactory);

    // ********* Set Bounds and Initial Guess *********
    double tValue = 0.5;
    tDataMng->setControlLowerBounds(tValue);
    tValue = 10;
    tDataMng->setControlUpperBounds(tValue);
    tValue = 1;
    tDataMng->setInitialGuess(tValue);

    // ********* Allocate Stage Manager *********
    locus::CriterionList<double> tInequalityList;
    locus::OptimalityCriteriaTestInequalityTwo<double> tInequality;
    tInequalityList.add(tInequality);
    locus::OptimalityCriteriaTestObjectiveTwo<double> tObjective;
    std::shared_ptr<locus::OptimalityCriteriaStageMng<double>> tStageMng =
            std::make_shared<locus::OptimalityCriteriaStageMng<double>>(tFactory, tObjective, tInequalityList);

    // ********* Allocate Optimality Criteria Algorithm *********
    std::shared_ptr<locus::SingleConstraintSubProblemTypeLP<double>> tSubProlem =
            std::make_shared<locus::SingleConstraintSubProblemTypeLP<double>>(*tDataMng);
    locus::OptimalityCriteria<double> tOptimalityCriteria(tDataMng, tStageMng, tSubProlem);
    tOptimalityCriteria.solve();

    size_t tVectorIndex = 0;
    const locus::Vector<double> & tControl = tDataMng->getCurrentControl(tVectorIndex);
    double tTolerance = 1e-6;
    double tGoldControlOne = 0.5;
    EXPECT_NEAR(tControl[0], tGoldControlOne, tTolerance);
    double tGoldControlTwo = 1.375;
    EXPECT_NEAR(tControl[1], tGoldControlTwo, tTolerance);
    size_t tGoldNumIterations = 5;
    EXPECT_EQ(tOptimalityCriteria.getNumIterationsDone(), tGoldNumIterations);
}

} // namespace LocusTest
