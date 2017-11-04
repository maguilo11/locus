/*
 * Locus_ConservativeConvexSeparableApproximationTest.cpp
 *
 *  Created on: Jun 14, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "Locus_UnitTestUtils.hpp"

#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_MultiVectorList.hpp"
#include "Locus_CcsaTestObjective.hpp"
#include "Locus_CcsaTestInequality.hpp"
#include "Locus_DualProblemStageMng.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_PrimalProblemStageMng.hpp"
#include "Locus_MethodMovingAsymptotes.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"
#include "Locus_ConservativeConvexSeparableAppxAlgorithm.hpp"
#include "Locus_GloballyConvergentMethodMovingAsymptotes.hpp"

namespace LocusTest
{

TEST(LocusTest, CcsaTestObjective)
{
    // ********* Allocate Data Factory *********
    locus::StandardVectorReductionOperations<double> tReduction;

    // ********* Allocate Criterion *********
    locus::CcsaTestObjective<double> tCriterion(tReduction);

    const size_t tNumState = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tDummyState(tNumVectors, tNumState);
    double tScalarValue = 1;
    const size_t tNumControls = 5;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);

    tScalarValue = 0.312;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tCriterion.value(tDummyState, tControl), tTolerance);

    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tDummyState, tControl, tGradient);
    tScalarValue = 0.0624;
    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls, tScalarValue);
    LocusTest::checkMultiVectorData(tGradient, tGold);
}

TEST(LocusTest, CcsaTestInequality)
{
    // ********* Allocate Criterion *********
    locus::CcsaTestInequality<double> tCriterion;

    const size_t tNumState = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tDummyState(tNumVectors, tNumState);
    double tScalarValue = 1;
    const size_t tNumControls = 5;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);

    tScalarValue = 124;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tCriterion.value(tDummyState, tControl), tTolerance);

    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tDummyState, tControl, tGradient);
    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls);
    tGold(0,0) = -183;
    tGold(0,1) = -111;
    tGold(0,2) = -57;
    tGold(0,3) = -21;
    tGold(0,4) = -3;
    LocusTest::checkMultiVectorData(tGradient, tGold);
}

TEST(LocusTest, ConservativeConvexSeparableAppxDataMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    locus::ConservativeConvexSeparableAppxDataMng<double> tDataMng(tDataFactory);

    // ********* TEST INTEGER AND SCALAR PARAMETERS *********
    size_t tOrdinalValue = 1;
    EXPECT_EQ(tOrdinalValue, tDataMng.getNumDualVectors());
    EXPECT_EQ(tOrdinalValue, tDataMng.getNumConstraints());
    EXPECT_EQ(tOrdinalValue, tDataMng.getNumControlVectors());

    double tScalarValue = 0.5;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tDataMng.getDualProblemBoundsScaleFactor(), tTolerance);
    tDataMng.setDualProblemBoundsScaleFactor(0.25);
    tScalarValue = 0.25;
    EXPECT_NEAR(tScalarValue, tDataMng.getDualProblemBoundsScaleFactor(), tTolerance);

    // ********* TEST CONSTRAINT GLOBALIZATION FACTORS *********
    tScalarValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tDualMultiVector(tNumVectors, tNumDuals, tScalarValue);
    LocusTest::checkMultiVectorData(tDualMultiVector, tDataMng.getConstraintGlobalizationFactors());
    tScalarValue = 2;
    locus::fill(tScalarValue, tDualMultiVector);
    tDataMng.setConstraintGlobalizationFactors(tDualMultiVector);
    tScalarValue = 0.5;
    locus::StandardVector<double> tDualVector(tNumDuals, tScalarValue);
    const size_t tVectorIndex = 0;
    tDataMng.setConstraintGlobalizationFactors(tVectorIndex, tDualVector);
    LocusTest::checkVectorData(tDualVector, tDataMng.getConstraintGlobalizationFactors(tVectorIndex));

    // ********* TEST OBJECTIVE FUNCTION *********
    tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarValue = 0.2;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tScalarValue = 0.25;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // ********* TEST INITIAL GUESS *********
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    tScalarValue = 0;
    locus::StandardMultiVector<double> tControlMultiVector(tNumVectors, tNumControls, tScalarValue);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 2;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setInitialGuess(tControlMultiVector);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 5;
    tDataMng.setInitialGuess(tScalarValue);
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 3;
    locus::StandardVector<double> tControlVector(tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentControl(tVectorIndex));

    tScalarValue = 33;
    tDataMng.setInitialGuess(tVectorIndex, tScalarValue);
    tControlVector.fill(tScalarValue);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentControl(tVectorIndex));

    // ********* TEST SET DUAL FUNCTIONS *********
    tScalarValue = 11;
    locus::fill(tScalarValue, tDualMultiVector);
    tDataMng.setDual(tDualMultiVector);
    LocusTest::checkMultiVectorData(tDualMultiVector, tDataMng.getDual());

    tScalarValue = 21;
    tDualVector.fill(tScalarValue);
    tDataMng.setDual(tVectorIndex, tDualVector);
    LocusTest::checkVectorData(tDualVector, tDataMng.getDual(tVectorIndex));

    // ********* TEST TRIAL STEP FUNCTIONS *********
    tScalarValue = 12;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setTrialStep(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getTrialStep());

    tScalarValue = 22;
    tControlVector.fill(tScalarValue);
    tDataMng.setTrialStep(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getTrialStep(tVectorIndex));

    // ********* TEST ACTIVE SET FUNCTIONS *********
    tScalarValue = 10;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setActiveSet(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getActiveSet());

    tScalarValue = 20;
    tControlVector.fill(tScalarValue);
    tDataMng.setActiveSet(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getActiveSet(tVectorIndex));

    // ********* TEST INACTIVE SET FUNCTIONS *********
    tScalarValue = 11;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setInactiveSet(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getInactiveSet());

    tScalarValue = 21;
    tControlVector.fill(tScalarValue);
    tDataMng.setInactiveSet(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getInactiveSet(tVectorIndex));

    // ********* TEST CURRENT CONTROL FUNCTIONS *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentControl(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentControl());

    tScalarValue = 2;
    tControlVector.fill(tScalarValue);
    tDataMng.setCurrentControl(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentControl(tVectorIndex));

    // ********* TEST PREVIOUS CONTROL FUNCTIONS *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setPreviousControl(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getPreviousControl());

    tScalarValue = 3;
    tControlVector.fill(tScalarValue);
    tDataMng.setPreviousControl(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getPreviousControl(tVectorIndex));

    // ********* TEST CURRENT GRADIENT FUNCTIONS *********
    tScalarValue = 3;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentObjectiveGradient(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentObjectiveGradient());

    tScalarValue = 4;
    tControlVector.fill(tScalarValue);
    tDataMng.setCurrentObjectiveGradient(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentObjectiveGradient(tVectorIndex));

    // ********* TEST CURRENT SIGMA FUNCTIONS *********
    tScalarValue = 5;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentSigma(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentSigma());

    tScalarValue = 6;
    tControlVector.fill(tScalarValue);
    tDataMng.setCurrentSigma(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentSigma(tVectorIndex));

    // ********* TEST CONTROL LOWER BOUNDS FUNCTIONS *********
    tScalarValue = 6;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setControlLowerBounds(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getControlLowerBounds());

    tScalarValue = 9;
    tDataMng.setControlLowerBounds(tScalarValue);
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getControlLowerBounds());

    tScalarValue = 7;
    tControlVector.fill(tScalarValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getControlLowerBounds(tVectorIndex));

    tScalarValue = 8;
    tControlVector.fill(tScalarValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tScalarValue);
    LocusTest::checkVectorData(tControlVector, tDataMng.getControlLowerBounds(tVectorIndex));

    // ********* TEST CONTROL UPPER BOUNDS FUNCTIONS *********
    tScalarValue = 61;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setControlUpperBounds(tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getControlUpperBounds());

    tScalarValue = 91;
    tDataMng.setControlUpperBounds(tScalarValue);
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getControlUpperBounds());

    tScalarValue = 71;
    tControlVector.fill(tScalarValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getControlUpperBounds(tVectorIndex));

    tScalarValue = 81;
    tControlVector.fill(tScalarValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tScalarValue);
    LocusTest::checkVectorData(tControlVector, tDataMng.getControlUpperBounds(tVectorIndex));

    // ********* TEST CONSTRAINT VALUES FUNCTIONS *********
    tScalarValue = 61;
    locus::fill(tScalarValue, tDualMultiVector);
    tDataMng.setCurrentConstraintValues(tDualMultiVector);
    LocusTest::checkMultiVectorData(tDualMultiVector, tDataMng.getCurrentConstraintValues());

    tScalarValue = 91;
    tDualVector.fill(tScalarValue);
    const size_t tConstraintIndex = 0;
    tDataMng.setCurrentConstraintValues(tConstraintIndex, tDualVector);
    LocusTest::checkVectorData(tDualVector, tDataMng.getCurrentConstraintValues(tConstraintIndex));

    // ********* TEST CONSTRAINT GRADIENT FUNCTIONS *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentConstraintGradients(tConstraintIndex, tControlMultiVector);
    LocusTest::checkMultiVectorData(tControlMultiVector, tDataMng.getCurrentConstraintGradients(tConstraintIndex));
    const locus::MultiVectorList<double> & tCurrentConstraintGradientList = tDataMng.getCurrentConstraintGradients();
    LocusTest::checkMultiVectorData(tControlMultiVector, tCurrentConstraintGradientList[tConstraintIndex]);

    tScalarValue = 9;
    tControlVector.fill(tScalarValue);
    tDataMng.setCurrentConstraintGradients(tConstraintIndex, tVectorIndex, tControlVector);
    LocusTest::checkVectorData(tControlVector, tDataMng.getCurrentConstraintGradients(tConstraintIndex, tVectorIndex));

    // ********* TEST COMPUTE STAGNATION MEASURE *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentControl(tControlMultiVector);
    tScalarValue = 4;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setPreviousControl(tControlMultiVector);
    tDataMng.computeStagnationMeasure();
    tScalarValue = 3;
    EXPECT_NEAR(tScalarValue, tDataMng.getStagnationMeasure(), tTolerance);

    // ********* TEST COMPUTE INACTIVE VECTOR NORM *********
    tScalarValue = 1;
    tControlVector.fill(tScalarValue);
    tControlVector[0] = 0;
    tDataMng.setInactiveSet(tVectorIndex, tControlVector);
    tScalarValue = 4;
    locus::fill(tScalarValue, tControlMultiVector);
    EXPECT_NEAR(tScalarValue, tDataMng.computeInactiveVectorNorm(tControlMultiVector), tTolerance);

    tScalarValue = 8;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.setCurrentObjectiveGradient(tControlMultiVector);
    tDataMng.computeNormInactiveGradient();
    tScalarValue = 4;
    EXPECT_NEAR(tScalarValue, tDataMng.getNormInactiveGradient(), tTolerance);

    // ********* TEST OBJECTIVE STAGNATION FUNCTION *********
    tScalarValue = 0.5;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    tScalarValue = 1.25;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    tScalarValue = 0.75;
    tDataMng.computeObjectiveStagnationMeasure();
    EXPECT_NEAR(tScalarValue, tDataMng.getObjectiveStagnationMeasure(), tTolerance);

    // ********* TEST FEASIBILITY MEASURE *********
    tScalarValue = 0.5;
    tDualVector.fill(tScalarValue);
    tDataMng.setCurrentConstraintValues(tVectorIndex, tDualVector);
    tDataMng.computeFeasibilityMeasure();
    EXPECT_NEAR(tScalarValue, tDataMng.getFeasibilityMeasure(), tTolerance);
}

TEST(LocusTest, PrimalProblemStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 5;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Criteria *********
    locus::CriterionList<double> tInequalityList;
    locus::CcsaTestInequality<double> tInequality;
    tInequalityList.add(tInequality);
    locus::CcsaTestObjective<double> tObjective(tDataFactory.getControlReductionOperations());

    // ********* Allocate Primal Stage Manager *********
    locus::PrimalProblemStageMng<double> tStageMng(tDataFactory, tObjective, tInequalityList);

    // ********* TEST OBJECTIVE EVALUATION *********
    size_t tOrdinalValue = 0;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumObjectiveFunctionEvaluations());
    double tScalarValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tScalarValue = 0.312;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tStageMng.evaluateObjective(tControl), tTolerance);
    tOrdinalValue = 1;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumObjectiveFunctionEvaluations());

    // ********* TEST CONSTRAINT EVALUATION *********
    tOrdinalValue = 0;
    const size_t tConstraintIndex = 0;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumConstraintEvaluations(tConstraintIndex));
    locus::StandardMultiVector<double> tConstraints(tNumVectors, tNumDuals);
    tStageMng.evaluateConstraints(tControl, tConstraints);
    tScalarValue = 124;
    locus::StandardMultiVector<double> tDualMultiVector(tNumVectors, tNumDuals, tScalarValue);
    LocusTest::checkMultiVectorData(tConstraints, tDualMultiVector);
    tOrdinalValue = 1;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumConstraintEvaluations(tConstraintIndex));

    // ********* TEST OBJECTIVE GRADIENT *********
    tOrdinalValue = 0;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumObjectiveGradientEvaluations());
    locus::StandardMultiVector<double> tObjectiveGradient(tNumVectors, tNumControls);
    tStageMng.computeGradient(tControl, tObjectiveGradient);
    tScalarValue = 0.0624;
    locus::StandardMultiVector<double> tControlMultiVector(tNumVectors, tNumControls, tScalarValue);
    LocusTest::checkMultiVectorData(tObjectiveGradient, tControlMultiVector);
    tOrdinalValue = 1;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumObjectiveGradientEvaluations());

    // ********* TEST CONSTRAINT GRADIENT *********
    tOrdinalValue = 0;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumConstraintGradientEvaluations(tConstraintIndex));
    locus::MultiVectorList<double> tConstraintGradients;
    tConstraintGradients.add(tControlMultiVector);
    tStageMng.computeConstraintGradients(tControl, tConstraintGradients);
    tControlMultiVector(0, 0) = -183;
    tControlMultiVector(0, 1) = -111;
    tControlMultiVector(0, 2) = -57;
    tControlMultiVector(0, 3) = -21;
    tControlMultiVector(0, 4) = -3;
    LocusTest::checkMultiVectorData(tConstraintGradients[tConstraintIndex], tControlMultiVector);
    tOrdinalValue = 1;
    EXPECT_EQ(tOrdinalValue, tStageMng.getNumConstraintGradientEvaluations(tConstraintIndex));
}

TEST(LocusTest, DualProblemStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 5;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Data Manager and Test Data *********
    locus::ConservativeConvexSeparableAppxDataMng<double> tDataMng(tDataFactory);
    double tScalarValue = 1e-2;
    tDataMng.setControlLowerBounds(tScalarValue);
    tScalarValue = 1;
    tDataMng.setControlUpperBounds(tScalarValue);
    tScalarValue = 0.1;
    tDataMng.setCurrentSigma(tScalarValue);
    tScalarValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControlMultiVector(tNumVectors, tNumControls, tScalarValue);
    tDataMng.setCurrentControl(tControlMultiVector);
    tDataMng.setCurrentObjectiveGradient(tControlMultiVector);
    tControlMultiVector(0, 0) = 2;
    const size_t tConstraintIndex = 0;
    tDataMng.setCurrentConstraintGradients(tConstraintIndex, tControlMultiVector);
    tScalarValue = 0.1;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    tScalarValue = 0.2;
    locus::StandardMultiVector<double> tDualMultiVector(tNumVectors, tNumDuals, tScalarValue);
    tDataMng.setCurrentConstraintValues(tDualMultiVector);

    // ********* Allocate Dual Problem Data Manager *********
    locus::DualProblemStageMng<double> tDualStageMng(tDataFactory);

    // ********* TEST UPDATE FUNCTION *********
    tDualStageMng.update(tDataMng);
    tScalarValue = 0.9;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getLowerAsymptotes(), tControlMultiVector);
    tScalarValue = 1.1;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getUpperAsymptotes(), tControlMultiVector);
    tScalarValue = 0.95;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getTrialControlLowerBounds(), tControlMultiVector);
    tScalarValue = 1.05;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getTrialControlUpperBounds(), tControlMultiVector);

    // ********* TEST UPDATE OBJECTIVE COEFFICIENTS FUNCTION *********
    tScalarValue = 0.5;
    tDataMng.setDualObjectiveGlobalizationFactor(tScalarValue);
    tDualStageMng.updateObjectiveCoefficients(tDataMng);
    tScalarValue = 0.0225;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getObjectiveCoefficientsP(), tControlMultiVector);
    tScalarValue = 0.0125;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getObjectiveCoefficientsQ(), tControlMultiVector);
    tScalarValue = -1.65;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tDualStageMng.getObjectiveCoefficientsR(), tTolerance);

    // ********* TEST UPDATE CONSTRAINT COEFFICIENTS FUNCTION *********
    tScalarValue = 0.5;
    locus::fill(tScalarValue, tDualMultiVector);
    tDataMng.setConstraintGlobalizationFactors(tDualMultiVector);
    tDualStageMng.updateConstraintCoefficients(tDataMng);
    tScalarValue = 0.0225;
    locus::fill(tScalarValue, tControlMultiVector);
    tControlMultiVector(0,0) = 0.0325;
    LocusTest::checkMultiVectorData(tDualStageMng.getConstraintCoefficientsP(tConstraintIndex), tControlMultiVector);
    tScalarValue = 0.0125;
    locus::fill(tScalarValue, tControlMultiVector);
    LocusTest::checkMultiVectorData(tDualStageMng.getConstraintCoefficientsQ(tConstraintIndex), tControlMultiVector);
    tScalarValue = -1.65;
    EXPECT_NEAR(tScalarValue, tDualStageMng.getObjectiveCoefficientsR(), tTolerance);

    // ********* TEST EVALUATE OBJECTIVE FUNCTION *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tDualMultiVector);
    double tObjectiveValue = tDualStageMng.evaluateObjective(tDualMultiVector);
    tScalarValue = -0.21245071085465561;
    EXPECT_NEAR(tScalarValue, tObjectiveValue, tTolerance);

    // ********* TEST TRIAL CONTROL CALCULATION *********
    tScalarValue = 0;
    locus::fill(tScalarValue, tControlMultiVector);
    tDualStageMng.getTrialControl(tControlMultiVector);
    tScalarValue = 0.985410196624969;
    locus::StandardMultiVector<double> tControlGold(tNumVectors, tNumControls, tScalarValue);
    tControlGold(0,0) = 0.980539949569856;
    LocusTest::checkMultiVectorData(tControlMultiVector, tControlGold);

    // ********* TEST OBJECTIVE GRADIENT FUNCTION *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tDualMultiVector);
    locus::StandardMultiVector<double> tDualGradient(tNumVectors, tNumDuals);
    tDualStageMng.computeGradient(tDualMultiVector, tDualGradient);
    tScalarValue = -0.14808035198890879;
    locus::StandardMultiVector<double> tDualGold(tNumVectors, tNumDuals, tScalarValue);
    LocusTest::checkMultiVectorData(tDualGradient, tDualGold);
}

TEST(LocusTest, ComputeKarushKuhnTuckerConditionsInexactness)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 5;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Data Manager and Test Data *********
    locus::ConservativeConvexSeparableAppxDataMng<double> tDataMng(tDataFactory);
    double tScalarValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControlMultiVector(tNumVectors, tNumControls, tScalarValue);
    tDataMng.setCurrentObjectiveGradient(tControlMultiVector);
    const size_t tConstraintIndex = 0;
    tControlMultiVector(0,0) = 2;
    tDataMng.setCurrentConstraintGradients(tConstraintIndex, tControlMultiVector);
    tScalarValue = 0.1;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    tScalarValue = 1;
    locus::StandardMultiVector<double> tDualMultiVector(tNumVectors, tNumDuals, tScalarValue);
    tDataMng.setCurrentConstraintValues(tDualMultiVector);

    // ********* TEST KARUSH-KUHN-TUCKER MEASURE *********
    tScalarValue = 0.1;
    locus::fill(tScalarValue, tDualMultiVector);
    tScalarValue = 1;
    locus::fill(tScalarValue, tControlMultiVector);
    tDataMng.computeKarushKuhnTuckerConditionsInexactness(tControlMultiVector, tDualMultiVector);
    double tGold = 1.02215458713;
    const double tTolerance = 1e-6;
    tScalarValue = tDataMng.getKarushKuhnTuckerConditionsInexactness();
    EXPECT_NEAR(tScalarValue, tGold, tTolerance);
}

TEST(LocusTest, GCMMA)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 5;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Primal Stage Manager *********
    locus::CcsaTestObjective<double> tObjective(tDataFactory.getControlReductionOperations());
    locus::CcsaTestInequality<double> tInequality;
    locus::CriterionList<double> tInequalityList;
    tInequalityList.add(tInequality);
    std::shared_ptr<locus::PrimalProblemStageMng<double>> tStageMng =
            std::make_shared<locus::PrimalProblemStageMng<double>>(tDataFactory, tObjective, tInequalityList);

    // ********* Allocate Primal Stage Manager *********
    std::shared_ptr<locus::GloballyConvergentMethodMovingAsymptotes<double>> tSubProblem =
            std::make_shared<locus::GloballyConvergentMethodMovingAsymptotes<double>>(tDataFactory);

    // ********* Allocate Data Manager and Test Data *********
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<double>> tDataMng =
            std::make_shared<locus::ConservativeConvexSeparableAppxDataMng<double>>(tDataFactory);
    double tScalarValue = 5;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = 1e-3;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 10;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Optimization Algorithm Data *********
    locus::ConservativeConvexSeparableAppxAlgorithm<double> tAlgorithm(tStageMng, tDataMng, tSubProblem);
    tAlgorithm.solve();

    size_t tOrdinalValue = 20;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::ccsa::stop_t::KKT_CONDITIONS_TOLERANCE, tAlgorithm.getStoppingCriterion());

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControlGold(tNumVectors, tNumControls);
    tControlGold(0,0) = 6.0166415287705108;
    tControlGold(0,1) = 5.3096820989910976;
    tControlGold(0,2) = 4.494714328434684;
    tControlGold(0,3) = 3.5002108656878752;
    tControlGold(0,4) = 2.1524085607477299;
    LocusTest::checkMultiVectorData(tDataMng->getCurrentControl(), tControlGold);

    tScalarValue = 1.3399564241;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tDataMng->getCurrentObjectiveFunctionValue(), tTolerance);

    const size_t tVectorIndex = 0;
    tScalarValue = -7.304985583e-8;
    locus::StandardVector<double> tConstraintGold(tNumDuals, tScalarValue);
    LocusTest::checkVectorData(tDataMng->getCurrentConstraintValues(tVectorIndex), tConstraintGold);

    tScalarValue = 0.4468391431565;
    locus::StandardMultiVector<double> tDualGold(tNumVectors, tNumDuals, tScalarValue);
    LocusTest::checkMultiVectorData(tDataMng->getDual(), tDualGold);
}

TEST(LocusTest, MMA)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 5;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Primal Stage Manager *********
    locus::CcsaTestObjective<double> tObjective(tDataFactory.getControlReductionOperations());
    locus::CcsaTestInequality<double> tInequality;
    locus::CriterionList<double> tInequalityList;
    tInequalityList.add(tInequality);
    std::shared_ptr<locus::PrimalProblemStageMng<double>> tStageMng =
            std::make_shared<locus::PrimalProblemStageMng<double>>(tDataFactory, tObjective, tInequalityList);

    // ********* Allocate Primal Stage Manager *********
    std::shared_ptr<locus::MethodMovingAsymptotes<double>> tSubProblem =
            std::make_shared<locus::MethodMovingAsymptotes<double>>(tDataFactory);

    // ********* Allocate Data Manager and Test Data *********
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<double>> tDataMng =
            std::make_shared<locus::ConservativeConvexSeparableAppxDataMng<double>>(tDataFactory);
    double tScalarValue = 5;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = 1e-3;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 10;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Optimization Algorithm Data *********
    locus::ConservativeConvexSeparableAppxAlgorithm<double> tAlgorithm(tStageMng, tDataMng, tSubProblem);
    tAlgorithm.solve();

    size_t tOrdinalValue = 35;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::ccsa::stop_t::KKT_CONDITIONS_TOLERANCE, tAlgorithm.getStoppingCriterion());

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControlGold(tNumVectors, tNumControls);
    tControlGold(0,0) = 6.0152032114486209;
    tControlGold(0,1) = 5.3089580592476127;
    tControlGold(0,2) = 4.4944130970002663;
    tControlGold(0,3) = 3.5021481844482856;
    tControlGold(0,4) = 2.1529382278989218;
    LocusTest::checkMultiVectorData(tDataMng->getCurrentControl(), tControlGold);

    tScalarValue = 1.3399564241;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarValue, tDataMng->getCurrentObjectiveFunctionValue(), tTolerance);

    const size_t tVectorIndex = 0;
    tScalarValue = -7.304985583e-8;
    locus::StandardVector<double> tConstraintGold(tNumDuals, tScalarValue);
    LocusTest::checkVectorData(tDataMng->getCurrentConstraintValues(tVectorIndex), tConstraintGold);

    tScalarValue = 0.44667576881352539;
    locus::StandardMultiVector<double> tDualGold(tNumVectors, tNumDuals, tScalarValue);
    LocusTest::checkMultiVectorData(tDataMng->getDual(), tDualGold);
}

} // LocusTest
