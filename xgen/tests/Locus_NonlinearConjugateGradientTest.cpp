/*
 * Locus_NonlinearConjugateGradientTest.cpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <cmath>
#include <limits>

#include "Locus_UnitTestUtils.hpp"

#include "Locus_Rosenbrock.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_NonlinearConjugateGradient.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"
#include "Locus_NonlinearConjugateGradientStageMng.hpp"

namespace LocusTest
{

TEST(LocusTest, NonlinearConjugateGradientDataMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);
    size_t tOrdinalValue = 1;
    EXPECT_EQ(tDataMng.getNumControlVectors(), tOrdinalValue);

    // ********* TEST OBJECTIVE FUNCTION VALUE *********
    const double tTolerance = 1e-6;
    double tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tScalarValue = 45;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarValue = 123;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // ********* TEST INITIAL GUESS FUNCTIONS *********
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    tScalarValue = 2;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tMultiVector(tNumVectors, tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tScalarValue);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.5;
    const size_t tVectorIndex = 0;
    tDataMng.setInitialGuess(tVectorIndex, tScalarValue);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setInitialGuess(tMultiVector);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 0.5;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tVector);

    // ********* TEST TRIAL STEP FUNCTIONS *********
    tScalarValue = 0.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setTrialStep(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tMultiVector);

    tScalarValue = 0.25;
    tVector.fill(tScalarValue);
    tDataMng.setTrialStep(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);

    // ********* TEST CURRENT CONTROL FUNCTIONS *********
    tScalarValue = 1.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentControl(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.25;
    tVector.fill(tScalarValue);
    tDataMng.setCurrentControl(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tVector);

    // ********* TEST PREVIOUS CONTROL FUNCTIONS *********
    tScalarValue = 1.21;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setPreviousControl(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tMultiVector);

    tScalarValue = 1.11;
    tVector.fill(tScalarValue);
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tVector);

    // ********* TEST CURRENT GRADIENT FUNCTIONS *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentGradient(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tMultiVector);

    tScalarValue = 3;
    tVector.fill(tScalarValue);
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentGradient(tVectorIndex), tVector);

    // ********* TEST PREVIOUS GRADIENT FUNCTIONS *********
    tScalarValue = 2.1;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setPreviousGradient(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tMultiVector);

    tScalarValue = 3.1;
    tVector.fill(tScalarValue);
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getPreviousGradient(tVectorIndex), tVector);

    // ********* TEST DEFAULT UPPER AND LOWER BOUNDS *********
    tScalarValue = -std::numeric_limits<double>::max();
    tVector.fill(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tVector);

    tScalarValue = std::numeric_limits<double>::max();
    tVector.fill(tScalarValue);
    locus::fill(tScalarValue,tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tVector);

    // ********* TEST LOWER BOUND FUNCTIONS *********
    tScalarValue = -10;
    tDataMng.setControlLowerBounds(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -9;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setControlLowerBounds(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -8;
    tVector.fill(tScalarValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tVector);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -7;
    tDataMng.setControlLowerBounds(tVectorIndex, tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    // ********* TEST UPPER BOUND FUNCTIONS *********
    tScalarValue = 10;
    tDataMng.setControlUpperBounds(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 9;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setControlUpperBounds(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 8;
    tVector.fill(tScalarValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tVector);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 7;
    tDataMng.setControlUpperBounds(tVectorIndex, tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    // ********* TEST COMPUTE CONTROL STAGNATION MEASURE *********
    tScalarValue = 3;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentControl(tMultiVector);
    tVector[0] = 2;
    tVector[1] = 2.5;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tDataMng.computeStagnationMeasure();
    tScalarValue = 1.;
    EXPECT_NEAR(tDataMng.getStagnationMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE OBJECTIVE STAGNATION MEASURE *********
    tScalarValue = 1.25;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    tScalarValue = 0.75;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    tDataMng.computeObjectiveStagnationMeasure();
    tScalarValue = 0.5;
    EXPECT_NEAR(tDataMng.getObjectiveStagnationMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE NORM GRADIENT *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentGradient(tMultiVector);
    tDataMng.computeNormGradient();
    tScalarValue = std::sqrt(2.);
    EXPECT_NEAR(tDataMng.getNormGradient(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE STATIONARITY MEASURE *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setTrialStep(tMultiVector);
    tDataMng.computeStationarityMeasure();
    tScalarValue = std::sqrt(8.);
    EXPECT_NEAR(tDataMng.getStationarityMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE STATIONARITY MEASURE *********
    tDataMng.storePreviousState();
    tScalarValue = 1.25;
    EXPECT_NEAR(tDataMng.getPreviousObjectiveFunctionValue(), tScalarValue, tTolerance);
    tScalarValue = 1;
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tMultiVector);
    tScalarValue = 3;
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tMultiVector);
}

TEST(LocusTest, NonlinearConjugateGradientStandardStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Test Evaluate Objective Function *********
    double tScalarValue = 2;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tScalarValue = 401;
    double tTolerance = 1e-6;
    size_t tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveFunctionEvaluations(), tOrdinalValue);
    EXPECT_NEAR(tStageMng.evaluateObjective(tControl), tScalarValue, tTolerance);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveFunctionEvaluations(), tOrdinalValue);

    // ********* Test Compute Gradient *********
    tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveGradientEvaluations(), tOrdinalValue);
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tStageMng.computeGradient(tControl, tGradient);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveGradientEvaluations(), tOrdinalValue);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 0) = 1602;
    tGoldVector(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // ********* Test Apply Vector to Hessian *********
    tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveHessianEvaluations(), tOrdinalValue);
    tScalarValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tScalarValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tStageMng.applyVectorToHessian(tControl, tVector, tHessianTimesVector);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveHessianEvaluations(), tOrdinalValue);
    tGoldVector(tVectorIndex, 0) = 3202;
    tGoldVector(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, PolakRibiere)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Polak-Ribiere Direction *********
    locus::PolakRibiere<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -103.4;
    tVector[1] = -250.8;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, FletcherReeves)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 11;
    tVector[1] = -22;
    tDataMng.setCurrentSteepestDescent(tVectorIndex, tVector);

    // ********* Allocate Fletcher-Reeves Direction *********
    locus::FletcherReeves<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -110;
    tVector[1] = -264;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, HestenesStiefel)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -3;
    tVector[1] = -2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 3;
    tVector[1] = 2;
    tDataMng.setCurrentSteepestDescent(tVectorIndex, tVector);

    // ********* Allocate Hestenes-Stiefel Direction *********
    locus::HestenesStiefel<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 1.3333333333333333;
    tVector[1] = -1.333333333333333;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, ConjugateDescent)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 11;
    tVector[1] = -22;
    tDataMng.setCurrentSteepestDescent(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setPreviousSteepestDescent(tVectorIndex, tVector);
    tVector[0] = -8;
    tVector[1] = -21;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Conjugate Descent Direction *********
    locus::ConjugateDescent<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -7;
    tVector[1] = -23;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiLiao)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -1;
    tVector[1] = 2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tVector[0] = 2;
    tVector[1] = 3;
    tDataMng.setCurrentControl(tVectorIndex, tVector);

    // ********* Allocate Dai-Liao Direction *********
    locus::DaiLiao<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 0.05;
    tVector[1] = -3.9;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, PerryShanno)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tVector[0] = 2;
    tVector[1] = 3;
    tDataMng.setCurrentControl(tVectorIndex, tVector);

    // ********* Allocate Perry-Shanno Direction *********
    locus::PerryShanno<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -0.419267707083;
    tVector[1] = -0.722989195678;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, LiuStorey)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Liu-Storey Direction *********
    locus::LiuStorey<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -103.4;
    tVector[1] = -250.8;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, HagerZhang)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::HagerZhang<double> tDirection(tDataFactory);
    // TEST 1: SCALE FACTOR SELECTED
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -14.367346938775;
    tVector[1] = -72.734693877551;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
    // TEST 2: SCALE FACTOR NOT SELECTED, LOWER BOUND USED INSTEAD
    tVector.fill(1e-1);
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 11;
    tVector[1] = -22;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiYuan)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -1;
    tVector[1] = 2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::DaiYuan<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -1.5;
    tVector[1] = -7;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiYuanHybrid)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::DaiYuanHybrid<double> tDirection(tDataFactory);
    // TEST 1: SCALED STEP
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 0.19642857142857;
    tVector[1] = -43.607142857142;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
    // TEST 2: UNSCALED STEP
    tVector[0] = -12;
    tVector[1] = -23;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 11;
    tVector[1] = 22;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 12.067522825323;
    tVector[1] = 8.009932778168;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, NonlinearConjugateGradientStateMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(tDataFactory);

    // ********* Allocate Nonlinear Conjugate Gradient State Manager *********
    locus::NonlinearConjugateGradientStateMng<double> tStateMng(tDataMng, tStageMng);

    // ********* Test Set Trial Step Function *********
    double tScalarValue = 0.1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tMultiVector(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setTrialStep(tMultiVector);
    LocusTest::checkMultiVectorData(tStateMng.getTrialStep(), tMultiVector);

    // ********* Test Set Current Control Function *********
    tScalarValue = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setCurrentControl(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getCurrentControl(), tControl);

    // ********* Test Set Current Control Function *********
    tScalarValue = 3;
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setCurrentGradient(tGradient);
    LocusTest::checkMultiVectorData(tStateMng.getCurrentGradient(), tGradient);

    // ********* Test Set Control Lower Bounds Function *********
    tScalarValue = std::numeric_limits<double>::min();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlLowerBounds(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getControlLowerBounds(), tControl);

    // ********* Test Set Control Upper Bounds Function *********
    tScalarValue = std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlUpperBounds(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getControlUpperBounds(), tControl);

    // ********* Test Evaluate Objective Function *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tControl);
    tScalarValue = 401;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tStateMng.evaluateObjective(tControl), tScalarValue, tTolerance);

    // ********* Test Set Current Objective Function *********
    tStateMng.setCurrentObjectiveValue(tScalarValue);
    EXPECT_NEAR(tStateMng.getCurrentObjectiveValue(), tScalarValue, tTolerance);

    // ********* Test Compute Gradient Function *********
    tScalarValue = 0;
    locus::fill(tScalarValue, tGradient);
    tStateMng.computeGradient(tControl, tGradient);
    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGold(tVectorIndex, 0) = 1602;
    tGold(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGold);

    // ********* Test Apply Vector to Hessian Function *********
    tScalarValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tScalarValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tStateMng.applyVectorToHessian(tControl, tVector, tHessianTimesVector);
    tGold(tVectorIndex, 0) = 3202;
    tGold(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGold);
}


TEST(LocusTest, NonlinearConjugateGradient_PolakRibiere_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    size_t tOrdinalValue = 37;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_PolakRibiere_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    size_t tOrdinalValue = 56;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_FletcherReeves_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setFletcherReevesMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 74;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_FletcherReeves_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setFletcherReevesMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 63;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HestenesStiefel_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHestenesStiefelMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 23;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HestenesStiefel_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHestenesStiefelMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 36;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HagerZhang_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHagerZhangMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 56;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HagerZhang_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHagerZhangMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 49;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuanHybrid_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanHybridMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 32;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuanHybrid_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanHybridMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 63;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-6;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuan_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 28;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_PerryShanno_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setPerryShannoMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 33;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_STEP, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_ConjugateDescentMethod_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setConjugateDescentMethod(tDataFactory.operator*());
    tScalarValue = 1e-3;
    tAlgorithm.setGradientTolerance(tScalarValue);
    tAlgorithm.solve();

    size_t tOrdinalValue = 35;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_ConjugateDescentMethod_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setConjugateDescentMethod(tDataFactory.operator*());
    tScalarValue = 1e-5;
    tAlgorithm.setGradientTolerance(tScalarValue);
    tAlgorithm.solve();

    size_t tOrdinalValue = 326;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_LiuStorey_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setLiuStoreyMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 50;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_LiuStorey_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setLiuStoreyMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 55;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_Daniels_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDanielsMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 28;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

} // namespace LocusTest
