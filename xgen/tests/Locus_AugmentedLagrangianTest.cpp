/*
 * Locus_AugmentedLagrangianTest.cpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "Locus_UnitTestUtils.hpp"

#include "Locus_Radius.hpp"
#include "Locus_Circle.hpp"
#include "Locus_Rosenbrock.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_AnalyticalHessian.hpp"
#include "Locus_LinearOperatorList.hpp"
#include "Locus_AnalyticalGradient.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_GradientOperatorList.hpp"
#include "Locus_IdentityPreconditioner.hpp"
#include "Locus_TrustRegionAlgorithmDataMng.hpp"
#include "Locus_AugmentedLagrangianStageMng.hpp"
#include "Locus_KelleySachsAugmentedLagrangian.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"

namespace LocusTest
{

TEST(LocusTest, TrustRegionAlgorithmDataMng)
{
    // ********* Test Factories for Dual Data *********
    locus::DataFactory<double> tDataFactory;

    // ********* Allocate Core Optimization Data Templates *********
    const size_t tNumDuals = 10;
    const size_t tNumDualVectors = 2;
    tDataFactory.allocateDual(tNumDuals, tNumDualVectors);
    const size_t tNumStates = 20;
    const size_t tNumStateVectors = 6;
    tDataFactory.allocateState(tNumStates, tNumStateVectors);
    const size_t tNumControls = 5;
    const size_t tNumControlVectors = 3;
    tDataFactory.allocateControl(tNumControls, tNumControlVectors);

    // ********* Allocate Reduction Operations *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateDualReductionOperations(tReductionOperations);
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Trust Region Algorithm Data Manager *********
    locus::TrustRegionAlgorithmDataMng<double> tDataMng(tDataFactory);

    // ********* Test Trust Region Algorithm Data Manager *********
    // TEST NUMBER OF VECTORS FUNCTIONS
    EXPECT_EQ(tNumDualVectors, tDataMng.getNumDualVectors());
    EXPECT_EQ(tNumControlVectors, tDataMng.getNumControlVectors());

    // TEST CURRENT OBJECTIVE FUNCTION VALUE INTERFACES
    double tGoldValue = std::numeric_limits<double>::max();
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tGoldValue = 0.123;
    tDataMng.setCurrentObjectiveFunctionValue(0.123);
    EXPECT_NEAR(tGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    // TEST PREVIOUS OBJECTIVE FUNCTION VALUE INTERFACES
    tGoldValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tGoldValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tGoldValue = 0.321;
    tDataMng.setPreviousObjectiveFunctionValue(0.321);
    EXPECT_NEAR(tGoldValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // 1) TEST INITIAL GUESS INTERFACES
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    double tValue = 0.5;
    tDataMng.setInitialGuess(0.5);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::StandardVector<double> tlocusControlVector(tNumControls, tValue);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    // 2) TEST INITIAL GUESS INTERFACES
    tValue = 0.3;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(0.1);
        tlocusControlVector.fill(tValue);
        tDataMng.setInitialGuess(tVectorIndex, tValue);
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    // 3) TEST INITIAL GUESS INTERFACES
    tValue = 0;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(0.1);
        tlocusControlVector.fill(tValue);
        tDataMng.setInitialGuess(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    // 4) TEST INITIAL GUESS INTERFACES
    locus::StandardMultiVector<double> tlocusControlMultiVector(tNumControlVectors, tlocusControlVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setInitialGuess(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector, tTolerance);

    // TEST DUAL VECTOR INTERFACES
    locus::StandardVector<double> tlocusDualVector(tNumDuals);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusDualVector.fill(tValue);
        tDataMng.setDual(tVectorIndex, tlocusDualVector);
        LocusTest::checkVectorData(tDataMng.getDual(tVectorIndex), tlocusDualVector, tTolerance);
    }

    tValue = 20;
    locus::StandardMultiVector<double> tlocusDualMultiVector(tNumDualVectors, tlocusDualVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusDualMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setDual(tlocusDualMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getDual(), tlocusDualMultiVector, tTolerance);

    // TEST TRIAL STEP INTERFACES
    tValue = 3;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setTrialStep(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setTrialStep(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tlocusControlMultiVector, tTolerance);

    // TEST ACTIVE SET INTERFACES
    tValue = 33;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setActiveSet(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getActiveSet(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setActiveSet(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getActiveSet(), tlocusControlMultiVector, tTolerance);

    // TEST INACTIVE SET INTERFACES
    tValue = 23;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setInactiveSet(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getInactiveSet(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getInactiveSet(), tlocusControlMultiVector, tTolerance);

    // TEST CURRENT CONTROL INTERFACES
    tValue = 30;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setCurrentControl(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector, tTolerance);

    // TEST PREVIOUS CONTROL INTERFACES
    tValue = 80;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setPreviousControl(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setPreviousControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector, tTolerance);

    // TEST CURRENT GRADIENT INTERFACES
    tValue = 7882;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setCurrentGradient(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getCurrentGradient(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector, tTolerance);

    // TEST PREVIOUS GRADIENT INTERFACES
    tValue = 101183;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setPreviousGradient(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getPreviousGradient(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setPreviousGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector, tTolerance);

    // TEST CONTROL LOWER BOUND INTERFACES
    tValue = -std::numeric_limits<double>::max();
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tlocusControlMultiVector, tTolerance);
    tValue = 1e-3;
    tDataMng.setControlLowerBounds(tValue);
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tlocusControlMultiVector, tTolerance);

    tValue = 1e-4;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + (static_cast<double>(tVectorIndex) * tValue);
        tDataMng.setControlLowerBounds(tVectorIndex, tValue);
        tlocusControlVector.fill(tValue);
        LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setControlLowerBounds(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1e-2;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue * static_cast<double>(tVectorIndex + 1);
        tlocusControlMultiVector[tVectorIndex].fill(tValue);
    }
    tDataMng.setControlLowerBounds(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tlocusControlMultiVector, tTolerance);

    // TEST CONTROL UPPER BOUND INTERFACES
    tValue = std::numeric_limits<double>::max();
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tlocusControlMultiVector, tTolerance);
    tValue = 1e-3;
    tDataMng.setControlUpperBounds(tValue);
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tlocusControlMultiVector, tTolerance);

    tValue = 1e-4;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + (static_cast<double>(tVectorIndex) * tValue);
        tDataMng.setControlUpperBounds(tVectorIndex, tValue);
        tlocusControlVector.fill(tValue);
        LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setControlUpperBounds(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1e-2;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue * static_cast<double>(tVectorIndex + 1);
        tlocusControlMultiVector[tVectorIndex].fill(tValue);
    }
    tDataMng.setControlUpperBounds(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tlocusControlMultiVector, tTolerance);

    // TEST GRADIENT INEXACTNESS FLAG FUNCTIONS
    EXPECT_FALSE(tDataMng.isGradientInexactnessToleranceExceeded());
    tDataMng.setGradientInexactnessFlag(true);
    EXPECT_TRUE(tDataMng.isGradientInexactnessToleranceExceeded());
    tDataMng.setGradientInexactnessFlag(false);
    EXPECT_FALSE(tDataMng.isGradientInexactnessToleranceExceeded());

    // TEST OBJECTIVE INEXACTNESS FLAG FUNCTIONS
    EXPECT_FALSE(tDataMng.isObjectiveInexactnessToleranceExceeded());
    tDataMng.setObjectiveInexactnessFlag(true);
    EXPECT_TRUE(tDataMng.isObjectiveInexactnessToleranceExceeded());
    tDataMng.setObjectiveInexactnessFlag(false);
    EXPECT_FALSE(tDataMng.isObjectiveInexactnessToleranceExceeded());

    // TEST COMPUTE STAGNATION MEASURE FUNCTION
    tValue = 1e-2;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        double tCurrentValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tCurrentValue);
        tDataMng.setCurrentControl(tVectorIndex, tlocusControlVector);
        double tPreviousValue = tValue * static_cast<double>(tVectorIndex * tVectorIndex);
        tlocusControlVector.fill(tPreviousValue);
        tDataMng.setPreviousControl(tVectorIndex, tlocusControlVector);
    }
    tValue = 1.97;
    tDataMng.computeStagnationMeasure();
    EXPECT_NEAR(tValue, tDataMng.getStagnationMeasure(), tTolerance);

    // TEST COMPUTE NORM OF PROJECTED VECTOR
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    locus::fill(2., tlocusControlMultiVector);
    tValue = 7.745966692414834;
    EXPECT_NEAR(tValue, tDataMng.computeProjectedVectorNorm(tlocusControlMultiVector), tTolerance);

    locus::fill(1., tlocusControlMultiVector);
    size_t tVectorIndex = 1;
    size_t tElementIndex = 2;
    tlocusControlMultiVector(tVectorIndex, tElementIndex) = 0;
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    locus::fill(2., tlocusControlMultiVector);
    tValue = 7.483314773547883;
    EXPECT_NEAR(tValue, tDataMng.computeProjectedVectorNorm(tlocusControlMultiVector), tTolerance);

    // TEST COMPUTE PROJECTED GRADIENT NORM
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    locus::fill(3., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    tDataMng.computeNormProjectedGradient();
    tValue = 11.61895003862225;
    EXPECT_NEAR(tValue, tDataMng.getNormProjectedGradient(), tTolerance);

    locus::fill(1., tlocusControlMultiVector);
    tVectorIndex = 0;
    tElementIndex = 0;
    tlocusControlMultiVector(tVectorIndex, tElementIndex) = 0;
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    tDataMng.computeNormProjectedGradient();
    tValue = 11.224972160321824;
    EXPECT_NEAR(tValue, tDataMng.getNormProjectedGradient(), tTolerance);

    // TEST COMPUTE STATIONARY MEASURE
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    locus::fill(-2., tlocusControlMultiVector);
    tDataMng.setControlLowerBounds(tlocusControlMultiVector);
    locus::fill(12., tlocusControlMultiVector);
    tDataMng.setControlUpperBounds(tlocusControlMultiVector);
    locus::fill(-1., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    tDataMng.computeStationarityMeasure();
    tValue = 3.872983346207417;
    EXPECT_NEAR(tValue, tDataMng.getStationarityMeasure(), tTolerance);

    // TEST RESET STAGE FUNCTION
    tValue = 1;
    tDataMng.setCurrentObjectiveFunctionValue(tValue);
    EXPECT_NEAR(tValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tValue = 2;
    tDataMng.setPreviousObjectiveFunctionValue(tValue);
    EXPECT_NEAR(tValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector);
    locus::fill(-2., tlocusControlMultiVector);
    tDataMng.setPreviousControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector);
    locus::fill(10., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector);
    locus::fill(-12., tlocusControlMultiVector);
    tDataMng.setPreviousGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector);

    tDataMng.resetCurrentStageDataToPreviousStageData();

    tValue = 2;
    EXPECT_NEAR(tValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    locus::fill(-2., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector);
    locus::fill(-12., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector);

    // TEST STORE CURRENT STAGE DATA FUNCTION
    tValue = 1;
    tDataMng.setCurrentObjectiveFunctionValue(tValue);
    EXPECT_NEAR(tValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tValue = 2;
    EXPECT_NEAR(tValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector);
    locus::fill(-2., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector);
    locus::fill(10., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector);
    locus::fill(-12., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector);

    tDataMng.storeCurrentStageData();

    tValue = 1;
    EXPECT_NEAR(tValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    locus::fill(1., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector);
    locus::fill(10., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector);
}

TEST(LocusTest, RosenbrockCriterion)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    double tValue = 2;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Rosenbrock<double> tCriterion;
    // TEST OBJECTIVE FUNCTION EVALUATION
    double tObjectiveValue = tCriterion.value(tState, tControl);
    const double tGoldValue = 401;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tObjectiveValue, tTolerance);

    // TEST GRADIENT EVALUATION FUNCTION
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 0) = 1602;
    tGoldVector(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST HESSIAN TIMES VECTOR FUNCTION
    tValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tCriterion.hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 3202;
    tGoldVector(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, CircleCriterion)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Circle<double> tCriterion;

    // TEST OBJECTIVE FUNCTION EVALUATION
    double tObjectiveValue = tCriterion.value(tState, tControl);
    const double tGoldValue = 2;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tObjectiveValue, tTolerance);

    // TEST GRADIENT EVALUATION FUNCTION
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 1) = -4;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST HESSIAN TIMES VECTOR FUNCTION
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tCriterion.hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, RadiusCriterion)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 0.5;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Radius<double> tCriterion;

    // TEST OBJECTIVE FUNCTION EVALUATION
    double tObjectiveValue = tCriterion.value(tState, tControl);
    const double tGoldValue = -0.5;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tObjectiveValue, tTolerance);

    // TEST GRADIENT EVALUATION FUNCTION
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    locus::fill(1., tGoldVector);
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST HESSIAN TIMES VECTOR FUNCTION
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    const size_t tVectorIndex = 0;
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tCriterion.hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, AnalyticalGradient)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    const size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Circle<double> tCriterion;
    locus::AnalyticalGradient<double> tGradient(tCriterion);

    // TEST COMPUTE FUNCTION
    tGradient.compute(tState, tControl, tOutput);

    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls);
    tGold(tVectorIndex, 0) = 0.0;
    tGold(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGold);
}

TEST(LocusTest, AnalyticalHessian)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    const size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);

    std::shared_ptr<locus::Circle<double>> tCriterion = std::make_shared<locus::Circle<double>>();
    locus::AnalyticalHessian<double> tHessian(tCriterion);

    // TEST APPLY VECTOR TO HESSIAN OPERATOR FUNCTION
    tHessian.apply(tState, tControl, tVector, tHessianTimesVector);

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, Preconditioner)
{
    locus::IdentityPreconditioner<double> tPreconditioner;

    const double tValue = 1;
    const size_t tNumVectors = 1;
    const size_t tNumControls = 2;
    const size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    // TEST APPLY PRECONDITIONER AND APPLY INVERSE PRECONDITIONER FUNCTIONS
    tPreconditioner.applyInvPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);
    locus::fill(0., tOutput);
    tPreconditioner.applyPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);

    // TEST CREATE FUNCTION
    std::shared_ptr<locus::Preconditioner<double>> tCopy = tPreconditioner.create();
    tCopy->applyInvPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);
    locus::fill(0., tOutput);
    tCopy->applyPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);
}

TEST(LocusTest, CriterionList)
{
    locus::CriterionList<double> tList;
    size_t tGoldInteger = 0;
    EXPECT_EQ(tGoldInteger, tList.size());

    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    tList.add(tCircle);
    tGoldInteger = 1;
    EXPECT_EQ(tGoldInteger, tList.size());
    tList.add(tRadius);
    tGoldInteger = 2;
    EXPECT_EQ(tGoldInteger, tList.size());

    // ** TEST FIRST CRITERION OBJECTIVE **
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    size_t tCriterionIndex = 0;
    double tOutput = tList[tCriterionIndex].value(tState, tControl);

    double tGoldScalar = 2;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);

    // TEST FIRST CRITERION GRADIENT
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tList[tCriterionIndex].gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 1) = -4;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST FIRST CRITERION HESSIAN TIMES VECTOR FUNCTION
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tList[tCriterionIndex].hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);

    // ** TEST SECOND CRITERION OBJECTIVE **
    tCriterionIndex = 1;
    locus::fill(0.5, tControl);
    tOutput = tList[tCriterionIndex].value(tState, tControl);
    tGoldScalar = -0.5;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);

    // TEST SECOND CRITERION GRADIENT
    locus::fill(0., tGradient);
    tList[tCriterionIndex].gradient(tState, tControl, tGradient);
    locus::fill(1., tGoldVector);
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST SECOND HESSIAN TIMES VECTOR FUNCTION
    locus::fill(0.5, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(0., tHessianTimesVector);
    tList[tCriterionIndex].hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);

    // **** TEST CREATE FUNCTION ****
    std::shared_ptr<locus::CriterionList<double>> tCopy = tList.create();
    // FIRST OBJECTIVE
    tCriterionIndex = 0;
    locus::fill(1.0, tControl);
    tOutput = tCopy->operator [](tCriterionIndex).value(tState, tControl);
    tGoldScalar = 2;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);
    // SECOND OBJECTIVE
    tCriterionIndex = 1;
    locus::fill(0.5, tControl);
    tOutput = tCopy->operator [](tCriterionIndex).value(tState, tControl);
    tGoldScalar = -0.5;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);
}

TEST(LocusTest, GradientOperatorList)
{
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;

    locus::AnalyticalGradient<double> tCircleGradient(tCircle);
    locus::AnalyticalGradient<double> tRadiusGradient(tRadius);
    locus::GradientOperatorList<double> tList;

    // ********* TEST ADD FUNCTION *********
    tList.add(tCircleGradient);
    size_t tIntegerGold = 1;
    EXPECT_EQ(tIntegerGold, tList.size());

    tList.add(tRadiusGradient);
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tList.size());

    // ********* ALLOCATE DATA STRUCTURES FOR TEST *********
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    // ********* TEST OPERATOR[] - FIRST CRITERION *********
    size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    size_t tGradientOperatorIndex = 0;
    tList[tGradientOperatorIndex].compute(tState, tControl, tOutput);

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(tVectorIndex, 0) = 0.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST OPERATOR[] - SECOND CRITERION *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 1;
    tList[tGradientOperatorIndex].compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - FIRST CRITERION *********
    tValue = 1.0;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 0;
    tList.ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 0.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - SECOND CRITERION *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 1;
    tList.ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - FIRST CRITERION *********
    std::shared_ptr<locus::GradientOperatorList<double>> tListCopy = tList.create();

    tValue = 1.0;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 0;
    tListCopy->ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 0.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - SECOND CRITERION *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 1;
    tListCopy->ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);
}

TEST(LocusTest, LinearOperatorList)
{
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;

    locus::AnalyticalHessian<double> tCircleHessian(tCircle);
    locus::AnalyticalHessian<double> tRadiusHessian(tRadius);
    locus::LinearOperatorList<double> tList;

    // ********* TEST ADD FUNCTION *********
    tList.add(tCircleHessian);
    size_t tIntegerGold = 1;
    EXPECT_EQ(tIntegerGold, tList.size());

    tList.add(tRadiusHessian);
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tList.size());

    // ********* ALLOCATE DATA STRUCTURES FOR TEST *********
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    // ********* TEST OPERATOR[] - FIRST CRITERION *********
    size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    size_t tLinearOperatorIndex = 0;
    tList[tLinearOperatorIndex].apply(tState, tControl, tVector, tOutput);

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST OPERATOR[] - SECOND CRITERION *********
    locus::fill(0., tOutput);
    tValue = 0.5;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 1;
    tList[tLinearOperatorIndex].apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - FIRST CRITERION *********
    locus::fill(0., tOutput);
    tValue = 1.0;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 0;
    tList.ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - SECOND CRITERION *********
    locus::fill(0., tOutput);
    tValue = 0.5;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 1;
    tList.ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - FIRST CRITERION *********
    std::shared_ptr<locus::LinearOperatorList<double>> tListCopy = tList.create();

    locus::fill(0., tOutput);
    tValue = 1.0;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 0;
    tListCopy->ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - SECOND CRITERION *********
    locus::fill(0., tOutput);
    tValue = 0.5;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 1;
    tListCopy->ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);
}

TEST(LocusTest, AugmentedLagrangianStageMng)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tList;
    tList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    locus::AugmentedLagrangianStageMng<double> tStageMng(tDataFactory, tCircle, tList);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER FUNCTIONALITIES *********
    size_t tIntegerGold = 0;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveFunctionEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveGradientEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveHessianEvaluations());
    for(size_t tIndex = 0; tIndex < tList.size(); tIndex++)
    {
        EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintEvaluations(tIndex));
        EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintHessianEvaluations(tIndex));
        EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintGradientEvaluations(tIndex));
    }

    double tScalarGold = 1;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);
    tScalarGold = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarGold, tStageMng.getNormObjectiveFunctionGradient(), tTolerance);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER OBJECTIVE EVALUATION *********
    double tValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);
    tValue = tStageMng.evaluateObjective(tControl);
    tScalarGold = 2.5;
    EXPECT_NEAR(tScalarGold, tValue, tTolerance);
    tIntegerGold = 1;
    const size_t tConstraintIndex = 0;;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveFunctionEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintEvaluations(tConstraintIndex));

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - UPDATE LAGRANGE MULTIPLIERS *********
    tScalarGold = 1.;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);
    EXPECT_FALSE(tStageMng.updateLagrangeMultipliers());
    locus::StandardMultiVector<double> tLagrangeMultipliers(tNumVectors, tNumDuals);
    tStageMng.getLagrangeMultipliers(tLagrangeMultipliers);
    locus::StandardMultiVector<double> tLagrangeMultipliersGold(tNumVectors, tNumDuals);
    LocusTest::checkMultiVectorData(tLagrangeMultipliers, tLagrangeMultipliersGold);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - UPDATE LAGRANGE MULTIPLIERS & EVALUATE CONSTRAINT *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    tStageMng.evaluateConstraint(tControl);
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintEvaluations(tConstraintIndex));

    tStageMng.updateCurrentConstraintValues();
    locus::StandardMultiVector<double> tCurrentConstraintValues(tNumVectors, tNumDuals);
    tStageMng.getCurrentConstraintValues(tCurrentConstraintValues);
    tValue = -0.5;
    locus::StandardMultiVector<double> tCurrentConstraintValuesGold(tNumVectors, tNumDuals, tValue);
    LocusTest::checkMultiVectorData(tCurrentConstraintValues, tCurrentConstraintValuesGold);

    tScalarGold = 0.2;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);
    EXPECT_FALSE(tStageMng.updateLagrangeMultipliers());
    tStageMng.getLagrangeMultipliers(tLagrangeMultipliers);
    tScalarGold = 0.04;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);

    tValue = -2.5;
    locus::fill(tValue, tLagrangeMultipliersGold);
    LocusTest::checkMultiVectorData(tLagrangeMultipliers, tLagrangeMultipliersGold);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - COMPUTE FEASIBILITY MEASURE *********
    tStageMng.computeCurrentFeasibilityMeasure();
    tScalarGold = 0.5;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentFeasibilityMeasure(), tTolerance);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - COMPUTE GRADIENT *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng.setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng.setConstraintGradients(tGradientList);

    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    tStageMng.computeGradient(tControl, tOutput);
    tScalarGold = 6.0827625302982193;
    EXPECT_NEAR(tScalarGold, tStageMng.getNormObjectiveFunctionGradient(), tTolerance);
    tIntegerGold = 1;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveGradientEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintGradientEvaluations(tConstraintIndex));
    locus::StandardMultiVector<double> tGoldMultiVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldMultiVector(tVectorIndex, 0) = -16;
    tGoldMultiVector(tVectorIndex, 1) = -21;
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - APPLY VECTOR TO HESSIAN *********
    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng.setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng.setConstraintHessians(tHessianList);

    tValue = 1.0;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tStageMng.applyVectorToHessian(tControl, tVector, tOutput);
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveHessianEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintHessianEvaluations(tConstraintIndex));
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintGradientEvaluations(tConstraintIndex));
    tGoldMultiVector(tVectorIndex, 0) = 22.0;
    tGoldMultiVector(tVectorIndex, 1) = 24.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - APPLY PRECONDITIONER *********
    tStageMng.applyVectorToPreconditioner(tControl, tVector, tOutput);
    tGoldMultiVector(tVectorIndex, 0) = 1.0;
    tGoldMultiVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);
    tStageMng.applyVectorToInvPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);
}

TEST(LocusTest, SteihaugTointSolver)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE SOLVER *********
    locus::ProjectedSteihaugTointPcg<double> tSolver(tDataFactory);

    // ********* TEST MAX NUM ITERATIONS FUNCTIONS *********
    size_t tIntegerGold = 200;
    EXPECT_EQ(tIntegerGold, tSolver.getMaxNumIterations());
    tIntegerGold = 300;
    tSolver.setMaxNumIterations(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tSolver.getMaxNumIterations());

    // ********* TEST NUM ITERATIONS DONE FUNCTIONS *********
    tIntegerGold = 0;
    EXPECT_EQ(tIntegerGold, tSolver.getNumIterationsDone());
    tIntegerGold = 2;
    tSolver.setNumIterationsDone(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tSolver.getNumIterationsDone());

    // ********* TEST SOLVER TOLERANCE FUNCTIONS *********
    double tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tSolver.getSolverTolerance());
    tScalarGold = 0.2;
    tSolver.setSolverTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getSolverTolerance());

    // ********* TEST SET TRUST REGION RADIUS FUNCTIONS *********
    tScalarGold = 0;
    EXPECT_EQ(tScalarGold, tSolver.getTrustRegionRadius());
    tScalarGold = 2;
    tSolver.setTrustRegionRadius(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getTrustRegionRadius());

    // ********* TEST RESIDUAL NORM FUNCTIONS *********
    tScalarGold = 0;
    EXPECT_EQ(tScalarGold, tSolver.getNormResidual());
    tScalarGold = 1e-2;
    tSolver.setNormResidual(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getNormResidual());

    // ********* TEST RELATIVE TOLERANCE FUNCTIONS *********
    tScalarGold = 1e-1;
    EXPECT_EQ(tScalarGold, tSolver.getRelativeTolerance());
    tScalarGold = 1e-3;
    tSolver.setRelativeTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getRelativeTolerance());

    // ********* TEST RELATIVE TOLERANCE EXPONENTIAL FUNCTIONS *********
    tScalarGold = 0.5;
    EXPECT_EQ(tScalarGold, tSolver.getRelativeToleranceExponential());
    tScalarGold = 0.75;
    tSolver.setRelativeToleranceExponential(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getRelativeToleranceExponential());

    // ********* TEST RELATIVE TOLERANCE EXPONENTIAL FUNCTIONS *********
    locus::krylov_solver::stop_t tStopGold = locus::krylov_solver::stop_t::MAX_ITERATIONS;
    EXPECT_EQ(tStopGold, tSolver.getStoppingCriterion());
    tStopGold = locus::krylov_solver::stop_t::TRUST_REGION_RADIUS;
    tSolver.setStoppingCriterion(tStopGold);
    EXPECT_EQ(tStopGold, tSolver.getStoppingCriterion());

    // ********* TEST INVALID CURVATURE FUNCTION *********
    double tScalarValue = -1;
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::NEGATIVE_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = 0;
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::ZERO_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::INF_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::NaN_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = 1;
    EXPECT_FALSE(tSolver.invalidCurvatureDetected(tScalarValue));

    // ********* TEST TOLERANCE SATISFIED FUNCTION *********
    tScalarValue = 5e-9;
    EXPECT_TRUE(tSolver.toleranceSatisfied(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::TOLERANCE, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(tSolver.toleranceSatisfied(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::NaN_NORM_RESIDUAL, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(tSolver.toleranceSatisfied(tScalarValue));
    EXPECT_EQ(locus::krylov_solver::stop_t::INF_NORM_RESIDUAL, tSolver.getStoppingCriterion());
    tScalarValue = 1;
    EXPECT_FALSE(tSolver.toleranceSatisfied(tScalarValue));

    // ********* TEST COMPUTE STEIHAUG TOINT STEP FUNCTION *********
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tNewtonStep(tNumVectors, tNumControls);
    tNewtonStep(0,0) = 0.345854922279793;
    tNewtonStep(0,1) = 1.498704663212435;
    locus::StandardMultiVector<double> tConjugateDirection(tNumVectors, tNumControls);
    tConjugateDirection(0,0) = 1.5;
    tConjugateDirection(0,1) = 6.5;
    locus::StandardMultiVector<double> tPrecTimesNewtonStep(tNumVectors, tNumControls);
    tPrecTimesNewtonStep(0,0) = 0.345854922279793;
    tPrecTimesNewtonStep(0,1) = 1.498704663212435;
    locus::StandardMultiVector<double> tPrecTimesConjugateDirection(tNumVectors, tNumControls);
    tPrecTimesConjugateDirection(0,0) = 1.5;
    tPrecTimesConjugateDirection(0,1) = 6.5;

    tScalarValue = 0.833854004007896;
    tSolver.setTrustRegionRadius(tScalarValue);
    tScalarValue = tSolver.computeSteihaugTointStep(tNewtonStep, tConjugateDirection, tPrecTimesNewtonStep, tPrecTimesConjugateDirection);

    double tTolerance = 1e-6;
    tScalarGold = -0.105569948186529;
    EXPECT_NEAR(tScalarGold, tScalarValue, tTolerance);
}

TEST(LocusTest, ProjectedSteihaugTointPcg)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tList;
    tList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    locus::AugmentedLagrangianStageMng<double> tStageMng(tDataFactory, tCircle, tList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng.setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng.setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng.setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng.setConstraintHessians(tHessianList);

    // ********* Allocate Trust Region Algorithm Data Manager *********
    locus::TrustRegionAlgorithmDataMng<double> tDataMng(tDataFactory);

    // ********* SET INITIAL DATA MANAGER VALUES *********
    double tScalarValue = 0.5;
    tDataMng.setInitialGuess(tScalarValue);
    tScalarValue = tStageMng.evaluateObjective(tDataMng.getCurrentControl());
    tStageMng.updateCurrentConstraintValues();
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    const double tTolerance = 1e-6;
    double tScalarGoldValue = 4.875;
    EXPECT_NEAR(tScalarGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls);
    tStageMng.computeGradient(tDataMng.getCurrentControl(), tVector);
    tDataMng.setCurrentGradient(tVector);
    tDataMng.computeNormProjectedGradient();
    tScalarGoldValue = 6.670832032063167;
    EXPECT_NEAR(tScalarGoldValue, tDataMng.getNormProjectedGradient(), tTolerance);
    tVector(0, 0) = -1.5;
    tVector(0, 1) = -6.5;
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tVector);

    // ********* ALLOCATE SOLVER DATA STRUCTURE *********
    locus::ProjectedSteihaugTointPcg<double> tSolver(tDataFactory);

    // ********* CONVERGENCE: SOLVER TOLERANCE MET *********
    tScalarValue = tDataMng.getNormProjectedGradient();
    tSolver.setTrustRegionRadius(tScalarValue);
    tSolver.solve(tStageMng, tDataMng);
    size_t tIntegerGoldValue = 2;
    EXPECT_EQ(tIntegerGoldValue, tSolver.getNumIterationsDone());
    EXPECT_EQ(locus::krylov_solver::stop_t::TOLERANCE, tSolver.getStoppingCriterion());
    EXPECT_TRUE(tSolver.getNormResidual() < tSolver.getSolverTolerance());
    tVector(0, 0) = -0.071428571428571;
    tVector(0, 1) = 1.642857142857143;
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tVector);

    // ********* CONVERGENCE: MAX NUMBER OF ITERATIONS *********
    tSolver.setMaxNumIterations(2);
    tSolver.setSolverTolerance(1e-15);
    tSolver.solve(tStageMng, tDataMng);
    EXPECT_EQ(tIntegerGoldValue, tSolver.getNumIterationsDone());
    EXPECT_EQ(locus::krylov_solver::stop_t::MAX_ITERATIONS, tSolver.getStoppingCriterion());
    EXPECT_FALSE(tSolver.getNormResidual() < tSolver.getSolverTolerance());
    tVector(0, 0) = -0.071428571428571;
    tVector(0, 1) = 1.642857142857143;
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tVector);

    // ********* CONVERGENCE: TRUST REGION RADIUS VIOLATED *********
    tSolver.setTrustRegionRadius(0.833854004007896);
    tSolver.solve(tStageMng, tDataMng);
    tIntegerGoldValue = 1;
    EXPECT_EQ(tIntegerGoldValue, tSolver.getNumIterationsDone());
    EXPECT_EQ(locus::krylov_solver::stop_t::TRUST_REGION_RADIUS, tSolver.getStoppingCriterion());
    EXPECT_FALSE(tSolver.getNormResidual() < tSolver.getSolverTolerance());
    tVector(0, 0) = 0.1875;
    tVector(0, 1) = 0.8125;
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tVector);
}

TEST(LocusTest, TrustRegionStepMng)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    locus::KelleySachsStepMng<double> tStepMng(tDataFactory);

    // ********* TEST ACTUAL REDUCTION FUNCTIONS *********
    double tTolerance = 1e-6;
    double tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualReduction(), tTolerance);
    tScalarGoldValue = 0.45;
    tStepMng.setActualReduction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualReduction(), tTolerance);

    // ********* TEST TRUST REGION RADIUS FUNCTIONS *********
    tScalarGoldValue = 1e4;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionRadius(), tTolerance);
    tScalarGoldValue = 1e2;
    tStepMng.setTrustRegionRadius(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionRadius(), tTolerance);

    // ********* TEST TRUST REGION CONTRACTION FUNCTIONS *********
    tScalarGoldValue = 0.5;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionContraction(), tTolerance);
    tScalarGoldValue = 0.25;
    tStepMng.setTrustRegionContraction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionContraction(), tTolerance);

    // ********* TEST TRUST REGION EXPANSION FUNCTIONS *********
    tScalarGoldValue = 2;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionExpansion(), tTolerance);
    tScalarGoldValue = 8;
    tStepMng.setTrustRegionExpansion(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionExpansion(), tTolerance);

    // ********* TEST MIN TRUST REGION RADIUS FUNCTIONS *********
    tScalarGoldValue = 1e-4;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinTrustRegionRadius(), tTolerance);
    tScalarGoldValue = 1e-2;
    tStepMng.setMinTrustRegionRadius(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinTrustRegionRadius(), tTolerance);

    // ********* TEST MAX TRUST REGION RADIUS FUNCTIONS *********
    tScalarGoldValue = 1e4;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMaxTrustRegionRadius(), tTolerance);
    tScalarGoldValue = 1e1;
    tStepMng.setMaxTrustRegionRadius(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMaxTrustRegionRadius(), tTolerance);

    // ********* TEST GRADIENT INEXACTNESS FUNCTIONS *********
    tScalarGoldValue = 1;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessToleranceConstant(), tTolerance);
    tScalarGoldValue = 2;
    tStepMng.setGradientInexactnessToleranceConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessToleranceConstant(), tTolerance);
    // TEST INEXACTNESS TOLERANCE: SELECT CURRENT TRUST REGION RADIUS
    tScalarGoldValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessTolerance(), tTolerance);
    tScalarGoldValue = 1e3;
    tStepMng.updateGradientInexactnessTolerance(tScalarGoldValue);
    tScalarGoldValue = 200;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessTolerance(), tTolerance);
    // TEST INEXACTNESS TOLERANCE: SELECT USER INPUT
    tScalarGoldValue = 1e1;
    tStepMng.updateGradientInexactnessTolerance(tScalarGoldValue);
    tScalarGoldValue = 20;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessTolerance(), tTolerance);

    // ********* TEST OBJECTIVE INEXACTNESS FUNCTIONS *********
    tScalarGoldValue = 1;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessToleranceConstant(), tTolerance);
    tScalarGoldValue = 3;
    tStepMng.setObjectiveInexactnessToleranceConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessToleranceConstant(), tTolerance);
    // TEST INEXACTNESS TOLERANCE
    tScalarGoldValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessTolerance(), tTolerance);
    tScalarGoldValue = 100;
    tStepMng.updateObjectiveInexactnessTolerance(tScalarGoldValue);
    tScalarGoldValue = 30;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessTolerance(), tTolerance);

    // ********* TEST ACTUAL OVER PREDICTED REDUCTION BOUND FUNCTIONS *********
    tScalarGoldValue = 0.25;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionMidBound(), tTolerance);
    tScalarGoldValue = 0.4;
    tStepMng.setActualOverPredictedReductionMidBound(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionMidBound(), tTolerance);

    tScalarGoldValue = 0.1;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionLowerBound(), tTolerance);
    tScalarGoldValue = 0.05;
    tStepMng.setActualOverPredictedReductionLowerBound(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionLowerBound(), tTolerance);

    tScalarGoldValue = 0.75;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionUpperBound(), tTolerance);
    tScalarGoldValue = 0.8;
    tStepMng.setActualOverPredictedReductionUpperBound(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionUpperBound(), tTolerance);

    // ********* TEST PREDICTED REDUCTION FUNCTIONS *********
    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getPredictedReduction(), tTolerance);
    tScalarGoldValue = 0.12;
    tStepMng.setPredictedReduction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getPredictedReduction(), tTolerance);

    // ********* TEST MIN COSINE ANGLE TOLERANCE FUNCTIONS *********
    tScalarGoldValue = 1e-2;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinCosineAngleTolerance(), tTolerance);
    tScalarGoldValue = 0.1;
    tStepMng.setMinCosineAngleTolerance(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinCosineAngleTolerance(), tTolerance);

    // ********* TEST ACTUAL OVER PREDICTED REDUCTION FUNCTIONS *********
    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReduction(), tTolerance);
    tScalarGoldValue = 0.23;
    tStepMng.setActualOverPredictedReduction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReduction(), tTolerance);

    // ********* TEST NUMBER OF TRUST REGION SUBPROBLEM ITERATIONS FUNCTIONS *********
    size_t tIntegerGoldValue = 0;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getNumTrustRegionSubProblemItrDone());
    tStepMng.updateNumTrustRegionSubProblemItrDone();
    tIntegerGoldValue = 1;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getNumTrustRegionSubProblemItrDone());

    tIntegerGoldValue = 30;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getMaxNumTrustRegionSubProblemItr());
    tIntegerGoldValue = 50;
    tStepMng.setMaxNumTrustRegionSubProblemItr(tIntegerGoldValue);
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getMaxNumTrustRegionSubProblemItr());

    EXPECT_TRUE(tStepMng.isInitialTrustRegionRadiusSetToNormProjectedGradient());
    tStepMng.setInitialTrustRegionRadiusSetToNormProjectedGradient(false);
    EXPECT_FALSE(tStepMng.isInitialTrustRegionRadiusSetToNormProjectedGradient());
}

TEST(LocusTest, KelleySachsStepMng)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tList;
    tList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    locus::AugmentedLagrangianStageMng<double> tStageMng(tDataFactory, tCircle, tList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng.setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng.setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng.setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng.setConstraintHessians(tHessianList);

    // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
    locus::TrustRegionAlgorithmDataMng<double> tDataMng(tDataFactory);

    // ********* SET INITIAL DATA MANAGER VALUES *********
    double tScalarValue = 0.5;
    tDataMng.setInitialGuess(tScalarValue);
    tScalarValue = tStageMng.evaluateObjective(tDataMng.getCurrentControl());
    tStageMng.updateCurrentConstraintValues();
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    const double tTolerance = 1e-6;
    double tScalarGoldValue = 4.875;
    EXPECT_NEAR(tScalarGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls);
    tStageMng.computeGradient(tDataMng.getCurrentControl(), tVector);
    tDataMng.setCurrentGradient(tVector);
    tDataMng.computeNormProjectedGradient();
    double tNormProjectedGradientGold = 6.670832032063167;
    EXPECT_NEAR(tNormProjectedGradientGold, tDataMng.getNormProjectedGradient(), tTolerance);
    tDataMng.computeStationarityMeasure();
    double tStationarityMeasureGold = 6.670832032063167;
    EXPECT_NEAR(tStationarityMeasureGold, tDataMng.getStationarityMeasure(), tTolerance);
    tVector(0, 0) = -1.5;
    tVector(0, 1) = -6.5;
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tVector);

    // ********* ALLOCATE SOLVER *********
    locus::ProjectedSteihaugTointPcg<double> tSolver(tDataFactory);

    // ********* ALLOCATE STEP MANAGER *********
    locus::KelleySachsStepMng<double> tStepMng(tDataFactory);
    tStepMng.setTrustRegionRadius(tNormProjectedGradientGold);

    // ********* TEST CONSTANT FUNCTIONS *********
    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEtaConstant(), tTolerance);
    tScalarGoldValue = 0.12;
    tStepMng.setEtaConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEtaConstant(), tTolerance);

    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEpsilonConstant(), tTolerance);
    tScalarGoldValue = 0.11;
    tStepMng.setEpsilonConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEpsilonConstant(), tTolerance);

    // ********* TEST SUBPROBLEM SOLVE *********
    tScalarValue = 0.01;
    tStepMng.setEtaConstant(tScalarValue);
    EXPECT_TRUE(tStepMng.solveSubProblem(tDataMng, tStageMng, tSolver));

    // VERIFY CURRENT SUBPROBLEM SOLVE RESULTS
    size_t tIntegerGoldValue = 4;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getNumTrustRegionSubProblemItrDone());
    tScalarGoldValue = 0.768899024566474;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReduction(), tTolerance);
    tScalarGoldValue = 1.757354736328125;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMidPointObjectiveFunctionValue(), tTolerance);
    tScalarGoldValue = 3.335416016031584;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionRadius(), tTolerance);
    tScalarGoldValue = -3.117645263671875;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualReduction(), tTolerance);
    tScalarGoldValue = -4.0546875;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getPredictedReduction(), tTolerance);
    tScalarGoldValue = 0.066708320320632;
    EXPECT_NEAR(tScalarGoldValue, tSolver.getSolverTolerance(), tTolerance);
    const locus::MultiVector<double> & tMidControl = tStepMng.getMidPointControls();
    locus::StandardMultiVector<double> tVectorGold(tNumVectors, tNumControls);
    tVectorGold(0, 0) = 0.6875;
    tVectorGold(0, 1) = 1.3125;
    LocusTest::checkMultiVectorData(tMidControl, tVectorGold);
    const locus::MultiVector<double> & tTrialStep = tDataMng.getTrialStep();
    tVectorGold(0, 0) = 0.1875;
    tVectorGold(0, 1) = 0.8125;
    LocusTest::checkMultiVectorData(tTrialStep, tVectorGold);
}

TEST(LocusTest, KelleySachsAlgorithm)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory->allocateDual(tNumDuals);
    tDataFactory->allocateControl(tNumControls);

    // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<double>> tDataMng =
            std::make_shared<locus::TrustRegionAlgorithmDataMng<double>>(*tDataFactory);
    double tScalarValue = 0.5;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -100;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 100;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tConstraintList;
    tConstraintList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    std::shared_ptr<locus::AugmentedLagrangianStageMng<double>> tStageMng =
            std::make_shared<locus::AugmentedLagrangianStageMng<double>>(*tDataFactory, tCircle, tConstraintList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng->setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng->setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng->setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng->setConstraintHessians(tHessianList);

    // ********* ALLOCATE KELLEY-SACHS ALGORITHM *********
    locus::KelleySachsAugmentedLagrangian<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);

    // TEST MAXIMUM NUMBER OF UPDATES
    size_t tIntegerGold = 10;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumUpdates());
    tIntegerGold = 5;
    tAlgorithm.setMaxNumUpdates(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumUpdates());

    // TEST NUMBER ITERATIONS DONE
    tIntegerGold = 0;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getNumIterationsDone());
    tIntegerGold = 3;
    tAlgorithm.setNumIterationsDone(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tAlgorithm.getNumIterationsDone());

    // TEST NUMBER ITERATIONS DONE
    tIntegerGold = 100;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumIterations());
    tIntegerGold = 30;
    tAlgorithm.setMaxNumIterations(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumIterations());

    // TEST STOPPING CRITERIA
    locus::algorithm::stop_t tGold = locus::algorithm::stop_t::NOT_CONVERGED;
    EXPECT_EQ(tGold, tAlgorithm.getStoppingCriterion());
    tGold = locus::algorithm::stop_t::NaN_NORM_TRIAL_STEP;
    tAlgorithm.setStoppingCriterion(tGold);
    EXPECT_EQ(tGold, tAlgorithm.getStoppingCriterion());

    // TEST GRADIENT TOLERANCE
    double tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getGradientTolerance());
    tScalarGold = 1e-4;
    tAlgorithm.setGradientTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getGradientTolerance());

    // TEST TRIAL STEP TOLERANCE
    tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getTrialStepTolerance());
    tScalarGold = 1e-3;
    tAlgorithm.setTrialStepTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getTrialStepTolerance());

    // TEST OBJECTIVE TOLERANCE
    tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getObjectiveTolerance());
    tScalarGold = 1e-5;
    tAlgorithm.setObjectiveTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getObjectiveTolerance());

    // TEST CONTROL STAGNATION TOLERANCE
    tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getStagnationTolerance());
    tScalarGold = 1e-2;
    tAlgorithm.setStagnationTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getStagnationTolerance());

    // TEST ACTUAL REDUCTION TOLERANCE
    tScalarGold = 1e-10;
    EXPECT_EQ(tScalarGold, tAlgorithm.getActualReductionTolerance());
    tScalarGold = 1e-9;
    tAlgorithm.setActualReductionTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getActualReductionTolerance());

    // TEST UPDATE CONTROL
    const size_t tNumVectors = 1;
    locus::KelleySachsStepMng<double> tStepMng(*tDataFactory);
    locus::StandardMultiVector<double> tMidControls(tNumVectors, tNumControls);
    tMidControls(0,0) = 0.6875;
    tMidControls(0,1) = 1.3125;
    tStepMng.setMidPointControls(tMidControls);
    locus::StandardMultiVector<double> tMidGradient(tNumVectors, tNumControls);
    tMidGradient(0,0) = 1.0185546875;
    tMidGradient(0,1) = 0.3876953125;
    tScalarValue = 1.757354736328125;
    tStepMng.setMidPointObjectiveFunctionValue(tScalarValue);
    tScalarValue = -3.117645263671875;
    tStepMng.setActualReduction(tScalarValue);
    EXPECT_TRUE(tAlgorithm.updateControl(tMidGradient, tStepMng, *tDataMng, *tStageMng));

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(0,0) = -0.3310546875;
    tGoldVector(0,1) = 0.9248046875;
    const locus::MultiVector<double> & tCurrentControl = tDataMng->getCurrentControl();
    LocusTest::checkMultiVectorData(tCurrentControl, tGoldVector);
    const double tTolerance = 1e-6;
    tScalarGold = 2.327059142438884;
    EXPECT_NEAR(tScalarGold, tStepMng.getActualReduction(), tTolerance);
    tScalarGold = 4.084413878767009;
    EXPECT_NEAR(tScalarGold, tDataMng->getCurrentObjectiveFunctionValue(), tTolerance);
}

TEST(LocusTest, KelleySachsAugmentedLagrangian)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory->allocateDual(tNumDuals);
    tDataFactory->allocateControl(tNumControls);

    // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<double>> tDataMng =
            std::make_shared<locus::TrustRegionAlgorithmDataMng<double>>(*tDataFactory);
    double tScalarValue = 0.5;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -100;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 100;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tConstraintList;
    tConstraintList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    std::shared_ptr<locus::AugmentedLagrangianStageMng<double>> tStageMng =
            std::make_shared<locus::AugmentedLagrangianStageMng<double>>(*tDataFactory, tCircle, tConstraintList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng->setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng->setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng->setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng->setConstraintHessians(tHessianList);

    // ********* ALLOCATE KELLEY-SACHS ALGORITHM *********
    locus::KelleySachsAugmentedLagrangian<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    // TEST NUMBER OF ITERATIONS AND STOPPING CRITERION
    size_t tIntegerGold = 25;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getNumIterationsDone());
    locus::algorithm::stop_t tGold = locus::algorithm::stop_t::CONTROL_STAGNATION;
    EXPECT_EQ(tGold, tAlgorithm.getStoppingCriterion());

    // TEST OBJECTIVE FUNCTION VALUE
    const double tTolerance = 1e-6;
    double tScalarGold = 2.678009477208421;
    EXPECT_NEAR(tScalarGold, tDataMng->getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarGold = 2.678009477208421;

    // TEST CURRENT CONSTRAINT VALUE
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tConstraintValues(tNumVectors, tNumDuals);
    tStageMng->getCurrentConstraintValues(tConstraintValues);
    locus::StandardMultiVector<double> tGoldConstraintValues(tNumVectors, tNumDuals);
    tGoldConstraintValues(0,0) = 1.876192258460918e-4;
    LocusTest::checkMultiVectorData(tConstraintValues, tGoldConstraintValues);

    // TEST LAGRANGE MULTIPLIERS
    locus::StandardMultiVector<double> tLagrangeMulipliers(tNumVectors, tNumDuals);
    tStageMng->getLagrangeMultipliers(tLagrangeMulipliers);
    locus::StandardMultiVector<double> tGoldtLagrangeMulipliers(tNumVectors, tNumDuals);
    tGoldtLagrangeMulipliers(0,0) = 2.209155776190176;
    LocusTest::checkMultiVectorData(tLagrangeMulipliers, tGoldtLagrangeMulipliers);

    // TEST CONTROL SOLUTION
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(0,0) = 0.311608429003505;
    tGoldVector(0,1) = 0.950309321326385;
    const locus::MultiVector<double> & tCurrentControl = tDataMng->getCurrentControl();
    LocusTest::checkMultiVectorData(tCurrentControl, tGoldVector);

    // TEST CURRENT AUGMENTED LAGRANGIAN GRADIENT
    tGoldVector(0,0) = 0.073079644963231;
    tGoldVector(0,1) = 0.222870312033612;
    const locus::MultiVector<double> & tCurrentGradient = tDataMng->getCurrentGradient();
    LocusTest::checkMultiVectorData(tCurrentGradient, tGoldVector);
}

} // namespace LocusTest
