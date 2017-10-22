/*
 * Locus_LineSearchTest.cpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "Locus_UnitTestUtils.hpp"

#include "Locus_Rosenbrock.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_QuadraticLineSearch.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStateMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMng.hpp"

namespace LocusTest
{

TEST(LocusTest, QuadraticLineSearch)
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

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    double tScalarValue = std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlUpperBounds(tControl);
    tScalarValue = -std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlLowerBounds(tControl);

    const size_t tVectorIndex = 0;
    tControl(tVectorIndex, 0) = 1.997506234413967;
    tControl(tVectorIndex, 1) = 3.990024937655861;
    tStateMng.setCurrentControl(tControl);
    tScalarValue = tStateMng.evaluateObjective(tControl);
    const double tTolerance = 1e-6;
    double tGoldScalarValue = 0.99501869156216238;
    EXPECT_NEAR(tScalarValue, tGoldScalarValue, tTolerance);
    tStateMng.setCurrentObjectiveValue(tScalarValue);

    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tStateMng.computeGradient(tControl, tGradient);
    tStateMng.setCurrentGradient(tGradient);

    locus::StandardMultiVector<double> tTrialStep(tNumVectors, tNumControls);
    tTrialStep(tVectorIndex, 0) = -1.997506234413967;
    tTrialStep(tVectorIndex, 1) = -3.990024937655861;
    tStateMng.setTrialStep(tTrialStep);

    // ********* Allocate Quadratic Line Search *********
    locus::QuadraticLineSearch<double> tLineSearch(tDataFactory);
    tLineSearch.step(tStateMng);

    size_t tOrdinalValue = 7;
    EXPECT_EQ(tOrdinalValue, tLineSearch.getNumIterationsDone());
    tGoldScalarValue = 0.00243606117022465;
    EXPECT_NEAR(tLineSearch.getStepValue(), tGoldScalarValue, tTolerance);
    tGoldScalarValue = 0.99472430176791571;
    EXPECT_NEAR(tStateMng.getCurrentObjectiveValue(), tGoldScalarValue, tTolerance);
    tControl(tVectorIndex, 0) = 1.9926401870390293;
    tControl(tVectorIndex, 1) = 3.9803049928370093;
    LocusTest::checkMultiVectorData(tStateMng.getCurrentControl(), tControl);
}

} // namespace LocusTest
