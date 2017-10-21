/*
 * Locus_BoundsTest.cpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "Locus_UnitTestUtils.hpp"

#include "Locus_Bounds.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_StandardMultiVector.hpp"

namespace LocusTest
{

TEST(LocusTest, Project)
{
    // ********* Allocate Input Data *********
    const size_t tNumVectors = 8;
    std::vector<double> tVectorGold = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tLocusVector(tVectorGold);
    // Default for second template typename is OrdinalType = size_t
    locus::StandardMultiVector<double> tData(tNumVectors, tLocusVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tData[tVectorIndex].update(1., tLocusVector, 0.);
    }

    // ********* Allocate Lower & Upper Bounds *********
    const double tLowerBoundValue = 2;
    const size_t tNumElementsPerVector = tVectorGold.size();
    locus::StandardMultiVector<double> tLowerBounds(tNumVectors, tNumElementsPerVector, tLowerBoundValue);
    const double tUpperBoundValue = 7;
    locus::StandardMultiVector<double> tUpperBounds(tNumVectors, tNumElementsPerVector, tUpperBoundValue);

    locus::bounds::project(tLowerBounds, tUpperBounds, tData);

    std::vector<double> tVectorBoundsGold = { 2, 2, 3, 4, 5, 6, 7, 7, 7, 7 };
    locus::StandardVector<double> tlocusBoundVector(tVectorBoundsGold);
    locus::StandardMultiVector<double> tGoldData(tNumVectors, tlocusBoundVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tGoldData[tVectorIndex].update(1., tlocusBoundVector, 0.);
    }
    LocusTest::checkMultiVectorData(tData, tGoldData);
}

TEST(LocusTest, CheckBounds)
{
    // ********* Allocate Lower & Upper Bounds *********
    const size_t tNumVectors = 1;
    const size_t tNumElements = 5;
    double tLowerBoundValue = 2;
    locus::StandardMultiVector<double> tLowerBounds(tNumVectors, tNumElements, tLowerBoundValue);
    double tUpperBoundValue = 7;
    locus::StandardMultiVector<double> tUpperBounds(tNumVectors, tNumElements, tUpperBoundValue);
    ASSERT_NO_THROW(locus::bounds::checkBounds(tLowerBounds, tUpperBounds));

    tUpperBoundValue = 2;
    locus::fill(tUpperBoundValue, tUpperBounds);
    ASSERT_THROW(locus::bounds::checkBounds(tLowerBounds, tUpperBounds), std::invalid_argument);
}

TEST(LocusTest, ComputeActiveAndInactiveSet)
{
    // ********* Allocate Input Data *********
    const size_t tNumVectors = 4;
    std::vector<double> tVectorGold = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tLocusVector(tVectorGold);
    // Default for second template typename is OrdinalType = size_t
    locus::StandardMultiVector<double> tData(tNumVectors, tLocusVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tData[tVectorIndex].update(1., tLocusVector, 0.);
    }

    // ********* Allocate Lower & Upper Bounds *********
    const double tLowerBoundValue = 2;
    const size_t tNumElementsPerVector = tVectorGold.size();
    locus::StandardMultiVector<double> tLowerBounds(tNumVectors, tNumElementsPerVector, tLowerBoundValue);
    const double tUpperBoundValue = 7;
    locus::StandardMultiVector<double> tUpperBounds(tNumVectors, tNumElementsPerVector, tUpperBoundValue);

    // ********* Allocate Active & Inactive Sets *********
    locus::StandardMultiVector<double> tActiveSet(tNumVectors, tNumElementsPerVector, tUpperBoundValue);
    locus::StandardMultiVector<double> tInactiveSet(tNumVectors, tNumElementsPerVector, tUpperBoundValue);

    // ********* Compute Active & Inactive Sets *********
    locus::bounds::project(tLowerBounds, tUpperBounds, tData);
    locus::bounds::computeActiveAndInactiveSets(tData, tLowerBounds, tUpperBounds, tActiveSet, tInactiveSet);

    std::vector<double> tActiveSetGold = { 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
    std::vector<double> tInactiveSetGold = { 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 };
    locus::StandardVector<double> tlocusActiveSetVectorGold(tActiveSetGold);
    locus::StandardVector<double> tlocusInactiveSetVectorGold(tInactiveSetGold);
    locus::StandardMultiVector<double> tActiveSetGoldData(tNumVectors, tlocusActiveSetVectorGold);
    locus::StandardMultiVector<double> tInactiveSetGoldData(tNumVectors, tlocusInactiveSetVectorGold);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tActiveSetGoldData[tVectorIndex].update(1., tlocusActiveSetVectorGold, 0.);
        tInactiveSetGoldData[tVectorIndex].update(1., tlocusInactiveSetVectorGold, 0.);
    }
    LocusTest::checkMultiVectorData(tActiveSet, tActiveSetGoldData);
    LocusTest::checkMultiVectorData(tInactiveSet, tInactiveSetGoldData);
}

} // namespace LocusTest
