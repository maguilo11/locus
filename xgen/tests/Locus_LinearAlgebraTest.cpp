/*
 * Locus_LinearAlgebraTest.cpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <limits>

#include "Locus_StandardVector.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_DistributedReductionOperations.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"

namespace LocusTest
{

TEST(LocusTest, size)
{
    const double tBaseValue = 1;
    const size_t tNumElements = 10;
    std::vector<double> tTemplateVector(tNumElements, tBaseValue);

    locus::StandardVector<double> tlocusVector(tTemplateVector);

    const size_t tGold = 10;
    EXPECT_EQ(tlocusVector.size(), tGold);
}

TEST(LocusTest, scale)
{
    const double tBaseValue = 1;
    const int tNumElements = 10;
    locus::StandardVector<double, int> tlocusVector(tNumElements, tBaseValue);

    double tScaleValue = 2;
    tlocusVector.scale(tScaleValue);

    double tGold = 2;
    double tTolerance = 1e-6;
    for(int tIndex = 0; tIndex < tlocusVector.size(); tIndex++)
    {
        EXPECT_NEAR(tlocusVector[tIndex], tGold, tTolerance);
    }
}

TEST(LocusTest, entryWiseProduct)
{
    std::vector<double> tTemplateVector =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double, size_t> tlocusVector1(tTemplateVector);
    locus::StandardVector<double, size_t> tlocusVector2(tTemplateVector);

    tlocusVector1.entryWiseProduct(tlocusVector2);

    double tTolerance = 1e-6;
    std::vector<double> tGold =
        { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };
    for(size_t tIndex = 0; tIndex < tlocusVector1.size(); tIndex++)
    {
        EXPECT_NEAR(tlocusVector1[tIndex], tGold[tIndex], tTolerance);
    }
}

TEST(LocusTest, modulus)
{
    std::vector<double> tTemplateVector =
        { -1, 2, -3, 4, 5, -6, -7, 8, -9, -10 };
    locus::StandardVector<double> tlocusVector(tTemplateVector);

    tlocusVector.modulus();

    double tTolerance = 1e-6;
    std::vector<double> tGold =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    for(size_t tIndex = 0; tIndex < tlocusVector.size(); tIndex++)
    {
        EXPECT_NEAR(tlocusVector[tIndex], tGold[tIndex], tTolerance);
    }
}

TEST(LocusTest, dot)
{
    std::vector<double> tTemplateVector1 =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tlocusVector1(tTemplateVector1);
    std::vector<double> tTemplateVector2 =
        { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    locus::StandardVector<double> tlocusVector2(tTemplateVector2);

    double tDot = tlocusVector1.dot(tlocusVector2);

    double tGold = 110;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tDot, tGold, tTolerance);
}

TEST(LocusTest, fill)
{
    std::vector<double> tTemplateVector =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tlocusVector(tTemplateVector);

    double tFillValue = 3;
    tlocusVector.fill(tFillValue);

    double tGold = 3.;
    double tTolerance = 1e-6;
    for(size_t tIndex = 0; tIndex < tlocusVector.size(); tIndex++)
    {
        EXPECT_NEAR(tlocusVector[tIndex], tGold, tTolerance);
    }
}

TEST(LocusTest, create)
{
    std::vector<double> tTemplateVector =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tlocusVector(tTemplateVector);

// TEST ONE: CREATE COPY OF BASE CONTAINER WITH THE SAME NUMBER OF ELEMENTS AS THE BASE VECTOR AND FILL IT WITH ZEROS
    std::shared_ptr<locus::Vector<double>> tCopy1 = tlocusVector.create();

    size_t tGoldSize1 = 10;
    EXPECT_EQ(tCopy1->size(), tGoldSize1);
    EXPECT_TRUE(tCopy1->size() == tlocusVector.size());

    double tGoldDot1 = 0;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tCopy1->dot(tlocusVector), tGoldDot1, tTolerance);
}

TEST(LocusTest, MultiVector)
{
    size_t tNumVectors = 8;
    std::vector<double> tVectorGold =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tlocusVector(tVectorGold);
    // Default for second template typename is OrdinalType = size_t
    locus::StandardMultiVector<double> tMultiVector1(tNumVectors, tlocusVector);

    size_t tGoldNumVectors = 8;
    EXPECT_EQ(tMultiVector1.getNumVectors(), tGoldNumVectors);

    double tGoldSum = 0;
    size_t tGoldSize = 10;

    double tTolerance = 1e-6;
    // Default for second template typename is OrdinalType = size_t
    locus::StandardVectorReductionOperations<double> tInterface;
    for(size_t tIndex = 0; tIndex < tMultiVector1.getNumVectors(); tIndex++)
    {
        EXPECT_EQ(tMultiVector1[tIndex].size(), tGoldSize);
        double tSumValue = tInterface.sum(tMultiVector1[tIndex]);
        EXPECT_NEAR(tSumValue, tGoldSum, tTolerance);
    }

    std::vector<std::shared_ptr<locus::Vector<double>>>tMultiVectorTemplate(tNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        tMultiVectorTemplate[tIndex] = tlocusVector.create();
        tMultiVectorTemplate[tIndex]->update(static_cast<double>(1.), tlocusVector, static_cast<double>(0.));
    }

    // Default for second template typename is OrdinalType = size_t
    tGoldSum = 55;
    locus::StandardMultiVector<double> tMultiVector2(tMultiVectorTemplate);
    for(size_t tVectorIndex = 0; tVectorIndex < tMultiVector2.getNumVectors(); tVectorIndex++)
    {
        EXPECT_EQ(tMultiVector2[tVectorIndex].size(), tGoldSize);
        for(size_t tElementIndex = 0; tElementIndex < tMultiVector2[tVectorIndex].size(); tElementIndex++)
        {
            EXPECT_NEAR(tMultiVector2(tVectorIndex, tElementIndex), tVectorGold[tElementIndex], tTolerance);
        }
        double tSumValue = tInterface.sum(tMultiVector2[tVectorIndex]);
        EXPECT_NEAR(tSumValue, tGoldSum, tTolerance);
    }
}

TEST(LocusTest, StandardVectorReductionOperations)
{
    std::vector<double> tTemplateVector =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double, size_t> tlocusVector(tTemplateVector);

    locus::StandardVectorReductionOperations<double, size_t> tInterface;

    // Test MAX
    double tMaxValue = tInterface.max(tlocusVector);
    double tTolerance = 1e-6;
    double tGoldMaxValue = 10;
    EXPECT_NEAR(tMaxValue, tGoldMaxValue, tTolerance);

    // Test MIN
    double tMinValue = tInterface.min(tlocusVector);
    double tGoldMinValue = 1.;
    EXPECT_NEAR(tMinValue, tGoldMinValue, tTolerance);

    // Test SUM
    double tSum = tInterface.sum(tlocusVector);
    double tGold = 55;
    EXPECT_NEAR(tSum, tGold, tTolerance);
}

TEST(LocusTest, DistributedReductionOperations)
{
    std::vector<double> tTemplateVector =
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double, size_t> tlocusVector(tTemplateVector);

    locus::DistributedReductionOperations<double, size_t> tReductionOperations;

    int tGold = std::numeric_limits<int>::max();
    MPI_Comm_size(MPI_COMM_WORLD, &tGold);
    size_t tNumRanks = tReductionOperations.getNumRanks();

    EXPECT_EQ(static_cast<size_t>(tGold), tNumRanks);

    double tTolerance = 1e-6;
    double tSum = tReductionOperations.sum(tlocusVector);
    double tGoldSum = static_cast<double>(tNumRanks) * 55.;
    EXPECT_NEAR(tSum, tGoldSum, tTolerance);

    double tGoldMax = 10;
    double tMax = tReductionOperations.max(tlocusVector);
    EXPECT_NEAR(tMax, tGoldMax, tTolerance);

    double tGoldMin = 1;
    double tMin = tReductionOperations.min(tlocusVector);
    EXPECT_NEAR(tMin, tGoldMin, tTolerance);

    // NOTE: Default OrdinalType = size_t
    std::shared_ptr<locus::ReductionOperations<double>> tReductionOperationsCopy = tReductionOperations.create();
    double tSumCopy = tReductionOperationsCopy->sum(tlocusVector);
    EXPECT_NEAR(tSumCopy, tGoldSum, tTolerance);
}

} // namespace LocusTest
