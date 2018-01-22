/*
 * Locus_ProbabilityDistributionTest.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <vector>
#include <numeric>

#include "Locus_Criterion.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_StandardMultiVector.hpp"

#include "Locus_UnitTestUtils.hpp"

#define _MATH_DEFINES_DEFINED

namespace locus
{

template<typename OrdinalType>
inline OrdinalType factorial(const OrdinalType & aInput)
{
    OrdinalType tOutput = 1;
    for(OrdinalType tIndex = 1; tIndex <= aInput; tIndex++)
    {
        tOutput *= tIndex;
    }
    return (tOutput);
}

template<typename ScalarType>
inline ScalarType pochhammer_symbol(const ScalarType & aX, const ScalarType & aN)
{
    const ScalarType tCoefficientOne = std::tgamma(aX + aN);
    const ScalarType tCoefficientTwo = std::tgamma(aX);
    ScalarType tOutput = tCoefficientOne / tCoefficientTwo;
    tOutput = std::isfinite(tOutput) ? tOutput : static_cast<ScalarType>(0);
    return (tOutput);
}

template<typename ScalarType>
inline void beta_shape_parameters(const ScalarType & aMinValue,
                                  const ScalarType & aMaxValue,
                                  const ScalarType & aMean,
                                  const ScalarType & aSigma,
                                  ScalarType & aAlpha,
                                  ScalarType & aBeta)
{
    // Scale mean/variance to lie in [0,1] for the standard beta distribution:
    ScalarType tMeanStd = (aMean - aMinValue) / (aMaxValue - aMinValue);
    ScalarType tVarianceStd = (static_cast<ScalarType>(1) / (aMaxValue - aMinValue))
            * (static_cast<ScalarType>(1) / (aMaxValue - aMinValue)) * aSigma;
    // Compute shape parameters for Beta distributions based on standard mean/variance:
    aAlpha = tMeanStd
            * (tMeanStd * (static_cast<ScalarType>(1) - tMeanStd) / tVarianceStd - static_cast<ScalarType>(1));
    aBeta = (tMeanStd * (static_cast<ScalarType>(1) - tMeanStd) / tVarianceStd - static_cast<ScalarType>(1)) - aAlpha;
}

template<typename ScalarType>
inline ScalarType beta(const ScalarType & aAlpha, const ScalarType & aBeta)
{
    ScalarType tCoefficientOne = std::lgamma(aAlpha);
    tCoefficientOne = std::isfinite(tCoefficientOne) == true ? tCoefficientOne : static_cast<ScalarType>(0);

    ScalarType tCoefficientTwo = std::lgamma(aBeta);
    tCoefficientTwo = std::isfinite(tCoefficientTwo) == true ? tCoefficientTwo : static_cast<ScalarType>(0);

    ScalarType tCoefficientThree = std::lgamma(aAlpha + aBeta);
    tCoefficientThree = std::isfinite(tCoefficientThree) == true ? tCoefficientThree : static_cast<ScalarType>(0);

    ScalarType tExponent = tCoefficientOne + tCoefficientTwo - tCoefficientThree;
    ScalarType tOutput = std::exp(tExponent);

    return (tOutput);
}

template<typename ScalarType, typename OrdinalType = size_t>
inline ScalarType incomplete_beta(const ScalarType & aValue, const ScalarType & aAlpha, const ScalarType & aBeta)
{
    const OrdinalType tNUM_TERMS = 21;
    const ScalarType tConstantTwo = static_cast<ScalarType>(1) - aBeta;

    ScalarType tSum = 0;
    for(OrdinalType tIndex = 0; tIndex <= tNUM_TERMS; tIndex++)
    {
        ScalarType tFactorial = locus::factorial<OrdinalType>(tIndex);
        ScalarType tDenominator = tFactorial * (aAlpha + tIndex);
        ScalarType tNumerator = locus::pochhammer_symbol<ScalarType>(tConstantTwo, tIndex);
        ScalarType tConstant = tNumerator / tDenominator;
        tConstant = std::isfinite(tConstant) ? tConstant : 0;
        tSum = tSum + (tConstant * std::pow(aValue, tIndex));
    }
    const ScalarType tConstant = std::pow(aValue, aAlpha);
    tSum = tConstant * tSum;

    return (tSum);
}

template<typename ScalarType>
inline ScalarType beta_pdf(const ScalarType & aValue, const ScalarType & aAlpha, const ScalarType & aBeta)
{
    const ScalarType tEPSILON = 1e-14;
    const ScalarType tAlpha = aAlpha + tEPSILON;
    const ScalarType tBeta = aBeta + tEPSILON;

    const ScalarType tNumerator = std::pow(aValue, tAlpha - static_cast<ScalarType>(1))
            * std::pow(static_cast<ScalarType>(1) - aValue, tBeta - static_cast<ScalarType>(1));
    const ScalarType tDenominator = locus::beta<ScalarType>(tAlpha, tBeta);
    const ScalarType tOutput = tNumerator / tDenominator;

    return (tOutput);
}

template<typename ScalarType, typename OrdinalType = size_t>
inline ScalarType beta_cdf(const ScalarType & aValue, const ScalarType & aAlpha, const ScalarType & aBeta)
{
    if(aValue < static_cast<ScalarType>(0))
    {
        const ScalarType tOutput = 0;
        return (tOutput);
    }
    else if(aValue > static_cast<ScalarType>(1))
    {
        const ScalarType tOutput = 1;
        return (tOutput);
    }

    const ScalarType tEPSILON = 1e-14;
    const ScalarType tAlpha = aAlpha + tEPSILON;
    const ScalarType tBeta = aBeta + tEPSILON;
    const ScalarType tNumerator = locus::incomplete_beta<ScalarType, OrdinalType>(aValue, tAlpha, tBeta);
    const ScalarType tDenominator = locus::beta<ScalarType>(tAlpha, tBeta);
    const ScalarType tOutput = tNumerator / tDenominator;

    return (tOutput);
}

template<typename ScalarType>
inline ScalarType beta_moment(const ScalarType & aOrder, const ScalarType & aAlpha, const ScalarType & aBeta)
{
    ScalarType tDenominator = locus::beta<ScalarType>(aAlpha, aBeta);
    ScalarType tMyAlpha = aAlpha + aOrder;
    ScalarType tNumerator = locus::beta<ScalarType>(tMyAlpha, aBeta);
    ScalarType tOutput = tNumerator / tDenominator;
    return (tOutput);
}

template<typename ScalarType>
inline ScalarType gaussian_pdf(const ScalarType & aValue, const ScalarType & aMean, const ScalarType & aSigma)
{
    ScalarType tConstant = static_cast<ScalarType>(1)
            / std::sqrt(static_cast<ScalarType>(2) * static_cast<ScalarType>(M_PI) * aSigma * aSigma);
    ScalarType tExponential = std::exp(static_cast<ScalarType>(-1) * (aValue - aMean) * (aValue - aMean)
            / (static_cast<ScalarType>(2) * aSigma * aSigma));
    ScalarType tOutput = tConstant * tExponential;
    return (tOutput);
}

template<typename ScalarType>
inline ScalarType gaussian_cdf(const ScalarType & aValue, const ScalarType & aMean, const ScalarType & aSigma)
{
    ScalarType tOutput = static_cast<ScalarType>(0.5) * (static_cast<ScalarType>(1)
            + std::erf((aValue - aMean) / (aSigma * std::sqrt(static_cast<ScalarType>(2)))));
    return (tOutput);
}

template<typename ScalarType, typename OrdinalType = size_t>
inline ScalarType compute_srom_cdf(const ScalarType & aSigma,
                                   const ScalarType & aX,
                                   const locus::Vector<ScalarType, OrdinalType> & aSamples,
                                   const locus::Vector<ScalarType, OrdinalType> & aSamplesProbability)
{
    ScalarType tSum = 0;
    OrdinalType tNumSamples = aSamples.size();
    for(OrdinalType tIndexJ = 0; tIndexJ < tNumSamples; tIndexJ++)
    {
        ScalarType tValue = (aX - aSamples[tIndexJ]) / (aSigma * std::sqrt(static_cast<ScalarType>(2)));
        tSum = tSum + aSamplesProbability[tIndexJ] *
                (static_cast<ScalarType>(0.5) * (static_cast<ScalarType>(1) + std::erf(tValue)));
    }
    return (tSum);
}

template<typename ScalarType, typename OrdinalType = size_t>
inline ScalarType compute_srom_moment(const ScalarType & aOrder,
                                      const locus::Vector<ScalarType, OrdinalType> & aSamples,
                                      const locus::Vector<ScalarType, OrdinalType> & aSamplesProbability)
{
    assert(aOrder >= static_cast<OrdinalType>(0));
    assert(aSamples.size() == aSamplesProbability.size());

    ScalarType tOutput = 0;
    OrdinalType tNumSamples = aSamples.size();
    for(OrdinalType tSampleIndex = 0; tSampleIndex < tNumSamples; tSampleIndex++)
    {
        tOutput = tOutput + (aSamplesProbability[tSampleIndex] * std::pow(aSamples[tSampleIndex], aOrder));
    }
    return (tOutput);
}

template<typename ScalarType, typename OrdinalType = size_t>
inline ScalarType shift_beta_moment(const OrdinalType & aOrder,
                                    const ScalarType & aShift,
                                    const ScalarType & aAlpha,
                                    const ScalarType & aBeta)
{
    assert(aOrder >= static_cast<OrdinalType>(0));
    ScalarType tOutput = 0;
    for(OrdinalType tIndex = 1; tIndex <= aOrder; tIndex++)
    {
        ScalarType tNumerator = locus::factorial<OrdinalType>(aOrder);
        ScalarType tDenominator = locus::factorial<OrdinalType>(tIndex) * locus::factorial<OrdinalType>(aOrder - tIndex);
        ScalarType tCoefficient = tNumerator / tDenominator;
        ScalarType tShiftParameter = tCoefficient * std::pow(aShift, static_cast<ScalarType>(tIndex));
        OrdinalType tMyOrder = aOrder - tIndex;
        ScalarType tMoment = locus::beta_moment<ScalarType>(tMyOrder, aAlpha, aBeta);
        tOutput = tOutput + tShiftParameter * tMoment;
    }
    return (tOutput);
}

template<typename ScalarType, typename OrdinalType = size_t>
class Distirbution
{
public:
    virtual ~Distirbution()
    {
    }

    virtual ScalarType pdf(const ScalarType & tInput) = 0;
    virtual ScalarType cdf(const ScalarType & tInput) = 0;
    virtual ScalarType moment(const OrdinalType & tInput) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class Beta : public locus::Distirbution<ScalarType, OrdinalType>
{
public:
    explicit Beta(const ScalarType & aMin,
                  const ScalarType & aMax,
                  const ScalarType & aMean,
                  const ScalarType & aVariance) :
            mMin(aMin),
            mMax(aMax),
            mMean(aMean),
            mVariance(aVariance),
            mBeta(0),
            mAlpha(0)
    {
        locus::beta_shape_parameters<ScalarType>(aMin, aMax, aMean, aVariance, mAlpha, mBeta);
    }
    virtual ~Beta()
    {
    }

    ScalarType min() const
    {
        return (mMin);
    }
    ScalarType max() const
    {
        return (mMax);
    }
    ScalarType mean() const
    {
        return (mMean);
    }
    ScalarType variance() const
    {
        return (mVariance);
    }
    ScalarType beta() const
    {
        return (mBeta);
    }
    ScalarType alpha() const
    {
        return (mAlpha);
    }

    ScalarType pdf(const ScalarType & aInput)
    {
        ScalarType tOutput = locus::beta_pdf<ScalarType>(aInput, mAlpha, mBeta);
        return (tOutput);
    }
    ScalarType cdf(const ScalarType & aInput)
    {
        ScalarType tOutput = locus::beta_cdf<ScalarType, OrdinalType>(aInput, mAlpha, mBeta);
        return (tOutput);
    }
    ScalarType moment(const OrdinalType & aInput)
    {
        ScalarType tOutput = locus::beta_moment<ScalarType>(aInput, mAlpha, mBeta);
        return (tOutput);
    }

private:
    ScalarType mMin;
    ScalarType mMax;
    ScalarType mMean;
    ScalarType mVariance;

    ScalarType mBeta;
    ScalarType mAlpha;

private:
    Beta(const locus::Beta<ScalarType, OrdinalType> & aRhs);
    locus::Beta<ScalarType, OrdinalType> & operator=(const locus::Beta<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class SromObjective : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    explicit SromObjective(const std::shared_ptr<locus::Distirbution<ScalarType, OrdinalType>> & aDistribution) :
            mMaxMomentOrder(4),
            mSromSigma(1e-3),
            mWeightCdfMisfit(1),
            mWeightMomentMisfit(1),
            mTrueMoments(),
            mDistribution(aDistribution)
    {
        mTrueMoments = std::make_shared<locus::StandardVector<ScalarType, OrdinalType>>(mMaxMomentOrder);
        for(OrdinalType tMomentIndex = 1; tMomentIndex <= mMaxMomentOrder; tMomentIndex++)
        {
            OrdinalType tMyIndex = tMomentIndex - static_cast<OrdinalType>(1);
            mTrueMoments->operator[](tMyIndex) = mDistribution->moment(tMomentIndex);
        }
    }
    virtual ~SromObjective()
    {
    }

    void setSromSigma(const ScalarType & aInput)
    {
        mSromSigma = aInput;
    }
    void setMaxMomentOrder(const OrdinalType & aInput)
    {
        assert(aInput > static_cast<OrdinalType>(0));

        mMaxMomentOrder = aInput;
        mTrueMoments = std::make_shared<locus::StandardVector<ScalarType, OrdinalType>>(aInput);
    }
    void setmWeightCdfMisfit(const ScalarType & aInput)
    {
        mWeightCdfMisfit = aInput;
    }
    void setWeightMomentMisfit(const ScalarType & aInput)
    {
        mWeightMomentMisfit = aInput;
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const ScalarType tMisfitMoments = this->computeMomentsError(aControl);
        const ScalarType tMisfitCDF = this->computeCummulativeDistributionFunctionMisfit(aControl);
        // CORRELATION MOMENT NOT IMPLEMENTED YET.
        const ScalarType tOutput = (mWeightCdfMisfit * tMisfitCDF) + (mWeightMomentMisfit * tMisfitMoments);
        return (tOutput);
    }

    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
    }

    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::SromObjective<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    ScalarType computeMomentsError(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        /* Compute error in moments up to order q for ith dimension (i.e. i-th random vector dimension).
         * Currently, the random vector dimension is set to one. Hence, random vector has size one and
         * samples are not correlated.*/

        const OrdinalType tSamplesVecIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tSamples = aControl[tSamplesVecIndex];
        const OrdinalType tProbabilitiesVecIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tProbabilities = aControl[tProbabilitiesVecIndex];
        assert(tSamples.size() == tProbabilities.size());

        ScalarType tSum = 0;
        for(OrdinalType tMomentOrder = 1; tMomentOrder <= mMaxMomentOrder; tMomentOrder++)
        {
            ScalarType tSromMoment = locus::compute_srom_moment<ScalarType, OrdinalType>(tMomentOrder, tSamples, tProbabilities);
            OrdinalType tMyIndex = tMomentOrder - static_cast<OrdinalType>(1);
            ScalarType tNumerator = tSromMoment - mTrueMoments->operator[](tMyIndex);
            ScalarType tValue = tNumerator / mTrueMoments->operator[](tMyIndex);
            tSum = tSum + (tValue * tValue);
        }
        return (tSum);
    }
    ScalarType computeCummulativeDistributionFunctionMisfit(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const OrdinalType tSamplesVecIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tSamples = aControl[tSamplesVecIndex];
        const OrdinalType tProbabilitiesVecIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tProbabilities = aControl[tProbabilitiesVecIndex];
        assert(tSamples.size() == tProbabilities.size());

        ScalarType tSum = 0;
        const OrdinalType tNumSamples = tSamples.size();
        for(OrdinalType tIndex = 0; tIndex < tNumSamples; tIndex++)
        {
            /* Get i^th dimension of j^th sample. Currently, the random vector dimension is
             * set to one. Hence, random vector has size one and samples are not correlated. */
            ScalarType tSample_ij = tSamples[tIndex];
            ScalarType tTrueCDF = mDistribution->cdf(tSample_ij);
            ScalarType tSromCDF = locus::compute_srom_cdf<ScalarType, OrdinalType>(mSromSigma, tSample_ij, tSamples, tProbabilities);
            ScalarType tMisfit = tSromCDF - tTrueCDF;
            tSum = tSum + (tMisfit * tMisfit);
        }
        return (tSum);
    }

private:
    OrdinalType mMaxMomentOrder;

    ScalarType mSromSigma;
    ScalarType mWeightCdfMisfit;
    ScalarType mWeightMomentMisfit;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mTrueMoments;
    std::shared_ptr<locus::Distirbution<ScalarType, OrdinalType>> mDistribution;

private:
    SromObjective(const locus::SromObjective<ScalarType, OrdinalType> & aRhs);
    locus::SromObjective<ScalarType, OrdinalType> & operator=(const locus::SromObjective<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

namespace LocusTest
{

TEST(LocusTest, Gaussian)
{
    const double tMean = 0;
    const double tSigma = 1;

    double tEndRange = 1e3;
    double tBeginRange = -1e3;
    double tIncrement = 0.1;
    double tRange = ((tEndRange - tBeginRange) / tIncrement) + 1;
    std::vector<double> tNumbers(tRange);

    std::vector<double> tPDF(tRange);
    std::vector<double> tCDF(tRange);
    for(size_t tIndex = 0; tIndex < tRange; tIndex++)
    {
        tNumbers[tIndex] = tBeginRange + (tIncrement * tIndex);
        tPDF[tIndex] = locus::gaussian_pdf<double>(tNumbers[tIndex], tMean, tSigma);
        tCDF[tIndex] = locus::gaussian_cdf<double>(tNumbers[tIndex], tMean, tSigma);
    }

    const double tBaseValue = 0;
    const double tTolerance = 1e-3;
    double tValue = std::accumulate(tPDF.begin(), tPDF.end(), tBaseValue) / tRange;
    EXPECT_NEAR(tValue, tMean, tTolerance);
}

TEST(LocusTest, SromCDF)
{
    double tSample = 0.276806509167094;
    locus::StandardVector<double> tSamples(4, 0.);
    tSamples[0] = 0.276806509167094;
    tSamples[1] = 0.431107226622461;
    tSamples[2] = 0.004622102620248;
    tSamples[3] = 0.224162021074166;
    locus::StandardVector<double> tSampleProbabilities(4, 0.);
    tSampleProbabilities[0] = 0.25;
    tSampleProbabilities[1] = 0.25;
    tSampleProbabilities[2] = 0.25;
    tSampleProbabilities[3] = 0.25;
    double tSigma = 1e-3;

    double tOutput = locus::compute_srom_cdf<double>(tSigma, tSample, tSamples, tSampleProbabilities);

    double tGold = 0.625;
    double tTolerance = 1e-4;
    EXPECT_NEAR(tGold, tOutput, tTolerance);
}

TEST(LocusTest, SromMoments)
{
    locus::StandardVector<double> tSamples(4, 0.);
    tSamples[0] = 0.276806509167094;
    tSamples[1] = 0.431107226622461;
    tSamples[2] = 0.004622102620248;
    tSamples[3] = 0.224162021074166;
    locus::StandardVector<double> tSampleProbabilities(4, 0.);
    tSampleProbabilities[0] = 0.25;
    tSampleProbabilities[1] = 0.25;
    tSampleProbabilities[2] = 0.25;
    tSampleProbabilities[3] = 0.25;

    locus::StandardVector<double> tMoments(4, 0.);
    for(size_t tIndex = 0; tIndex < tMoments.size(); tIndex++)
    {
        double tOrder = tIndex + static_cast<size_t>(1);
        tMoments[tIndex] = locus::compute_srom_moment<double>(tOrder, tSamples, tSampleProbabilities);
    }

    locus::StandardVector<double> tGold(4, 0.);
    tGold[0] = 0.234174464870992;
    tGold[1] = 0.078186314972017;
    tGold[2] = 0.028149028892565;
    tGold[3] = 0.010734332952929;
    LocusTest::checkVectorData(tMoments, tGold);
}

TEST(LocusTest, factorial)
{
    size_t tGold = 1;
    size_t tValue = locus::factorial<size_t>(0);
    EXPECT_EQ(tGold, tValue);

    tGold = 1;
    tValue = locus::factorial<size_t>(1);
    EXPECT_EQ(tGold, tValue);

    tGold = 362880;
    tValue = locus::factorial<size_t>(9);
    EXPECT_EQ(tGold, tValue);
}

TEST(LocusTest, Beta)
{
    double tAlpha = 1;
    double tBeta = 3;
    double tValue = locus::beta<double>(tAlpha, tBeta);

    double tTolerance = 1e-6;
    double tGold = 1. / 3.;
    EXPECT_NEAR(tGold, tValue, tTolerance);
}

TEST(LocusTest, PochhammerSymbol)
{
    // TEST ONE: NON-FINITE NUMBER CASE
    double tOutput = locus::pochhammer_symbol<double>(-2, 0);
    const double tTolerance = 1e-5;
    EXPECT_NEAR(0, tOutput, tTolerance);

    // TEST TWO: FINITE NUMBER CASE
    tOutput = locus::pochhammer_symbol<double>(2.166666666666666, 4.333333333333333);
    double tGold = 265.98433449717857;
    EXPECT_NEAR(tGold, tOutput, tTolerance);
}

TEST(LocusTest, BetaMoment)
{
    size_t tOrder = 3;
    double tAlpha = 2.166666666666666;
    double tBeta = 4.333333333333333;
    double tValue = locus::beta_moment<double>(tOrder, tAlpha, tBeta);

    double tTolerance = 1e-6;
    double tGold = 0.068990559186638;
    EXPECT_NEAR(tGold, tValue, tTolerance);
}

TEST(LocusTest, ComputeBetaShapeParameters)
{
    const double tMean = 90;
    const double tMaxValue = 135;
    const double tMinValue = 67.5;
    const double tVariance = 135;
    double tAlphaShapeParameter = 0;
    double tBetaShapeParameter = 0;
    locus::beta_shape_parameters<double>(tMinValue,
                                         tMaxValue,
                                         tMean,
                                         tVariance,
                                         tAlphaShapeParameter,
                                         tBetaShapeParameter);

    const double tTolerance = 1e-6;
    const double tGoldAlpha = 2.166666666666666;
    const double tGoldBeta = 4.333333333333333;
    EXPECT_NEAR(tGoldBeta, tBetaShapeParameter, tTolerance);
    EXPECT_NEAR(tGoldAlpha, tAlphaShapeParameter, tTolerance);
}

TEST(LocusTest, IncompleteBeta)
{
    double tSample = 1.;
    const double tAlpha = 2.166666666666666;
    const double tBeta = 4.333333333333333;

    double tOutput = locus::incomplete_beta<double>(tSample, tAlpha, tBeta);

    const double tTolerance = 1e-5;
    const double tGold = locus::beta<double>(tAlpha, tBeta);
    EXPECT_NEAR(tOutput, tGold, tTolerance);
}

TEST(LocusTest, BetaPDF)
{
    const size_t tRange = 4;
    std::vector<double> tPDF(tRange);
    std::vector<double> tShapeParam(tRange);
    tShapeParam[0] = 0.5;
    tShapeParam[1] = 1;
    tShapeParam[2] = 2;
    tShapeParam[3] = 4;
    const double tSample = 0.5;
    for(size_t tIndex = 0; tIndex < tRange; tIndex++)
    {
        tPDF[tIndex] = locus::beta_pdf<double>(tSample, tShapeParam[tIndex], tShapeParam[tIndex]);
    }

    const double tTolerance = 1e-6;
    std::vector<double> tGoldPDF(tRange, 0.);
    tGoldPDF[0] = 0.636619772367582;
    tGoldPDF[1] = 1.0;
    tGoldPDF[2] = 1.5;
    tGoldPDF[3] = 2.1875;
    for(size_t tIndex = 0; tIndex < tRange; tIndex++)
    {
        EXPECT_NEAR(tPDF[tIndex], tGoldPDF[tIndex], tTolerance);
    }
}

TEST(LocusTest, BetaCDF)
{
    const double tBeta = 3;
    const double tSample = 0.5;
    const size_t tRange = 11;
    std::vector<double> tCDF(tRange, 0.);
    for(size_t tIndex = 0; tIndex < tRange; tIndex++)
    {
        tCDF[tIndex] = locus::beta_cdf<double>(tSample, tIndex, tBeta);
    }

    const double tTolerance = 1e-6;
    std::vector<double> tGoldCDF(tRange, 0.);
    tGoldCDF[0] = 1;
    tGoldCDF[1] = 0.875;
    tGoldCDF[2] = 0.6875;
    tGoldCDF[3] = 0.5;
    tGoldCDF[4] = 0.34375;
    tGoldCDF[5] = 0.2265625;
    tGoldCDF[6] = 0.14453125;
    tGoldCDF[7] = 0.08984375;
    tGoldCDF[8] = 0.0546875;
    tGoldCDF[9] = 0.03271484375;
    tGoldCDF[10] = 0.019287109375;
    for(size_t tIndex = 0; tIndex < tRange; tIndex++)
    {
        EXPECT_NEAR(tCDF[tIndex], tGoldCDF[tIndex], tTolerance);
    }
}

} //namespace LocusTest
