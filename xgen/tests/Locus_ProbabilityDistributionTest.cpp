/*
 * Locus_ProbabilityDistributionTest.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "Locus_UnitTestUtils.hpp"

#include <cmath>
#include <random>
#include <vector>
#include <numeric>

#define _MATH_DEFINES_DEFINED

namespace locus
{

inline void compute_beta_shape_parameters(const double & aMinValue,
                                          const double & aMaxValue,
                                          const double & aMean,
                                          const double & aSigma,
                                          double & aBetaShapeParameter,
                                          double & aAlphaShapeParameter)
{
    // Scale mean/variance to lie in [0,1] for the standard beta distribution:
    double tMeanStd = (aMean - aMinValue) / (aMaxValue - aMinValue);
    double tVarianceStd = (static_cast<double>(1) / (aMaxValue - aMinValue))
            * (static_cast<double>(1) / (aMaxValue - aMinValue)) * aSigma;
    // Compute shape parameters for Beta distributions based on standard mean/variance:
    aBetaShapeParameter = tMeanStd
            * (tMeanStd * (static_cast<double>(1) - tMeanStd) / tVarianceStd - static_cast<double>(1));
    aAlphaShapeParameter = (tMeanStd * (static_cast<double>(1) - tMeanStd) / tVarianceStd - static_cast<double>(1))
            - aBetaShapeParameter;
}

inline double beta_pdf(const double & aValue, const double & aAlpha, const double & aBeta)
{
    assert(aBeta > static_cast<double>(0));
    assert(aAlpha > static_cast<double>(0));

    double tNumerator = std::pow(aValue, aAlpha - static_cast<double>(1))
            * std::pow(static_cast<double>(1) - aValue, aBeta - static_cast<double>(1));
    double tDenominator = std::tgamma(aAlpha) * std::tgamma(aBeta) / std::tgamma(aAlpha + aBeta);
    double tOutput = tNumerator / tDenominator;
    return (tOutput);
}

inline double gaussian_pdf(const double & aValue, const double & aMean, const double & aSigma)
{
    double tConstant = static_cast<double>(1)
            / std::sqrt(static_cast<double>(2) * static_cast<double>(M_PI) * aSigma * aSigma);
    double tExponential = std::exp(static_cast<double>(-1) * (aValue - aMean) * (aValue - aMean)
            / (static_cast<double>(2) * aSigma * aSigma));
    double tOutput = tConstant * tExponential;
    return (tOutput);
}

inline double gaussian_cdf(const double & aValue, const double & aMean, const double & aSigma)
{
    double tOutput = static_cast<double>(0.5)
            * (static_cast<double>(1) + std::erf((aValue - aMean) / (aSigma * std::sqrt(static_cast<double>(2)))));
    return (tOutput);
}

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
    for(int tIndex = 0; tIndex < tRange; tIndex++)
    {
        tNumbers[tIndex] = tBeginRange + (tIncrement * tIndex);
        tPDF[tIndex] = locus::gaussian_pdf(tNumbers[tIndex], tMean, tSigma);
        tCDF[tIndex] = locus::gaussian_cdf(tNumbers[tIndex], tMean, tSigma);
    }

    const double tBaseValue = 0;
    const double tTolerance = 1e-3;
    double tValue = std::accumulate(tPDF.begin(), tPDF.end(), tBaseValue) / tRange;
    EXPECT_NEAR(tValue, tMean, tTolerance);
}

} //namespace LocusTest
