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

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include <iostream>

#include "Locus_Criterion.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"

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
inline ScalarType compute_srom_cdf(const ScalarType & aX,
                                   const ScalarType & aSigma,
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
    SromObjective() :
            mSromSigma(1e-3),
            mSqrtConstant(0),
            mWeightCdfMisfit(1),
            mWeightMomentMisfit(1),
            mSromSigmaTimesSigma(0),
            mTrueMoments(),
            mMomentsMisfit(),
            mDistribution()
    {
    }
    explicit SromObjective(const std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> & aDataFactory,
                           const std::shared_ptr<locus::Distirbution<ScalarType, OrdinalType>> & aDistribution) :
            mSromSigma(1e-3),
            mSqrtConstant(0),
            mWeightCdfMisfit(1),
            mWeightMomentMisfit(1),
            mSromSigmaTimesSigma(0),
            mTrueMoments(),
            mMomentsMisfit(aDataFactory->control().create()),
            mDistribution(aDistribution)
    {
        this->initialize(aDataFactory.operator*());
    }
    virtual ~SromObjective()
    {
    }

    void setSromSigma(const ScalarType & aInput)
    {
        mSromSigma = aInput;
        mSromSigmaTimesSigma = mSromSigma * mSromSigma;
    }
    void setCdfMisfitTermWeight(const ScalarType & aInput)
    {
        mWeightCdfMisfit = aInput;
    }
    void setMomentMisfitTermWeight(const ScalarType & aInput)
    {
        mWeightMomentMisfit = aInput;
    }

    void cacheData()
    {
        return;
    }
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        // NOTE: CORRELATION TERM IS NOT IMPLEMENTED YET. THIS TERM WILL BE ADDED IN THE NEAR FUTURE
        const ScalarType tMomentsMisfit = this->computeMomentsMisfit(aControl);
        const ScalarType tCummulativeDistributionFunctionMisfit =
                this->computeCumulativeDistributionFunctionMisfit(aControl);
        const ScalarType tOutput = static_cast<ScalarType>(0.5)
                * (mWeightCdfMisfit * tCummulativeDistributionFunctionMisfit + mWeightMomentMisfit * tMomentsMisfit);
        return (tOutput);
    }

    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        // NOTE: CORRELATION TERM IS NOT IMPLEMENTED YET. THIS TERM WILL BE ADDED IN THE NEAR FUTURE
        assert(aControl.getNumVectors() == aOutput.getNumVectors());
        assert(aControl.getNumVectors() >= static_cast<OrdinalType>(2));

        const OrdinalType tNumSampleDimensions = aOutput.getNumVectors() - static_cast<OrdinalType>(1);
        const locus::Vector<ScalarType, OrdinalType> & tProbabilities = aControl[tNumSampleDimensions];
        locus::Vector<ScalarType, OrdinalType> & tGradientProbabilities = aOutput[tNumSampleDimensions];

        const OrdinalType tNumProbabilities = tProbabilities.size();
        for(OrdinalType tIndexI = 0; tIndexI < tNumSampleDimensions; tIndexI++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMySamples = aControl[tIndexI];
            locus::Vector<ScalarType, OrdinalType> & tMySamplesGradient = aOutput[tIndexI];
            locus::Vector<ScalarType, OrdinalType> & tMyMomentMisfit = mMomentsMisfit->operator[](tIndexI);
            for(OrdinalType tIndexJ = 0; tIndexJ < tNumProbabilities; tIndexJ++)
            {
                // Samples' Gradient
                ScalarType tSample_ij = tMySamples[tIndexJ];
                ScalarType tProbability_ij = tProbabilities[tIndexJ];
                ScalarType tPartialCDFwrtSample =
                        this->partialCumulativeDistributionFunctionWrtSamples(tSample_ij, tProbability_ij, tMySamples, tProbabilities);
                ScalarType tPartialMomentWrtSample = this->partialMomentsWrtSamples(tSample_ij, tProbability_ij, tMyMomentMisfit);
                tMySamplesGradient[tIndexJ] = (mWeightCdfMisfit * tPartialCDFwrtSample) +
                        (mWeightMomentMisfit * tPartialMomentWrtSample);

                // Probabilities' Gradient
                ScalarType tPartialCDFwrtProbability =
                        this->partialCumulativeDistributionFunctionWrtProbabilities(tSample_ij, tMySamples, tProbabilities);
                ScalarType tPartialMomentWrtProbability = this->partialMomentsWrtProbabilities(tSample_ij, tMyMomentMisfit);
                tGradientProbabilities[tIndexJ] = tGradientProbabilities[tIndexJ] +
                        (mWeightCdfMisfit * tPartialCDFwrtProbability + mWeightMomentMisfit * tPartialMomentWrtProbability);
            }
        }
    }

    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::update(static_cast<ScalarType>(1), aVector, static_cast<ScalarType>(0), aOutput);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::SromObjective<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mSromSigmaTimesSigma = mSromSigma * mSromSigma;
        mSqrtConstant = static_cast<ScalarType>(2) * static_cast<ScalarType>(M_PI) * mSromSigmaTimesSigma;
        mSqrtConstant = std::sqrt(mSqrtConstant);

        const OrdinalType tVECTOR_INDEX = 0;
        mTrueMoments = aDataFactory.control(tVECTOR_INDEX).create();
        const OrdinalType tNumMoments = mTrueMoments->size();
        for(OrdinalType tIndex = 0; tIndex < tNumMoments; tIndex++)
        {
            OrdinalType tMyOrder = tIndex + static_cast<OrdinalType>(1);
            mTrueMoments->operator[](tIndex) = mDistribution->moment(tMyOrder);
        }
    }
    ScalarType partialMomentsWrtSamples(const ScalarType & aSampleIJ,
                                        const ScalarType & aProbabilityIJ,
                                        const locus::Vector<ScalarType, OrdinalType> & aMomentsMisfit)
    {
        ScalarType tSum = 0;
        const OrdinalType tNumMoments = aMomentsMisfit.size();
        for(OrdinalType tIndexK = 0; tIndexK < tNumMoments; tIndexK++)
        {
            ScalarType tTrueMomentTimesTrueMoment = mTrueMoments->operator[](tIndexK)
                    * mTrueMoments->operator[](tIndexK);
            ScalarType tConstant = (static_cast<ScalarType>(1) / tTrueMomentTimesTrueMoment);
            ScalarType tMomentOrder = tIndexK + static_cast<OrdinalType>(1);
            tSum = tSum + (tConstant * aMomentsMisfit[tIndexK] * tMomentOrder * aProbabilityIJ
                    * std::pow(aSampleIJ, static_cast<ScalarType>(tIndexK)));
        }
        return (tSum);
    }
    ScalarType partialMomentsWrtProbabilities(const ScalarType & aSampleIJ,
                                              const locus::Vector<ScalarType, OrdinalType> & aMomentsMisfit)
    {
        // Compute sensitivity in dimension k:
        ScalarType tSum = 0;
        const OrdinalType tNumMoments = aMomentsMisfit.size();
        for(OrdinalType tIndexK = 0; tIndexK < tNumMoments; tIndexK++)
        {
            // Sum over first q moments:
            ScalarType tConstant = static_cast<ScalarType>(1)
                    / (std::pow(mTrueMoments->operator[](tIndexK), static_cast<ScalarType>(2)));
            ScalarType tExponent = tIndexK + static_cast<OrdinalType>(1);
            tSum = tSum + (tConstant * aMomentsMisfit[tIndexK] * std::pow(aSampleIJ, tExponent));
        }
        return (tSum);
    }
    ScalarType partialCumulativeDistributionFunctionWrtSamples(const ScalarType & aSampleIJ,
                                                               const ScalarType & aProbabilityIJ,
                                                               const locus::Vector<ScalarType, OrdinalType> & aSamples,
                                                               const locus::Vector<ScalarType, OrdinalType> & aProbabilities)
    {
        ScalarType tSum = 0;
        const OrdinalType tNumProbabilities = aProbabilities.size();
        for(OrdinalType tProbIndexK = 0; tProbIndexK < tNumProbabilities; tProbIndexK++)
        {
            ScalarType tSample_ik = aSamples[tProbIndexK];
            ScalarType tConstant = aProbabilities[tProbIndexK] / mSqrtConstant;
            ScalarType tExponent = (static_cast<ScalarType>(-1) / (static_cast<ScalarType>(2) * mSromSigmaTimesSigma))
                    * (aSampleIJ - tSample_ik) * (aSampleIJ - tSample_ik);
            tSum = tSum + (tConstant * std::exp(tExponent));
        }

        const ScalarType tTruePDF = mDistribution->pdf(aSampleIJ);
        const ScalarType tTrueCDF = mDistribution->cdf(aSampleIJ);
        const ScalarType tSromCDF = locus::compute_srom_cdf<ScalarType, OrdinalType>(aSampleIJ, mSromSigma, aSamples, aProbabilities);
        const ScalarType tMisfitCDF = tSromCDF - tTrueCDF;
        const ScalarType tTermOne = tMisfitCDF * (tSum - tTruePDF);

        ScalarType tTermTwo = 0;
        for(OrdinalType tProbIndexL = 0; tProbIndexL < tNumProbabilities; tProbIndexL++)
        {
            ScalarType tSample_il= aSamples[tProbIndexL];
            ScalarType tTrueCDF = mDistribution->cdf(tSample_il);
            ScalarType tSromCDF = locus::compute_srom_cdf<ScalarType, OrdinalType>(tSample_il, mSromSigma, aSamples, aProbabilities);
            ScalarType tMisfitCDF = tSromCDF - tTrueCDF;
            ScalarType tConstant = aProbabilityIJ / mSqrtConstant;
            ScalarType tExponent = (static_cast<ScalarType>(-1) / (static_cast<ScalarType>(2) * mSromSigmaTimesSigma))
                    * (tSample_il - aSampleIJ) * (tSample_il - aSampleIJ);
            tTermTwo = tTermTwo + (tMisfitCDF * tConstant * std::exp(tExponent));
        }

        const ScalarType tOutput = tTermOne - tTermTwo;
        return (tOutput);
    }
    ScalarType partialCumulativeDistributionFunctionWrtProbabilities(const ScalarType & aSampleIJ,
                                                                     const locus::Vector<ScalarType, OrdinalType> & aSamples,
                                                                     const locus::Vector<ScalarType, OrdinalType> & aProbabilities)
    {
        ScalarType tSum = 0;
        const OrdinalType tNumProbabilities = aProbabilities.size();
        for(OrdinalType tProbIndexK = 0; tProbIndexK < tNumProbabilities; tProbIndexK++)
        {
            // Compute different in true/SROM CDFs at x_ij:
            ScalarType tTrueCDF = mDistribution->cdf(aSamples[tProbIndexK]);
            ScalarType tSromCDF =
                    locus::compute_srom_cdf<ScalarType, OrdinalType>(aSamples[tProbIndexK], mSromSigma, aSamples, aProbabilities);
            ScalarType tMisfitCDF = tSromCDF - tTrueCDF;
            // Compute CDF derivative term:
            ScalarType tNumerator = aSamples[tProbIndexK] - aSampleIJ;
            ScalarType tDenominator = std::sqrt(static_cast<ScalarType>(2)) * mSromSigma;
            ScalarType tSensitivity = static_cast<ScalarType>(1) + std::erf(tNumerator / tDenominator);
            // Add contribution to summation in k:
            tSum = tSum + (tMisfitCDF * tSensitivity);
        }
        tSum = static_cast<ScalarType>(0.5) * tSum;
        return (tSum);
    }
    /*!
     * Compute misfit in moments up to order q for ith dimension (i.e. i-th random vector dimension).
     * Currently, the random vector dimension is always set to one. Hence, random vector has size one
     * and samples are not correlated.
     **/
    ScalarType computeMomentsMisfit(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() >= static_cast<OrdinalType>(2));
        const OrdinalType tNumSampleDimensions = aControl.getNumVectors() - static_cast<OrdinalType>(1);
        const locus::Vector<ScalarType, OrdinalType> & tProbabilities = aControl[tNumSampleDimensions];

        ScalarType tTotalSum = 0;
        const OrdinalType tMaxMomentOrder = mTrueMoments->size();
        for(OrdinalType tDimIndex = 0; tDimIndex < tNumSampleDimensions; tDimIndex++)
        {
            ScalarType tMomentSum = 0;
            const locus::Vector<ScalarType, OrdinalType> & tMySamples = aControl[tDimIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyMomentMisfit = mMomentsMisfit->operator[](tDimIndex);
            for(OrdinalType tMomentIndex = 0; tMomentIndex < tMaxMomentOrder; tMomentIndex++)
            {
                OrdinalType tMomentOrder = tMomentIndex + static_cast<OrdinalType>(1);
                ScalarType tSromMoment =
                        locus::compute_srom_moment<ScalarType, OrdinalType>(tMomentOrder, tMySamples, tProbabilities);
                tMyMomentMisfit[tMomentIndex] = tSromMoment - mTrueMoments->operator[](tMomentIndex);
                ScalarType tValue = tMyMomentMisfit[tMomentIndex] / mTrueMoments->operator[](tMomentIndex);
                tMomentSum = tMomentSum + (tValue * tValue);
            }
            tTotalSum = tTotalSum + tMomentSum;
        }
        return (tTotalSum);
    }
    /*!
     * Compute misfit in Cumulative Distribution Function (CDF) for i-th dimension (i.e. i-th random vector
     * dimension). Currently, the random vector dimension is always set to one. Hence, random vector has size
     * one and samples are not correlated.
     **/
    ScalarType computeCumulativeDistributionFunctionMisfit(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() >= static_cast<OrdinalType>(2));
        const OrdinalType tNumSampleDimensions = aControl.getNumVectors() - static_cast<OrdinalType>(1);
        const locus::Vector<ScalarType, OrdinalType> & tProbabilities = aControl[tNumSampleDimensions];

        ScalarType tTotalSum = 0;
        const OrdinalType tNumProbabilities = tProbabilities.size();
        for(OrdinalType tDimIndex = 0; tDimIndex < tNumSampleDimensions; tDimIndex++)
        {
            ScalarType tMySampleSum = 0;
            const locus::Vector<ScalarType, OrdinalType> & tMySamples = aControl[tDimIndex];
            for(OrdinalType tProbIndex = 0; tProbIndex < tNumProbabilities; tProbIndex++)
            {
                ScalarType tSample_ij = tMySamples[tProbIndex];
                ScalarType tTrueCDF = mDistribution->cdf(tSample_ij);
                ScalarType tSromCDF =
                        locus::compute_srom_cdf<ScalarType, OrdinalType>(tSample_ij, mSromSigma, tMySamples, tProbabilities);
                ScalarType tMisfit = tSromCDF - tTrueCDF;
                tMySampleSum = tMySampleSum + (tMisfit * tMisfit);
            }
            tTotalSum = tTotalSum + tMySampleSum;
        }
        return (tTotalSum);
    }

private:
    ScalarType mSromSigma;
    ScalarType mSqrtConstant;
    ScalarType mWeightCdfMisfit;
    ScalarType mWeightMomentMisfit;
    ScalarType mSromSigmaTimesSigma;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mTrueMoments;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mMomentsMisfit;
    std::shared_ptr<locus::Distirbution<ScalarType, OrdinalType>> mDistribution;

private:
    SromObjective(const locus::SromObjective<ScalarType, OrdinalType> & aRhs);
    locus::SromObjective<ScalarType, OrdinalType> & operator=(const locus::SromObjective<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class SromConstraint : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    SromConstraint(const std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> & aReductionOperations) :
            mReductionOperations(aReductionOperations)
    {
    }
    virtual ~SromConstraint()
    {
    }

    void cacheData()
    {
        return;
    }
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const OrdinalType tVectorIndex = aControl.getNumVectors() - static_cast<OrdinalType>(1);
        const locus::Vector<ScalarType, OrdinalType> & tProbabilities = aControl[tVectorIndex];
        ScalarType tSum = mReductionOperations->sum(tProbabilities);
        ScalarType tOutput = tSum - static_cast<ScalarType>(1);
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        const OrdinalType tNumDimensions = aControl.getNumVectors() - static_cast<OrdinalType>(1);
        for(OrdinalType tIndex = 0; tIndex < tNumDimensions; tIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMySamplesGradient = aOutput[tIndex];
            tMySamplesGradient.fill(static_cast<ScalarType>(0));
        }
        locus::Vector<ScalarType, OrdinalType> & tMyProbabilityGradient = aOutput[tNumDimensions];
        tMyProbabilityGradient.fill(static_cast<ScalarType>(1));
    }
    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::fill(static_cast<ScalarType>(0), aOutput);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::SromConstraint<ScalarType, OrdinalType>>(mReductionOperations);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mReductionOperations;

private:
    SromConstraint(const locus::SromConstraint<ScalarType, OrdinalType> & aRhs);
    locus::SromConstraint<ScalarType, OrdinalType> & operator=(const locus::SromConstraint<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Diagnostics
{
public:
    Diagnostics() :
        mRandomNumLowerBound(0.05),
        mRandomNumUpperBound(0.1),
        mInitialSuperscript(1),
        mFinalSuperscript(8)
    {
    }
    ~Diagnostics()
    {
    }

    void setRandomNumberLowerBound(const ScalarType & aInput)
    {
        mRandomNumLowerBound = aInput;
    }

    void setRandomNumberUpperBound(const ScalarType & aInput)
    {
        mRandomNumUpperBound = aInput;
    }

    void setFinalSuperscript(const OrdinalType & aInput)
    {
        mFinalSuperscript = aInput;
    }

    void setInitialSuperscript(const OrdinalType & aInput)
    {
        mInitialSuperscript = aInput;
    }

    void checkCriterionGradient(locus::Criterion<ScalarType, OrdinalType> & aCriterion,
                                locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                std::ostringstream & aOutputMsg)
    {
        aOutputMsg << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
                   << std::setw(20) << "abs(Error)" << "\n";

        const OrdinalType tNumVectors = aControl.getNumVectors();
        assert(tNumVectors > static_cast<OrdinalType>(0));
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tStep = aControl.create();
        assert(tStep.get() != nullptr);
        unsigned int tRANDOM_SEED = 1;
        std::srand(tRANDOM_SEED);
        this->random(mRandomNumLowerBound, mRandomNumUpperBound, *tStep);
        this->random(mRandomNumLowerBound, mRandomNumUpperBound, aControl);
        // NOTE: Think how to syncronize if working with owned and shared data

        aCriterion.value(aControl);
        aCriterion.cacheData();
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tGradient = aControl.create();
        assert(tGradient.get() != nullptr);
        aCriterion.gradient(aControl, *tGradient);

        const ScalarType tGradientDotStep = locus::dot(*tGradient, *tStep);
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tWork = aControl.create();
        assert(tWork.get() != nullptr);
        for(OrdinalType tIndex = mInitialSuperscript; tIndex <= mFinalSuperscript; tIndex++)
        {
            // Compute \hat{x} = x + \epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            ScalarType tEpsilon = static_cast<ScalarType>(1) /
                    std::pow(static_cast<ScalarType>(10), tIndex);
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            locus::update(tEpsilon, *tStep, static_cast<ScalarType>(1), *tWork);
            ScalarType tObjectiveValueAtPlusEpsilon = aCriterion.value(*tWork);

            // Compute \hat{x} = x - \epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            ScalarType tMultiplier = static_cast<ScalarType>(-1) * tEpsilon;
            locus::update(tMultiplier, *tStep, static_cast<ScalarType>(1), *tWork);
            ScalarType tObjectiveValueAtMinusEpsilon = aCriterion.value(*tWork);

            // Compute \hat{x} = x + 2\epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            tMultiplier = static_cast<ScalarType>(2) * tEpsilon;
            locus::update(tMultiplier, *tStep, static_cast<ScalarType>(1), *tWork);
            ScalarType tObjectiveValueAtPlusTwoEpsilon = aCriterion.value(*tWork);

            // Compute \hat{x} = x - 2\epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            tMultiplier = static_cast<ScalarType>(-2) * tEpsilon;
            locus::update(tMultiplier, *tStep, static_cast<ScalarType>(1), *tWork);
            ScalarType tObjectiveValueAtMinusTwoEpsilon = aCriterion.value(*tWork);

            // Compute objective value approximation via a five point stencil finite difference procedure
            ScalarType tObjectiveAppx = (-tObjectiveValueAtPlusTwoEpsilon
                    + static_cast<ScalarType>(8) * tObjectiveValueAtPlusEpsilon
                    - static_cast<ScalarType>(8) * tObjectiveValueAtMinusEpsilon
                    + tObjectiveValueAtMinusTwoEpsilon) / (static_cast<ScalarType>(12) * tEpsilon);

            ScalarType tAppxError = std::abs(tObjectiveAppx - tGradientDotStep);
            aOutputMsg << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
            << tGradientDotStep << std::setw(19) << tObjectiveAppx << std::setw(19) << tAppxError << "\n";
        }
    }

    void checkCriterionHessian(locus::Criterion<ScalarType, OrdinalType> & aCriterion,
                               locus::MultiVector<ScalarType, OrdinalType> & aControl,
                               std::ostringstream & aOutputMsg)
    {
        aOutputMsg << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Hess*Step " << std::setw(18) << "FD Approx"
                   << std::setw(20) << "abs(Error)" << "\n";

        const OrdinalType tNumVectors = aControl.getNumVectors();
        assert(tNumVectors > static_cast<OrdinalType>(0));
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tStep = aControl.create();
        assert(tStep.get() != nullptr);
        unsigned int tRANDOM_SEED = 1;
        std::srand(tRANDOM_SEED);
        this->random(mRandomNumLowerBound, mRandomNumUpperBound, *tStep);
        this->random(mRandomNumLowerBound, mRandomNumUpperBound, aControl);
        // NOTE: Think how to syncronize if working with owned and shared data

        // Compute true Hessian times Step and corresponding norm value
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tHessianTimesStep = aControl.create();
        assert(tHessianTimesStep.get() != nullptr);
        aCriterion.hessian(aControl, *tStep, *tHessianTimesStep);
        const ScalarType tNormHesianTimesStep = locus::norm(*tHessianTimesStep);

        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tGradient = aControl.create();
        assert(tGradient.get() != nullptr);
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tAppxHessianTimesStep = aControl.create();
        assert(tAppxHessianTimesStep.get() != nullptr);

        // Compute 5-point stencil finite difference approximation
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tWork = aControl.create();
        assert(tWork.get() != nullptr);
        for(OrdinalType tIndex = mInitialSuperscript; tIndex <= mFinalSuperscript; tIndex++)
        {
            // Compute \hat{x} = x + \epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            ScalarType tEpsilon = static_cast<ScalarType>(1) /
                    std::pow(static_cast<ScalarType>(10), tIndex);
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            locus::update(tEpsilon, *tStep, static_cast<ScalarType>(1), *tWork);
            this->gradient(*tWork, *tGradient, aCriterion);
            locus::update(static_cast<ScalarType>(8), *tGradient, static_cast<ScalarType>(0), *tAppxHessianTimesStep);

            // Compute \hat{x} = x - \epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            ScalarType tMultiplier = static_cast<ScalarType>(-1) * tEpsilon;
            locus::update(tMultiplier, *tStep, static_cast<ScalarType>(1), *tWork);
            this->gradient(*tWork, *tGradient, aCriterion);
            locus::update(static_cast<ScalarType>(-8), *tGradient, static_cast<ScalarType>(1), *tAppxHessianTimesStep);

            // Compute \hat{x} = x + 2\epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            tMultiplier = static_cast<ScalarType>(2) * tEpsilon;
            locus::update(tMultiplier, *tStep, static_cast<ScalarType>(1), *tWork);
            this->gradient(*tWork, *tGradient, aCriterion);
            locus::update(static_cast<ScalarType>(-1), *tGradient, static_cast<ScalarType>(1), *tAppxHessianTimesStep);

            // Compute \hat{x} = x - 2\epsilon\Delta{x}, where x denotes the control vector and \Delta{x} denotes the step.
            locus::update(static_cast<ScalarType>(1), aControl, static_cast<ScalarType>(0), *tWork);
            tMultiplier = static_cast<ScalarType>(-2) * tEpsilon;
            locus::update(tMultiplier, *tStep, static_cast<ScalarType>(1), *tWork);
            this->gradient(*tWork, *tGradient, aCriterion);
            locus::update(static_cast<ScalarType>(1), *tGradient, static_cast<ScalarType>(1), *tAppxHessianTimesStep);

            // Comptute \frac{F(x)}{12}, where F(x) denotes the finite difference approximation of \nabla_{x}^{2}f(x)\Delta{x}
            // and f(x) denotes the respective criterion being evaluated/tested.
            tMultiplier = static_cast<ScalarType>(1) / (static_cast<ScalarType>(12) * tEpsilon);
            locus::scale(tMultiplier, *tAppxHessianTimesStep);
            ScalarType tNormAppxHesianTimesStep = locus::norm(*tAppxHessianTimesStep);

            // Compute error between true and finite differenced Hessian times step calculation.
            locus::update(static_cast<ScalarType>(1), *tHessianTimesStep, static_cast<ScalarType>(-1), *tAppxHessianTimesStep);
            ScalarType tNumerator = locus::norm(*tAppxHessianTimesStep);
            ScalarType tDenominator = std::numeric_limits<ScalarType>::epsilon() + tNormHesianTimesStep;
            ScalarType tAppxError = tNumerator / tDenominator;

            aOutputMsg << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
            << tNormHesianTimesStep << std::setw(19) << tNormAppxHesianTimesStep << std::setw(19) << tAppxError << "\n";
        }
    }

private:
    void random(const ScalarType & aLowerBound, const ScalarType & aUpperBound, locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        const OrdinalType tNumVectors = aInput.getNumVectors();
        assert(tNumVectors > static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyVector = aInput[tVectorIndex];
            const OrdinalType tMyLength = tMyVector.size();
            assert(tMyLength > static_cast<ScalarType>(0));
            for(OrdinalType tElemIndex = 0; tElemIndex < tMyLength; tElemIndex++)
            {
                tMyVector[tElemIndex] = aLowerBound + ((aUpperBound - aLowerBound) * static_cast<ScalarType>(std::rand() / RAND_MAX));
            }
        }
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aGradient,
                  locus::Criterion<ScalarType, OrdinalType> & aCriterion)
    {
        locus::fill(static_cast<ScalarType>(0), aGradient);
        aCriterion.value(aControl);
        aCriterion.cacheData();
        aCriterion.gradient(aControl, aGradient);
    }

private:
    ScalarType mRandomNumLowerBound;
    ScalarType mRandomNumUpperBound;
    OrdinalType mInitialSuperscript;
    OrdinalType mFinalSuperscript;

private:
    Diagnostics(const locus::Diagnostics<ScalarType, OrdinalType> & aRhs);
    locus::Diagnostics<ScalarType, OrdinalType> & operator=(const locus::Diagnostics<ScalarType, OrdinalType> & aRhs);
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

    double tOutput = locus::compute_srom_cdf<double>(tSample, tSigma, tSamples, tSampleProbabilities);

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

TEST(LocusTest, BetaDistribution)
{
    const double tMean = 90;
    const double tMax = 135;
    const double tMin = 67.5;
    const double tVariance = 135;
    locus::Beta<double> tDistribution(tMin, tMax, tMean, tVariance);

    // TEST INPUTS
    const double tTolerance = 1e-5;
    EXPECT_NEAR(tMin, tDistribution.min(), tTolerance);
    EXPECT_NEAR(tMax, tDistribution.max(), tTolerance);
    EXPECT_NEAR(tMean, tDistribution.mean(), tTolerance);
    EXPECT_NEAR(tVariance, tDistribution.variance(), tTolerance);

    // TEST BETA PDF & CDF
    double tSample = 0.276806509167094;
    double tGoldPDF = 2.179085850493935;
    EXPECT_NEAR(tGoldPDF, tDistribution.pdf(tSample), tTolerance);
    double tGoldCDF = 0.417022004702574;
    EXPECT_NEAR(tGoldCDF, tDistribution.cdf(tSample), tTolerance);

    // TEST BETA MOMENTS
    const size_t tNumMoments = 4;
    locus::StandardVector<double> tGoldMoments(tNumMoments);
    tGoldMoments[0] = 0.333333333333333;
    tGoldMoments[1] = 0.140740740740740;
    tGoldMoments[2] = 0.068990559186638;
    tGoldMoments[3] = 0.037521181312031;
    for(size_t tIndex = 0; tIndex < tNumMoments; tIndex++)
    {
        size_t tOrder = tIndex + 1;
        EXPECT_NEAR(tGoldMoments[tIndex], tDistribution.moment(tOrder), tTolerance);
    }
}

TEST(LocusTest, SromObjectiveTestOne)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumVectors = 2;
    const size_t tNumControls = 4;
    tDataFactory->allocateControl(tNumControls, tNumVectors);

    // ********* ALLOCATE BETA DISTRIBUTION *********
    const double tMean = 90;
    const double tMax = 135;
    const double tMin = 67.5;
    const double tVariance = 135;
    std::shared_ptr<locus::Beta<double>> tDistribution =
            std::make_shared<locus::Beta<double>>(tMin, tMax, tMean, tVariance);

    // ********* SET TEST DATA: SAMPLES AND PROBABILITIES *********
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    size_t tVectorIndex = 0;
    tControl(tVectorIndex, 0) = 0.276806509167094;
    tControl(tVectorIndex, 1) = 0.431107226622461;
    tControl(tVectorIndex, 2) = 0.004622102620248;
    tControl(tVectorIndex, 3) = 0.224162021074166;
    tVectorIndex = 1;
    tControl[tVectorIndex].fill(0.25);

    // ********* TEST OBJECTIVE FUNCTION *********
    locus::SromObjective<double> tObjective(tDataFactory, tDistribution);
    double tValue = tObjective.value(tControl);
    double tTolerance = 1e-5;
    double tGold = 0.617109315688096;
    EXPECT_NEAR(tGold, tValue, tTolerance);

    // ********* TEST OBJECTIVE GRADIENT *********
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tObjective.gradient(tControl, tGradient);
    locus::StandardMultiVector<double> tGradientGold(tNumVectors, tNumControls);
    tVectorIndex = 0;
    tGradientGold(tVectorIndex, 0) = -2.010045017107233;
    tGradientGold(tVectorIndex, 1) = -3.878346258927178;
    tGradientGold(tVectorIndex, 2) = -0.237208262654126;
    tGradientGold(tVectorIndex, 3) = -1.271234346951175;
    tVectorIndex = 1;
    tGradientGold(tVectorIndex, 0) = -0.524038145360132;
    tGradientGold(tVectorIndex, 1) = -2.239056684273221;
    tGradientGold(tVectorIndex, 2) = 0.493570515676146;
    tGradientGold(tVectorIndex, 3) = -0.104442116117926;
    LocusTest::checkMultiVectorData(tGradient, tGradientGold);
}

TEST(LocusTest, SromObjectiveTestTwo)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumVectors = 3;
    const size_t tNumControls = 4;
    tDataFactory->allocateControl(tNumControls, tNumVectors);

    // ********* ALLOCATE BETA DISTRIBUTION *********
    const double tMean = 90;
    const double tMax = 135;
    const double tMin = 67.5;
    const double tVariance = 135;
    std::shared_ptr<locus::Beta<double>> tDistribution =
            std::make_shared<locus::Beta<double>>(tMin, tMax, tMean, tVariance);

    // ********* SET TEST DATA: SAMPLES AND PROBABILITIES *********
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    size_t tVectorIndex = 0;
    tControl(tVectorIndex, 0) = 0.276806509167094;
    tControl(tVectorIndex, 1) = 0.004622102620248;
    tControl(tVectorIndex, 2) = 0.376806509167094;
    tControl(tVectorIndex, 3) = 0.104622102620248;
    tVectorIndex = 1;
    tControl(tVectorIndex, 0) = 0.431107226622461;
    tControl(tVectorIndex, 1) = 0.224162021074166;
    tControl(tVectorIndex, 2) = 0.531107226622461;
    tControl(tVectorIndex, 3) = 0.324162021074166;
    tVectorIndex = 2;
    tControl[tVectorIndex].fill(0.25);

    // ********* TEST OBJECTIVE FUNCTION *********
    locus::SromObjective<double> tObjective(tDataFactory, tDistribution);
    double tValue = tObjective.value(tControl);
    double tTolerance = 1e-5;
    double tGold = 1.032230626961365;
    EXPECT_NEAR(tGold, tValue, tTolerance);

    // ********* TEST OBJECTIVE GRADIENT *********
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tObjective.gradient(tControl, tGradient);
    locus::StandardMultiVector<double> tGradientGold(tNumVectors, tNumControls);
    tVectorIndex = 0;
    tGradientGold(tVectorIndex, 0) = -2.42724408126656;
    tGradientGold(tVectorIndex, 1) = -0.337450847795798;
    tGradientGold(tVectorIndex, 2) = -3.887791716578634;
    tGradientGold(tVectorIndex, 3) = -1.076413326527892;
    tVectorIndex = 1;
    tGradientGold(tVectorIndex, 0) = 0.096246202011561;
    tGradientGold(tVectorIndex, 1) = 0.520617569090164;
    tGradientGold(tVectorIndex, 2) = -0.321363712239195;
    tGradientGold(tVectorIndex, 3) = 0.384504837554259;
    tVectorIndex = 2;
    tGradientGold(tVectorIndex, 0) = -0.53206506489113;
    tGradientGold(tVectorIndex, 1) = 0.619653114279367;
    tGradientGold(tVectorIndex, 2) = -1.84853491196106;
    tGradientGold(tVectorIndex, 3) = 0.426963908092988;
    LocusTest::checkMultiVectorData(tGradient, tGradientGold);
}

TEST(LocusTest, SromConstraint)
{
    // ********* SET TEST DATA: SAMPLES AND PROBABILITIES *********
    const size_t tNumVectors = 2;
    const size_t tNumControls = 4;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    size_t tVectorIndex = 0;
    tControl(tVectorIndex, 0) = 0.183183326166505;
    tControl(tVectorIndex, 1) = 0.341948604575779;
    tControl(tVectorIndex, 2) = 0.410656896223290;
    tControl(tVectorIndex, 3) = 0.064209040541960;
    tVectorIndex = 1;
    tControl(tVectorIndex, 0) = 0.434251989288042;
    tControl(tVectorIndex, 1) = 0.351721349341024;
    tControl(tVectorIndex, 2) = 0.001250000000000;
    tControl(tVectorIndex, 3) = 0.212776663693648;

    // ********* TEST CONSTRAINT EVALUATION *********
    std::shared_ptr<locus::StandardVectorReductionOperations<double>> tReductions =
            std::make_shared<locus::StandardVectorReductionOperations<double>>();
    locus::SromConstraint<double> tConstraint(tReductions);
    double tValue = tConstraint.value(tControl);

    double tGoldValue = 0;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tValue, tTolerance);

    // ********* TEST CONSTRAINT GRADIENT *********
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tConstraint.gradient(tControl, tGradient);
    locus::StandardMultiVector<double> tGradientGold(tNumVectors, tNumControls);
    tVectorIndex = 0;
    tGradientGold(tVectorIndex, 0) = 0;
    tGradientGold(tVectorIndex, 1) = 0;
    tGradientGold(tVectorIndex, 2) = 0;
    tGradientGold(tVectorIndex, 3) = 0;
    tVectorIndex = 1;
    tGradientGold(tVectorIndex, 0) = 1;
    tGradientGold(tVectorIndex, 1) = 1;
    tGradientGold(tVectorIndex, 2) = 1;
    tGradientGold(tVectorIndex, 3) = 1;
    LocusTest::checkMultiVectorData(tGradient, tGradientGold);
}

TEST(LocusTest, CheckSromObjectiveGradient)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumVectors = 2;
    const size_t tNumControls = 4;
    tDataFactory->allocateControl(tNumControls, tNumVectors);

    // ********* ALLOCATE BETA DISTRIBUTION *********
    const double tMean = 90;
    const double tMax = 135;
    const double tMin = 67.5;
    const double tVariance = 135;
    std::shared_ptr<locus::Beta<double>> tDistribution =
            std::make_shared<locus::Beta<double>>(tMin, tMax, tMean, tVariance);

    // ********* CHECK CONSTRAINT GRADIENT *********
    std::ostringstream tMsg;
    locus::Diagnostics<double> tDiagnostics;
    locus::SromObjective<double> tObjective(tDataFactory, tDistribution);
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    tDiagnostics.checkCriterionGradient(tObjective, tControl, tMsg);
    std::cout << tMsg.str().c_str();
}

TEST(LocusTest, CheckSromConstraintGradient)
{
    // ********* CHECK CONSTRAINT GRADIENT *********
    std::shared_ptr<locus::StandardVectorReductionOperations<double>> tReductions =
            std::make_shared<locus::StandardVectorReductionOperations<double>>();
    locus::SromConstraint<double> tConstraint(tReductions);

    const size_t tNumVectors = 2;
    const size_t tNumControls = 4;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);

    std::ostringstream tMsg;
    locus::Diagnostics<double> tDiagnostics;
    tDiagnostics.checkCriterionGradient(tConstraint, tControl, tMsg);
    std::cout << tMsg.str().c_str();
}

TEST(LocusTest, SolveSromProblem)
{

}

} //namespace LocusTest
