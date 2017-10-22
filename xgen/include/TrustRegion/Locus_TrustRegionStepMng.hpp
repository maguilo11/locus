/*
 * Locus_TrustRegionStepMng.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_TRUSTREGIONSTEPMNG_HPP_
#define LOCUS_TRUSTREGIONSTEPMNG_HPP_

#include <cmath>
#include <limits>

#include "Locus_SteihaugTointSolver.hpp"
#include "Locus_TrustRegionStageMng.hpp"
#include "Locus_TrustRegionAlgorithmDataMng.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class TrustRegionStepMng
{
public:
    TrustRegionStepMng() :
            mActualReduction(0),
            mTrustRegionRadius(1e4),
            mPredictedReduction(0),
            mMinTrustRegionRadius(1e-4),
            mMaxTrustRegionRadius(1e4),
            mTrustRegionExpansion(2.),
            mTrustRegionContraction(0.5),
            mMinCosineAngleTolerance(1e-2),
            mGradientInexactnessTolerance(std::numeric_limits<ScalarType>::max()),
            mObjectiveInexactnessTolerance(std::numeric_limits<ScalarType>::max()),
            mActualOverPredictedReduction(0),
            mActualOverPredictedReductionMidBound(0.25),
            mActualOverPredictedReductionLowerBound(0.1),
            mActualOverPredictedReductionUpperBound(0.75),
            mGradientInexactnessToleranceConstant(1),
            mObjectiveInexactnessToleranceConstant(1),
            mNumTrustRegionSubProblemItrDone(0),
            mMaxNumTrustRegionSubProblemItr(30),
            mIsInitialTrustRegionSetToNormProjectedGradient(true)
    {
    }

    virtual ~TrustRegionStepMng()
    {
    }

    void setTrustRegionRadius(const ScalarType & aInput)
    {
        mTrustRegionRadius = aInput;
    }
    ScalarType getTrustRegionRadius() const
    {
        return (mTrustRegionRadius);
    }
    void setTrustRegionContraction(const ScalarType & aInput)
    {
        mTrustRegionContraction = aInput;
    }
    ScalarType getTrustRegionContraction() const
    {
        return (mTrustRegionContraction);
    }
    void setTrustRegionExpansion(const ScalarType & aInput)
    {
        mTrustRegionExpansion = aInput;
    }
    ScalarType getTrustRegionExpansion() const
    {
        return (mTrustRegionExpansion);
    }
    void setMinTrustRegionRadius(const ScalarType & aInput)
    {
        mMinTrustRegionRadius = aInput;
    }
    ScalarType getMinTrustRegionRadius() const
    {
        return (mMinTrustRegionRadius);
    }
    void setMaxTrustRegionRadius(const ScalarType & aInput)
    {
        mMaxTrustRegionRadius = aInput;
    }
    ScalarType getMaxTrustRegionRadius() const
    {
        return (mMaxTrustRegionRadius);
    }

    void setGradientInexactnessToleranceConstant(const ScalarType & aInput)
    {
        mGradientInexactnessToleranceConstant = aInput;
    }
    ScalarType getGradientInexactnessToleranceConstant() const
    {
        return (mGradientInexactnessToleranceConstant);
    }
    void updateGradientInexactnessTolerance(const ScalarType & aInput)
    {
        ScalarType tMinValue = std::min(mTrustRegionRadius, aInput);
        mGradientInexactnessTolerance = mGradientInexactnessToleranceConstant * tMinValue;
    }
    ScalarType getGradientInexactnessTolerance() const
    {
        return (mGradientInexactnessTolerance);
    }

    void setObjectiveInexactnessToleranceConstant(const ScalarType & aInput)
    {
        mObjectiveInexactnessToleranceConstant = aInput;
    }
    ScalarType getObjectiveInexactnessToleranceConstant() const
    {
        return (mObjectiveInexactnessToleranceConstant);
    }
    void updateObjectiveInexactnessTolerance(const ScalarType & aInput)
    {
        mObjectiveInexactnessTolerance = mObjectiveInexactnessToleranceConstant
                * mActualOverPredictedReductionLowerBound * std::abs(aInput);
    }
    ScalarType getObjectiveInexactnessTolerance() const
    {
        return (mObjectiveInexactnessTolerance);
    }


    void setActualOverPredictedReductionMidBound(const ScalarType & aInput)
    {
        mActualOverPredictedReductionMidBound = aInput;
    }
    ScalarType getActualOverPredictedReductionMidBound() const
    {
        return (mActualOverPredictedReductionMidBound);
    }
    void setActualOverPredictedReductionLowerBound(const ScalarType & aInput)
    {
        mActualOverPredictedReductionLowerBound = aInput;
    }
    ScalarType getActualOverPredictedReductionLowerBound() const
    {
        return (mActualOverPredictedReductionLowerBound);
    }
    void setActualOverPredictedReductionUpperBound(const ScalarType & aInput)
    {
        mActualOverPredictedReductionUpperBound = aInput;
    }
    ScalarType getActualOverPredictedReductionUpperBound() const
    {
        return (mActualOverPredictedReductionUpperBound);
    }

    void setActualReduction(const ScalarType & aInput)
    {
        mActualReduction = aInput;
    }
    ScalarType getActualReduction() const
    {
        return (mActualReduction);
    }
    void setPredictedReduction(const ScalarType & aInput)
    {
        mPredictedReduction = aInput;
    }
    ScalarType getPredictedReduction() const
    {
        return (mPredictedReduction);
    }
    void setMinCosineAngleTolerance(const ScalarType & aInput)
    {
        mMinCosineAngleTolerance = aInput;
    }
    ScalarType getMinCosineAngleTolerance() const
    {
        return (mMinCosineAngleTolerance);
    }
    void setActualOverPredictedReduction(const ScalarType & aInput)
    {
        mActualOverPredictedReduction = aInput;
    }
    ScalarType getActualOverPredictedReduction() const
    {
        return (mActualOverPredictedReduction);
    }

    void setNumTrustRegionSubProblemItrDone(const OrdinalType & aInput)
    {
        mNumTrustRegionSubProblemItrDone = aInput;
    }
    void updateNumTrustRegionSubProblemItrDone()
    {
        mNumTrustRegionSubProblemItrDone++;
    }
    OrdinalType getNumTrustRegionSubProblemItrDone() const
    {
        return (mNumTrustRegionSubProblemItrDone);
    }
    void setMaxNumTrustRegionSubProblemItr(const OrdinalType & aInput)
    {
        mMaxNumTrustRegionSubProblemItr = aInput;
    }
    OrdinalType getMaxNumTrustRegionSubProblemItr() const
    {
        return (mMaxNumTrustRegionSubProblemItr);
    }


    void setInitialTrustRegionRadiusSetToNormProjectedGradient(const bool & aInput)
    {
        mIsInitialTrustRegionSetToNormProjectedGradient = aInput;
    }
    bool isInitialTrustRegionRadiusSetToNormProjectedGradient() const
    {
        return (mIsInitialTrustRegionSetToNormProjectedGradient);
    }

    virtual bool solveSubProblem(locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                                 locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                                 locus::SteihaugTointSolver<ScalarType, OrdinalType> & aSolver) = 0;

private:
    ScalarType mActualReduction;
    ScalarType mTrustRegionRadius;
    ScalarType mPredictedReduction;
    ScalarType mMinTrustRegionRadius;
    ScalarType mMaxTrustRegionRadius;
    ScalarType mTrustRegionExpansion;
    ScalarType mTrustRegionContraction;
    ScalarType mMinCosineAngleTolerance;
    ScalarType mGradientInexactnessTolerance;
    ScalarType mObjectiveInexactnessTolerance;

    ScalarType mActualOverPredictedReduction;
    ScalarType mActualOverPredictedReductionMidBound;
    ScalarType mActualOverPredictedReductionLowerBound;
    ScalarType mActualOverPredictedReductionUpperBound;

    ScalarType mGradientInexactnessToleranceConstant;
    ScalarType mObjectiveInexactnessToleranceConstant;

    OrdinalType mNumTrustRegionSubProblemItrDone;
    OrdinalType mMaxNumTrustRegionSubProblemItr;

    bool mIsInitialTrustRegionSetToNormProjectedGradient;

private:
    TrustRegionStepMng(const locus::TrustRegionStepMng<ScalarType, OrdinalType> & aRhs);
    locus::TrustRegionStepMng<ScalarType, OrdinalType> & operator=(const locus::TrustRegionStepMng<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_TRUSTREGIONSTEPMNG_HPP_ */
