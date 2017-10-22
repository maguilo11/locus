/*
 * Locus_KelleySachsAlgorithm.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_KELLEYSACHSALGORITHM_HPP_
#define LOCUS_KELLEYSACHSALGORITHM_HPP_

#include <memory>

#include "Locus_Types.hpp"
#include "Locus_Bounds.hpp"
#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_KelleySachsStepMng.hpp"
#include "Locus_TrustRegionStageMng.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class KelleySachsAlgorithm
{
public:
    explicit KelleySachsAlgorithm(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumUpdates(10),
            mMaxNumOuterIterations(100),
            mNumOuterIterationsDone(0),
            mGradientTolerance(1e-8),
            mTrialStepTolerance(1e-8),
            mObjectiveTolerance(1e-8),
            mStagnationTolerance(1e-8),
            mStationarityMeasure(0.),
            mActualReductionTolerance(1e-10),
            mStoppingCriterion(locus::algorithm::NOT_CONVERGED),
            mControlWorkVector(aDataFactory.control().create())
    {
    }
    virtual ~KelleySachsAlgorithm()
    {
    }

    void setGradientTolerance(const ScalarType & aInput)
    {
        mGradientTolerance = aInput;
    }
    void setTrialStepTolerance(const ScalarType & aInput)
    {
        mTrialStepTolerance = aInput;
    }
    void setObjectiveTolerance(const ScalarType & aInput)
    {
        mObjectiveTolerance = aInput;
    }
    void setStagnationTolerance(const ScalarType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    void setActualReductionTolerance(const ScalarType & aInput)
    {
        mActualReductionTolerance = aInput;
    }

    void setMaxNumUpdates(const OrdinalType & aInput)
    {
        mMaxNumUpdates = aInput;
    }
    void setNumIterationsDone(const OrdinalType & aInput)
    {
        mNumOuterIterationsDone = aInput;
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumOuterIterations = aInput;
    }
    void setStoppingCriterion(const locus::algorithm::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }
    ScalarType getGradientTolerance() const
    {
        return (mGradientTolerance);
    }
    ScalarType getTrialStepTolerance() const
    {
        return (mTrialStepTolerance);
    }
    ScalarType getObjectiveTolerance() const
    {
        return (mObjectiveTolerance);
    }
    ScalarType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    ScalarType getActualReductionTolerance() const
    {
        return (mActualReductionTolerance);
    }

    OrdinalType getMaxNumUpdates() const
    {
        return (mMaxNumUpdates);
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumOuterIterationsDone);
    }
    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumOuterIterations);
    }
    locus::algorithm::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }

    bool updateControl(const locus::MultiVector<ScalarType, OrdinalType> & aMidGradient,
                       locus::KelleySachsStepMng<ScalarType, OrdinalType> & aStepMng,
                       locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                       locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        bool tControlUpdated = false;

        ScalarType tXi = 1.;
        ScalarType tBeta = 1e-2;
        ScalarType tAlpha = tBeta;
        ScalarType tMu = static_cast<ScalarType>(1) - static_cast<ScalarType>(1e-4);

        ScalarType tMidActualReduction = aStepMng.getActualReduction();
        ScalarType tMidObjectiveValue = aStepMng.getMidPointObjectiveFunctionValue();
        const locus::MultiVector<ScalarType, OrdinalType> & tMidControl = aStepMng.getMidPointControls();
        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();

        OrdinalType tIteration = 0;
        while(tIteration < mMaxNumUpdates)
        {
            // Compute trial point based on the mid gradient (i.e. mid steepest descent)
            ScalarType tLambda = -tXi / tAlpha;
            locus::update(static_cast<ScalarType>(1), tMidControl, static_cast<ScalarType>(0), *mControlWorkVector);
            locus::update(tLambda, aMidGradient, static_cast<ScalarType>(1), *mControlWorkVector);
            locus::bounds::project(tLowerBounds, tUpperBounds, *mControlWorkVector);

            // Compute trial objective function
            ScalarType tTolerance = aStepMng.getObjectiveInexactnessTolerance();
            ScalarType tTrialObjectiveValue = aStageMng.evaluateObjective(*mControlWorkVector, tTolerance);
            // Compute actual reduction
            ScalarType tTrialActualReduction = tTrialObjectiveValue - tMidObjectiveValue;
            // Check convergence
            if(tTrialActualReduction < -tMu * tMidActualReduction)
            {
                tControlUpdated = true;
                aDataMng.setCurrentControl(*mControlWorkVector);
                aStepMng.setActualReduction(tTrialActualReduction);
                aDataMng.setCurrentObjectiveFunctionValue(tTrialObjectiveValue);
                break;
            }
            // Compute scaling for next iteration
            if(tIteration == 1)
            {
                tXi = tAlpha;
            }
            else
            {
                tXi = tXi * tBeta;
            }
            tIteration++;
        }

        if(tIteration >= mMaxNumUpdates)
        {
            aDataMng.setCurrentControl(tMidControl);
            aDataMng.setCurrentObjectiveFunctionValue(tMidObjectiveValue);
        }

        return (tControlUpdated);
    }

    virtual void solve() = 0;

private:
    OrdinalType mMaxNumUpdates;
    OrdinalType mMaxNumOuterIterations;
    OrdinalType mNumOuterIterationsDone;

    ScalarType mGradientTolerance;
    ScalarType mTrialStepTolerance;
    ScalarType mObjectiveTolerance;
    ScalarType mStagnationTolerance;
    ScalarType mStationarityMeasure;
    ScalarType mActualReductionTolerance;

    locus::algorithm::stop_t mStoppingCriterion;

    std::shared_ptr<locus::MultiVector<ScalarType,OrdinalType>> mControlWorkVector;

private:
    KelleySachsAlgorithm(const locus::KelleySachsAlgorithm<ScalarType, OrdinalType> & aRhs);
    locus::KelleySachsAlgorithm<ScalarType, OrdinalType> & operator=(const locus::KelleySachsAlgorithm<ScalarType, OrdinalType> & aRhs);
};
}

#endif /* LOCUS_KELLEYSACHSALGORITHM_HPP_ */
