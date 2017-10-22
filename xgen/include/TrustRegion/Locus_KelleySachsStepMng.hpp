/*
 * Locus_KelleySachsStepMng.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_KELLEYSACHSSTEPMNG_HPP_
#define LOCUS_KELLEYSACHSSTEPMNG_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <cassert>

#include "Locus_Bounds.hpp"
#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_TrustRegionStepMng.hpp"
#include "Locus_SteihaugTointSolver.hpp"
#include "Locus_TrustRegionAlgorithmDataMng.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class KelleySachsStepMng : public locus::TrustRegionStepMng<ScalarType, OrdinalType>
{
public:
    explicit KelleySachsStepMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            locus::TrustRegionStepMng<ScalarType, OrdinalType>(),
            mEta(0),
            mEpsilon(0),
            mNormInactiveGradient(0),
            mStationarityMeasureConstant(std::numeric_limits<ScalarType>::min()),
            mMidPointObjectiveFunction(0),
            mTrustRegionRadiusFlag(false),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mMidControls(aDataFactory.control().create()),
            mWorkMultiVec(aDataFactory.control().create()),
            mLowerBoundLimit(aDataFactory.control().create()),
            mUpperBoundLimit(aDataFactory.control().create()),
            mInactiveGradient(aDataFactory.control().create()),
            mMatrixTimesVector(aDataFactory.control().create()),
            mProjectedTrialStep(aDataFactory.control().create()),
            mProjectedCauchyStep(aDataFactory.control().create()),
            mActiveProjectedTrialStep(aDataFactory.control().create()),
            mInactiveProjectedTrialStep(aDataFactory.control().create())
    {
        // NOTE: INITIALIZE WORK VECTOR
    }
    virtual ~KelleySachsStepMng()
    {
    }

    //! Returns adaptive constants eta, which ensures superlinear convergence
    ScalarType getEtaConstant() const
    {
        return (mEta);
    }
    //! Sets adaptive constants eta, which ensures superlinear convergence
    void setEtaConstant(const ScalarType & aInput)
    {
        mEta = aInput;
    }
    //! Returns adaptive constants epsilon, which ensures superlinear convergence
    ScalarType getEpsilonConstant() const
    {
        return (mEpsilon);
    }
    //! Sets adaptive constants epsilon, which ensures superlinear convergence
    void setEpsilonConstant(const ScalarType &  aInput)
    {
        mEpsilon = aInput;
    }
    void setStationarityMeasureConstant(const ScalarType &  aInput)
    {
        mStationarityMeasureConstant = aInput;
    }
    //! Sets objective function value computed with the control values at the mid-point
    void setMidPointObjectiveFunctionValue(const ScalarType & aInput)
    {
        mMidPointObjectiveFunction = aInput;
    }
    //! Returns objective function value computed with the control values at the mid-point
    ScalarType getMidPointObjectiveFunctionValue() const
    {
        return (mMidPointObjectiveFunction);
    }
    //! Sets control values at the mid-point
    void setMidPointControls(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mMidControls.get() != nullptr);
        assert(aInput.getNumVectors() == mMidControls->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mMidControls);
    }
    //! Returns control values at the mid-point
    const locus::MultiVector<ScalarType, OrdinalType> & getMidPointControls() const
    {
        return (mMidControls.operator*());
    }

    bool solveSubProblem(locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                         locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                         locus::SteihaugTointSolver<ScalarType, OrdinalType> & aSolver)
    {
        mTrustRegionRadiusFlag = false;
        bool tTrialControlAccepted = true;
        this->setNumTrustRegionSubProblemItrDone(1);
        ScalarType  tMinTrustRegionRadius = this->getMinTrustRegionRadius();
        if(this->getTrustRegionRadius() < tMinTrustRegionRadius)
        {
            this->setTrustRegionRadius(tMinTrustRegionRadius);
        }

        OrdinalType tMaxNumSubProblemItr = this->getMaxNumTrustRegionSubProblemItr();
        while(this->getNumTrustRegionSubProblemItrDone() <= tMaxNumSubProblemItr)
        {
            // Compute active and inactive sets
            this->computeActiveAndInactiveSet(aDataMng);
            // Set solver tolerance
            this->setSolverTolerance(aDataMng, aSolver);
            // Compute descent direction
            ScalarType tTrustRegionRadius = this->getTrustRegionRadius();
            aSolver.setTrustRegionRadius(tTrustRegionRadius);
            aSolver.solve(aStageMng, aDataMng);
            // Compute projected trial step
            this->computeProjectedTrialStep(aDataMng);
            // Apply projected trial step to Hessian operator
            this->applyProjectedTrialStepToHessian(aDataMng, aStageMng);
            // Compute predicted reduction based on mid trial control
            ScalarType tPredictedReduction = this->computePredictedReduction(aDataMng);

            if(aDataMng.isObjectiveInexactnessToleranceExceeded() == true)
            {
                tTrialControlAccepted = false;
                break;
            }

            // Update objective function inexactness tolerance (bound)
            this->updateObjectiveInexactnessTolerance(tPredictedReduction);
            // Evaluate current mid objective function
            ScalarType tTolerance = this->getObjectiveInexactnessTolerance();
            mMidPointObjectiveFunction = aStageMng.evaluateObjective(*mMidControls, tTolerance);
            // Compute actual reduction based on mid trial control
            ScalarType tCurrentObjectiveFunctionValue = aDataMng.getCurrentObjectiveFunctionValue();
            ScalarType tActualReduction = mMidPointObjectiveFunction - tCurrentObjectiveFunctionValue;
            this->setActualReduction(tActualReduction);
            // Compute actual over predicted reduction ratio
            ScalarType tActualOverPredReduction = tActualReduction /
                    (tPredictedReduction + std::numeric_limits<ScalarType>::epsilon());
            this->setActualOverPredictedReduction(tActualOverPredReduction);
            // Update trust region radius: io_->printTrustRegionSubProblemDiagnostics(aDataMng, aSolver, this);
            if(this->updateTrustRegionRadius(aDataMng) == true)
            {
                break;
            }
            this->updateNumTrustRegionSubProblemItrDone();
        }
        return (tTrialControlAccepted);
    }

private:
    void setSolverTolerance(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                            locus::SteihaugTointSolver<ScalarType, OrdinalType> & aSolver)
    {
        ScalarType tCummulativeDotProduct = 0;
        const OrdinalType tNumVectors = aDataMng.getNumControlVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = aDataMng.getCurrentGradient(tVectorIndex);
            locus::Vector<ScalarType, OrdinalType> & tMyInactiveGradient = (*mInactiveGradient)[tVectorIndex];

            tMyInactiveGradient.update(static_cast<ScalarType>(1), tMyCurrentGradient, static_cast<ScalarType>(0));
            tMyInactiveGradient.entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += tMyInactiveGradient.dot(tMyInactiveGradient);
        }
        mNormInactiveGradient = std::sqrt(tCummulativeDotProduct);
        ScalarType tSolverStoppingTolerance = this->getEtaConstant() * mNormInactiveGradient;
        aSolver.setSolverTolerance(tSolverStoppingTolerance);
    }
    void computeProjectedTrialStep(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        // Project trial control
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mMidControls);
        const locus::MultiVector<ScalarType, OrdinalType> & tTrialStep = aDataMng.getTrialStep();
        locus::update(static_cast<ScalarType>(1), tTrialStep, static_cast<ScalarType>(1), *mMidControls);
        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, *mMidControls);

        // Compute projected trial step
        locus::update(static_cast<ScalarType>(1), *mMidControls, static_cast<ScalarType>(0), *mProjectedTrialStep);
        locus::update(static_cast<ScalarType>(-1), tCurrentControl, static_cast<ScalarType>(1), *mProjectedTrialStep);
    }
    ScalarType computePredictedReduction(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tProjTrialStepDotInactiveGradient = 0;
        ScalarType tProjTrialStepDotHessTimesProjTrialStep = 0;
        const OrdinalType tNumVectors = aDataMng.getNumControlVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveGradient = mInactiveGradient->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyMatrixTimesVector = mMatrixTimesVector->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyProjectedTrialStep = mProjectedTrialStep->operator[](tVectorIndex);

            tProjTrialStepDotInactiveGradient += tMyProjectedTrialStep.dot(tMyInactiveGradient);
            tProjTrialStepDotHessTimesProjTrialStep += tMyProjectedTrialStep.dot(tMyMatrixTimesVector);
        }

        ScalarType tPredictedReduction = tProjTrialStepDotInactiveGradient
                + (static_cast<ScalarType>(0.5) * tProjTrialStepDotHessTimesProjTrialStep);
        this->setPredictedReduction(tPredictedReduction);

        return (tPredictedReduction);
    }
    bool updateTrustRegionRadius(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tActualReduction = this->getActualReduction();
        ScalarType tCurrentTrustRegionRadius = this->getTrustRegionRadius();
        ScalarType tActualOverPredReduction = this->getActualOverPredictedReduction();
        ScalarType tActualOverPredMidBound = this->getActualOverPredictedReductionMidBound();
        ScalarType tActualOverPredLowerBound = this->getActualOverPredictedReductionLowerBound();
        ScalarType tActualOverPredUpperBound = this->getActualOverPredictedReductionUpperBound();

        bool tStopTrustRegionSubProblem = false;
        ScalarType tActualReductionLowerBound = this->computeActualReductionLowerBound(aDataMng);
        if(tActualReduction >= tActualReductionLowerBound)
        {
            tCurrentTrustRegionRadius = this->getTrustRegionContraction()
                    * tCurrentTrustRegionRadius;
            mTrustRegionRadiusFlag = true;
        }
        else if(tActualOverPredReduction < tActualOverPredLowerBound)
        {
            tCurrentTrustRegionRadius = this->getTrustRegionContraction()
                    * tCurrentTrustRegionRadius;
            mTrustRegionRadiusFlag = true;
        }
        else if(tActualOverPredReduction >= tActualOverPredLowerBound && tActualOverPredReduction < tActualOverPredMidBound)
        {
            tStopTrustRegionSubProblem = true;
        }
        else if(tActualOverPredReduction >= tActualOverPredMidBound && tActualOverPredReduction < tActualOverPredUpperBound)
        {
            tCurrentTrustRegionRadius = this->getTrustRegionExpansion()
                    * tCurrentTrustRegionRadius;
            tStopTrustRegionSubProblem = true;
        }
        else if(tActualOverPredReduction > tActualOverPredUpperBound && mTrustRegionRadiusFlag == true)
        {
            tCurrentTrustRegionRadius =
                    static_cast<ScalarType>(2) * this->getTrustRegionExpansion() * tCurrentTrustRegionRadius;
            tStopTrustRegionSubProblem = true;
        }
        else
        {
            ScalarType tMaxTrustRegionRadius = this->getMaxTrustRegionRadius();
            tCurrentTrustRegionRadius = this->getTrustRegionExpansion() * tCurrentTrustRegionRadius;
            tCurrentTrustRegionRadius = std::min(tMaxTrustRegionRadius, tCurrentTrustRegionRadius);
        }
        this->setTrustRegionRadius(tCurrentTrustRegionRadius);

        return (tStopTrustRegionSubProblem);
    }

    void applyProjectedTrialStepToHessian(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                                          locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        // Compute active projected trial step
        locus::fill(static_cast<ScalarType>(0), *mMatrixTimesVector);
        locus::update(static_cast<ScalarType>(1),
                      *mProjectedTrialStep,
                      static_cast<ScalarType>(0),
                      *mActiveProjectedTrialStep);
        const locus::MultiVector<ScalarType, OrdinalType> & tActiveSet = aDataMng.getActiveSet();
        locus::entryWiseProduct(tActiveSet, *mActiveProjectedTrialStep);

        // Compute inactive projected trial step
        locus::update(static_cast<ScalarType>(1),
                      *mProjectedTrialStep,
                      static_cast<ScalarType>(0),
                      *mInactiveProjectedTrialStep);
        const locus::MultiVector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet();
        locus::entryWiseProduct(tInactiveSet, *mInactiveProjectedTrialStep);

        // Apply inactive projected trial step to Hessian
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToHessian(tCurrentControl, *mInactiveProjectedTrialStep, *mMatrixTimesVector);

        // Compute Hessian times projected trial step, i.e. ( ActiveSet + (InactiveSet' * Hess * InactiveSet) ) * Vector
        locus::entryWiseProduct(tInactiveSet, *mMatrixTimesVector);
        locus::update(static_cast<ScalarType>(1),
                      *mActiveProjectedTrialStep,
                      static_cast<ScalarType>(1),
                      *mMatrixTimesVector);
    }

    ScalarType computeActualReductionLowerBound(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tConditionOne = this->getTrustRegionRadius()
                / (mNormInactiveGradient + std::numeric_limits<ScalarType>::epsilon());
        ScalarType tLambda = std::min(tConditionOne, static_cast<ScalarType>(1.));

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mWorkMultiVec);
        locus::update(-tLambda, *mInactiveGradient, static_cast<ScalarType>(1), *mWorkMultiVec);

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, *mWorkMultiVec);

        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mProjectedCauchyStep);
        locus::update(static_cast<ScalarType>(-1), *mWorkMultiVec, static_cast<ScalarType>(1), *mProjectedCauchyStep);

        const ScalarType tSLOPE_CONSTANT = 1e-4;
        ScalarType tNormProjectedCauchyStep = locus::norm(*mProjectedCauchyStep);
        ScalarType tLowerBound = -mStationarityMeasureConstant * tSLOPE_CONSTANT * tNormProjectedCauchyStep;

        return (tLowerBound);
    }

    ScalarType computeLambdaScaleFactor(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        const locus::MultiVector<ScalarType, OrdinalType> & tGradient = aDataMng.getCurrentGradient();
        locus::update(static_cast<ScalarType>(1), tGradient, static_cast<ScalarType>(0), *mWorkMultiVec);
        const locus::MultiVector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet();
        locus::entryWiseProduct(tInactiveSet, *mWorkMultiVec);
        ScalarType tNormCurrentProjectedGradient = locus::norm(*mWorkMultiVec);

        ScalarType tCondition = 0;
        const ScalarType tCurrentTrustRegionRadius = this->getTrustRegionRadius();
        if(tNormCurrentProjectedGradient > 0)
        {
            tCondition = tCurrentTrustRegionRadius / tNormCurrentProjectedGradient;
        }
        else
        {
            ScalarType tNormProjectedGradient = aDataMng.getNormProjectedGradient();
            tCondition = tCurrentTrustRegionRadius / tNormProjectedGradient;
        }
        ScalarType tLambda = std::min(tCondition, static_cast<ScalarType>(1.));

        return (tLambda);
    }

    void computeActiveAndInactiveSet(locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        const locus::MultiVector<ScalarType, OrdinalType> & tMyLowerBound = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tMyUpperBound = aDataMng.getControlUpperBounds();
        // Compute current lower bound limit
        locus::fill(mEpsilon, *mWorkMultiVec);
        locus::update(static_cast<ScalarType>(1), tMyLowerBound, static_cast<ScalarType>(0), *mLowerBoundLimit);
        locus::update(static_cast<ScalarType>(-1), *mWorkMultiVec, static_cast<ScalarType>(1), *mLowerBoundLimit);
        // Compute current upper bound limit
        locus::update(static_cast<ScalarType>(1), tMyUpperBound, static_cast<ScalarType>(0), *mUpperBoundLimit);
        locus::update(static_cast<ScalarType>(1), *mWorkMultiVec, static_cast<ScalarType>(1), *mUpperBoundLimit);

        // Compute active and inactive sets
        const locus::MultiVector<ScalarType, OrdinalType> & tMyCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tMyCurrentControl, static_cast<ScalarType>(0), *mWorkMultiVec);

        ScalarType tLambda = this->computeLambdaScaleFactor(aDataMng);
        const locus::MultiVector<ScalarType, OrdinalType> & tMyGradient = aDataMng.getCurrentGradient();
        locus::update(-tLambda, tMyGradient, static_cast<ScalarType>(1), *mWorkMultiVec);
        locus::bounds::computeActiveAndInactiveSets(*mWorkMultiVec,
                                                    *mLowerBoundLimit,
                                                    *mUpperBoundLimit,
                                                    *mActiveSet,
                                                    *mInactiveSet);
        aDataMng.setActiveSet(*mActiveSet);
        aDataMng.setInactiveSet(*mInactiveSet);
    }

private:
    ScalarType mEta;
    ScalarType mEpsilon;
    ScalarType mNormInactiveGradient;
    ScalarType mStationarityMeasureConstant;
    ScalarType mMidPointObjectiveFunction;

    bool mTrustRegionRadiusFlag;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mMidControls;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mWorkMultiVec;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mLowerBoundLimit;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mUpperBoundLimit;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mMatrixTimesVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mProjectedTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mProjectedCauchyStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveProjectedTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveProjectedTrialStep;

private:
    KelleySachsStepMng(const locus::KelleySachsStepMng<ScalarType, OrdinalType> & aRhs);
    locus::KelleySachsStepMng<ScalarType, OrdinalType> & operator=(const locus::KelleySachsStepMng<ScalarType, OrdinalType> & aRhs);
};

}

#endif /* LOCUS_KELLEYSACHSSTEPMNG_HPP_ */
