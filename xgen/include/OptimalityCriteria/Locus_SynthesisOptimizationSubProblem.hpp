/*
 * locus_SynthesisOptimizationSubProblem.hpp
 *
 *  Created on: Oct 17, 2017
 */

#ifndef PLATO_SYNTHESISOPTIMIZATIONSUBPROBLEM_HPP_
#define PLATO_SYNTHESISOPTIMIZATIONSUBPROBLEM_HPP_

#include <cmath>
#include <memory>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_OptimalityCriteriaDataMng.hpp"
#include "Locus_OptimalityCriteriaSubProblem.hpp"
#include "Locus_OptimalityCriteriaStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class SynthesisOptimizationSubProblem : public locus::OptimalityCriteriaSubProblem<ScalarType,OrdinalType>
{
public:
    explicit SynthesisOptimizationSubProblem(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMoveLimit(0.2),
            mScaleFactor(0.01),
            mDampingPower(0.5),
            mDualLowerBound(0),
            mDualUpperBound(1e7),
            mBisectionTolerance(1e-4),
            mInequalityGradientDotDeltaControl(0),
            mWorkControl(aDataFactory.control().create())
    {
    }
    virtual ~SynthesisOptimizationSubProblem()
    {
    }

    ScalarType getMoveLimit() const
    {
        return (mMoveLimit);
    }
    ScalarType getScaleFactor() const
    {
        return (mScaleFactor);
    }
    ScalarType getDampingPower() const
    {
        return (mDampingPower);
    }
    ScalarType getDualLowerBound() const
    {
        return (mDualLowerBound);
    }
    ScalarType getDualUpperBound() const
    {
        return (mDualUpperBound);
    }
    ScalarType getBisectionTolerance() const
    {
        return (mBisectionTolerance);
    }

    void setMoveLimit(const ScalarType & aInput)
    {
        mMoveLimit = aInput;
    }
    void setScaleFactor(const ScalarType & aInput)
    {
        mScaleFactor = aInput;
    }
    void setDampingPower(const ScalarType & aInput)
    {
        mDampingPower = aInput;
    }
    void setDualLowerBound(const ScalarType & aInput)
    {
        mDualLowerBound = aInput;
    }
    void setDualUpperBound(const ScalarType & aInput)
    {
        mDualUpperBound = aInput;
    }
    void setBisectionTolerance(const ScalarType & aInput)
    {
        mBisectionTolerance = aInput;
    }

    void solve(locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng,
               locus::OptimalityCriteriaStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        ScalarType tDualLowerBound = this->getDualLowerBound();
        ScalarType tDualUpperBound = this->getDualUpperBound();
        ScalarType tBisectionTolerance = this->getBisectionTolerance();

        const OrdinalType tNumConstraints = aDataMng.getNumConstraints();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = aDataMng.getCurrentControl();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            ScalarType tDualMisfit = tDualUpperBound - tDualLowerBound;
            ScalarType tTrialDual = std::numeric_limits<ScalarType>::max();
            while(tDualMisfit >= tBisectionTolerance)
            {
                tTrialDual = static_cast<ScalarType>(0.5) * (tDualUpperBound + tDualLowerBound);
                this->updateControl(tTrialDual, aDataMng);

                const ScalarType tMyCurrentConstraintValue = aDataMng.getCurrentConstraintValues(tConstraintIndex);
                ScalarType tFirstOrderTaylorApproximation = tMyCurrentConstraintValue + mInequalityGradientDotDeltaControl;
                if(tFirstOrderTaylorApproximation > static_cast<ScalarType>(0))
                {
                    tDualLowerBound = tTrialDual;
                }
                else
                {
                    tDualUpperBound = tTrialDual;
                }
                tDualMisfit = tDualUpperBound - tDualLowerBound;
            }
            aDataMng.setCurrentDual(tConstraintIndex, tTrialDual);
        }
    }

private:
    void updateControl(const ScalarType & aTrialDual, locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mInequalityGradientDotDeltaControl = 0;
        const ScalarType tMoveLimit = this->getMoveLimit();
        const ScalarType tDampingPower = this->getDampingPower();

        const OrdinalType tNumControlVectors = aDataMng.getNumControlVectors();
        for(OrdinalType tMyVectorIndex = 0; tMyVectorIndex < tNumControlVectors; tMyVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyTrialControls = mWorkControl->operator[](tMyVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControls = aDataMng.getPreviousControl(tMyVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyControlLowerBounds = aDataMng.getControlLowerBounds(tMyVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyControlUpperBounds = aDataMng.getControlUpperBounds(tMyVectorIndex);

            const locus::Vector<ScalarType, OrdinalType> & tMyObjectiveGradient = aDataMng.getObjectiveGradient(tMyVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyInequalityGradient = aDataMng.getInequalityGradient(tMyVectorIndex);

            OrdinalType tMyNumControls = tMyPreviousControls.size();
            for(OrdinalType tMyControlIndex = 0; tMyControlIndex < tMyNumControls; tMyControlIndex++)
            {
                if(tMyInequalityGradient[tMyControlIndex] == static_cast<ScalarType>(0))
                {
                    tMyTrialControls[tMyControlIndex] = tMyPreviousControls[tMyControlIndex];
                }
                else
                {
                    ScalarType tMyDesignVariableOffset = (mScaleFactor
                            * (tMyControlUpperBounds[tMyControlIndex] - tMyControlLowerBounds[tMyControlIndex]))
                            - tMyControlLowerBounds[tMyControlIndex];
                    ScalarType tMyValue = -tMyObjectiveGradient[tMyControlIndex]
                            / (aTrialDual * tMyInequalityGradient[tMyControlIndex]);
                    ScalarType tFabsValue = std::abs(tMyValue);
                    ScalarType tSignValue = copysign(1.0, tMyValue);
                    ScalarType tMyTrialControlValue = ((tMyPreviousControls[tMyControlIndex] + tMyDesignVariableOffset)
                            * tSignValue * std::pow(tFabsValue, tDampingPower)) - tMyDesignVariableOffset;

                    ScalarType tMyControlValue = tMyPreviousControls[tMyControlIndex] + tMoveLimit;
                    tMyTrialControlValue = std::min(tMyControlValue, tMyTrialControlValue);
                    tMyControlValue = tMyPreviousControls[tMyControlIndex] - tMoveLimit;
                    tMyTrialControlValue = std::max(tMyControlValue, tMyTrialControlValue);
                    tMyTrialControlValue = std::min(tMyControlUpperBounds[tMyControlIndex], tMyTrialControlValue);
                    tMyTrialControlValue = std::max(tMyControlLowerBounds[tMyControlIndex], tMyTrialControlValue);
                    tMyTrialControls[tMyControlIndex] = tMyTrialControlValue;
                }
            }
            aDataMng.setCurrentControl(tMyVectorIndex, tMyTrialControls);
            /*Compute Delta Control*/
            tMyTrialControls.update(static_cast<ScalarType>(-1), tMyPreviousControls, static_cast<ScalarType>(1));
            mInequalityGradientDotDeltaControl += tMyInequalityGradient.dot(tMyTrialControls);
        }
    }

private:
    ScalarType mMoveLimit;
    ScalarType mScaleFactor;
    ScalarType mDampingPower;
    ScalarType mDualLowerBound;
    ScalarType mDualUpperBound;
    ScalarType mBisectionTolerance;
    ScalarType mInequalityGradientDotDeltaControl;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mWorkControl;

private:
    SynthesisOptimizationSubProblem(const locus::SynthesisOptimizationSubProblem<ScalarType, OrdinalType>&);
    locus::SynthesisOptimizationSubProblem<ScalarType, OrdinalType> & operator=(const locus::SynthesisOptimizationSubProblem<ScalarType, OrdinalType>&);
};

} // namespace locus

#endif /* PLATO_SYNTHESISOPTIMIZATIONSUBPROBLEM_HPP_ */
