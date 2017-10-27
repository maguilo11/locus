/*
 * Locus_SynthesisOptimizationSubProblem.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_SYNTHESISOPTIMIZATIONSUBPROBLEM_HPP_
#define LOCUS_SYNTHESISOPTIMIZATIONSUBPROBLEM_HPP_

#include <cmath>
#include <memory>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_OptimalityCriteriaDataMng.hpp"
#include "Locus_OptimalityCriteriaSubProblem.hpp"
#include "Locus_OptimalityCriteriaStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class SynthesisOptimizationSubProblem : public locus::OptimalityCriteriaSubProblem<ScalarType,OrdinalType>
{
public:
    explicit SynthesisOptimizationSubProblem(const locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng) :
            mMoveLimit(0.01),
            mDampingPower(0.5),
            mDualLowerBound(0),
            mDualUpperBound(1e4),
            mBisectionTolerance(1e-4),
            mInequalityGradientDotDeltaControl(0),
            mWorkControl(aDataMng.getCurrentControl().create())
    {
    }
    virtual ~SynthesisOptimizationSubProblem()
    {
    }

    ScalarType getMoveLimit() const
    {
        return (mMoveLimit);
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

                const locus::Vector<ScalarType, OrdinalType> & tInequalityValues = aDataMng.getCurrentConstraintValues();
                ScalarType mFirstOrderTaylorApproximation = tInequalityValues[tConstraintIndex] + mInequalityGradientDotDeltaControl;
                if(mFirstOrderTaylorApproximation > static_cast<ScalarType>(0.))
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
        ScalarType tMoveLimit = this->getMoveLimit();
        ScalarType tDampingPower = this->getDampingPower();

        OrdinalType tNumControlVectors = aDataMng.getNumControlVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tTrialControl = mWorkControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tPreviousControl = aDataMng.getPreviousControl(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tControlLowerBound = aDataMng.getControlLowerBounds(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tControlUpperBound = aDataMng.getControlUpperBounds(tVectorIndex);

            const locus::Vector<ScalarType, OrdinalType> & tObjectiveGradient = aDataMng.getObjectiveGradient(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tInequalityGradient = aDataMng.getInequalityGradient(tVectorIndex);

            OrdinalType tNumControls = tPreviousControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                if(tInequalityGradient[tControlIndex] == static_cast<ScalarType>(0))
                {
                    tTrialControl[tControlIndex] = tPreviousControl[tControlIndex];
                }
                else
                {
                    ScalarType tTrialControlValue = -tObjectiveGradient[tControlIndex]
                            / (aTrialDual * tInequalityGradient[tControlIndex]);
                    ScalarType tFabsValue = std::abs(tTrialControlValue);
                    ScalarType tSignValue = copysign(1.0, tTrialControlValue);
                    tTrialControlValue = tPreviousControl[tControlIndex] * tSignValue
                            * std::pow(tFabsValue, tDampingPower);
                    ScalarType tProposedControlValue = tPreviousControl[tControlIndex] + tMoveLimit;
                    tTrialControlValue = std::min(tProposedControlValue, tTrialControlValue);
                    tTrialControlValue = std::min(tControlUpperBound[tControlIndex], tTrialControlValue);
                    tProposedControlValue = tPreviousControl[tControlIndex] - tMoveLimit;
                    tTrialControlValue = std::max(tProposedControlValue, tTrialControlValue);
                    tTrialControlValue = std::max(tControlLowerBound[tControlIndex], tTrialControlValue);
                    tTrialControl[tControlIndex] = tTrialControlValue;
                }
            }
            aDataMng.setCurrentControl(tVectorIndex, tTrialControl);
            /*Compute Delta Control*/
            tTrialControl.update(static_cast<ScalarType>(-1), tPreviousControl, static_cast<ScalarType>(1));
            mInequalityGradientDotDeltaControl += tInequalityGradient.dot(tTrialControl);
        }
    }

private:
    ScalarType mMoveLimit;
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

}

#endif /* LOCUS_SYNTHESISOPTIMIZATIONSUBPROBLEM_HPP_ */
