/*
 * Locus_AugmentedLagrangianStageMng.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_AUGMENTEDLAGRANGIANSTAGEMNG_HPP_
#define LOCUS_AUGMENTEDLAGRANGIANSTAGEMNG_HPP_

#include <limits>
#include <vector>
#include <memory>

#include "Locus_Vector.hpp"
#include "Locus_StateData.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_LinearOperator.hpp"
#include "Locus_Preconditioner.hpp"
#include "Locus_MultiVectorList.hpp"
#include "Locus_GradientOperator.hpp"
#include "Locus_LinearOperatorList.hpp"
#include "Locus_ReductionOperations.hpp"
#include "Locus_TrustRegionStageMng.hpp"
#include "Locus_GradientOperatorList.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class AugmentedLagrangianStageMng : public locus::TrustRegionStageMng<ScalarType, OrdinalType>
{
public:
    AugmentedLagrangianStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory,
                                const locus::Criterion<ScalarType, OrdinalType> & aObjective,
                                const locus::CriterionList<ScalarType, OrdinalType> & aConstraints) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumObjectiveHessianEvaluations(0),
            mMinPenaltyValue(1e-10),
            mPenaltyScaleFactor(0.2),
            mCurrentFeasibilityMeasure(std::numeric_limits<ScalarType>::max()),
            mCurrentLagrangeMultipliersPenalty(1),
            mNormObjectiveFunctionGradient(std::numeric_limits<ScalarType>::max()),
            mNumConstraintEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mNumConstraintGradientEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mNumConstraintHessianEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory)),
            mState(aDataFactory.state().create()),
            mDualWorkVec(aDataFactory.dual().create()),
            mControlWorkVec(aDataFactory.control().create()),
            mObjectiveGradient(aDataFactory.control().create()),
            mLagrangeMultipliers(aDataFactory.dual().create()),
            mWorkConstraintValues(aDataFactory.dual().create()),
            mCurrentConstraintValues(aDataFactory.dual().create()),
            mCostraintGradients(std::make_shared<locus::MultiVectorList<ScalarType, OrdinalType>>()),
            mObjective(aObjective.create()),
            mConstraints(aConstraints.create()),
            mPreconditioner(std::make_shared<locus::IdentityPreconditioner<ScalarType, OrdinalType>>()),
            mObjectiveHessian(nullptr),
            mConstraintHessians(nullptr),
            mDualReductionOperations(aDataFactory.getDualReductionOperations().create()),
            mObjectiveGradientOperator(nullptr),
            mConstraintGradientOperators(nullptr)
    {
        this->initialize();
    }
    virtual ~AugmentedLagrangianStageMng()
    {
    }

    OrdinalType getNumObjectiveFunctionEvaluations() const
    {
        return (mNumObjectiveFunctionEvaluations);
    }
    OrdinalType getNumObjectiveGradientEvaluations() const
    {
        return (mNumObjectiveGradientEvaluations);
    }
    OrdinalType getNumObjectiveHessianEvaluations() const
    {
        return (mNumObjectiveHessianEvaluations);
    }
    OrdinalType getNumConstraintEvaluations(const OrdinalType & aIndex) const
    {
        assert(mNumConstraintEvaluations.empty() == false);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mNumConstraintEvaluations.size());
        return (mNumConstraintEvaluations[aIndex]);
    }
    OrdinalType getNumConstraintGradientEvaluations(const OrdinalType & aIndex) const
    {
        assert(mNumConstraintGradientEvaluations.empty() == false);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mNumConstraintGradientEvaluations.size());
        return (mNumConstraintGradientEvaluations[aIndex]);
    }
    OrdinalType getNumConstraintHessianEvaluations(const OrdinalType & aIndex) const
    {
        assert(mNumConstraintHessianEvaluations.empty() == false);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mNumConstraintHessianEvaluations.size());
        return (mNumConstraintHessianEvaluations[aIndex]);
    }

    ScalarType getCurrentLagrangeMultipliersPenalty() const
    {
        return (mCurrentLagrangeMultipliersPenalty);
    }
    ScalarType getNormObjectiveFunctionGradient() const
    {
        return (mNormObjectiveFunctionGradient);
    }

    void getLagrangeMultipliers(locus::MultiVector<ScalarType, OrdinalType> & aInput) const
    {
        locus::update(1., *mLagrangeMultipliers, 0., aInput);
    }
    void getCurrentConstraintValues(locus::MultiVector<ScalarType, OrdinalType> & aInput) const
    {
        locus::update(1., *mCurrentConstraintValues, 0., aInput);
    }

    void setObjectiveGradient(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mObjectiveGradientOperator = aInput.create();
    }
    void setConstraintGradients(const locus::GradientOperatorList<ScalarType, OrdinalType> & aInput)
    {
        mConstraintGradientOperators = aInput.create();
    }
    void setObjectiveHessian(const locus::LinearOperator<ScalarType, OrdinalType> & aInput)
    {
        mObjectiveHessian = aInput.create();
    }
    void setConstraintHessians(const locus::LinearOperatorList<ScalarType, OrdinalType> & aInput)
    {
        mConstraintHessians = aInput.create();
    }
    void setPreconditioner(const locus::Preconditioner<ScalarType, OrdinalType> & aInput)
    {
        mPreconditioner = aInput.create();
    }

    void update(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mStateData->setCurrentTrialStep(aDataMng.getTrialStep());
        mStateData->setCurrentControl(aDataMng.getCurrentControl());
        mStateData->setCurrentObjectiveGradient(aDataMng.getCurrentGradient());
        mStateData->setCurrentObjectiveFunctionValue(aDataMng.getCurrentObjectiveFunctionValue());

        mObjectiveGradientOperator->update(mStateData.operator*());
        mObjectiveHessian->update(mStateData.operator*());
        mPreconditioner->update(mStateData.operator*());

        const OrdinalType tNumConstraints = mConstraintGradientOperators->size();
        assert(tNumConstraints == mCostraintGradients->size());
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintGradient =
                    mCostraintGradients->operator[](tConstraintIndex);
            mStateData->setCurrentConstraintGradient(tMyConstraintGradient);
            mConstraintGradientOperators->operator[](tConstraintIndex).update(mStateData.operator*());
            mConstraintHessians->operator[](tConstraintIndex).update(mStateData.operator*());
        }
    }
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                  ScalarType aTolerance = std::numeric_limits<ScalarType>::max())
    {
        // Evaluate objective function, f(\mathbf{z})
        ScalarType tObjectiveValue = mObjective->value(*mState, aControl);
        this->increaseObjectiveFunctionEvaluationCounter();

        // Evaluate inequality constraints, h(\mathbf{u}(\mathbf{z}),\mathbf{z})
        this->evaluateConstraint(aControl);

        // Evaluate Lagrangian functional, \ell(\mathbf{u}(\mathbf{z}),\mathbf{z},\mu) =
        //   f(\mathbf{u}(\mathbf{z}),\mathbf{z}) + \mu^{T}h(\mathbf{u}(\mathbf{z}),\mathbf{z})
        ScalarType tLagrangeMultipliersDotInequalityValue =
                locus::dot(*mLagrangeMultipliers, *mWorkConstraintValues);
        ScalarType tLagrangianValue = tObjectiveValue + tLagrangeMultipliersDotInequalityValue;

        // Evaluate augmented Lagrangian functional, \mathcal{L}(\mathbf{z}),\mathbf{z},\mu) =
        //   \ell(\mathbf{u}(\mathbf{z}),\mathbf{z},\mu) +
        //   \frac{1}{2\beta}(h(\mathbf{u}(\mathbf{z}),\mathbf{z})^{T}h(\mathbf{u}(\mathbf{z}),\mathbf{z})),
        //   where \beta\in\mathbb{R} denotes a penalty parameter
        ScalarType tInequalityValueDotInequalityValue =
                locus::dot(*mWorkConstraintValues, *mWorkConstraintValues);
        ScalarType tAugmentedLagrangianValue = tLagrangianValue
                + ((static_cast<ScalarType>(0.5) / mCurrentLagrangeMultipliersPenalty) * tInequalityValueDotInequalityValue);

        return (tAugmentedLagrangianValue);
    }

    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mObjectiveGradientOperator.get() != nullptr);

        locus::fill(static_cast<ScalarType>(0), aOutput);
        // Compute objective function gradient: \frac{\partial f}{\partial\mathbf{z}}
        locus::fill(static_cast<ScalarType>(0), *mObjectiveGradient);
        mObjectiveGradientOperator->compute(*mState, aControl, *mObjectiveGradient);
        mNormObjectiveFunctionGradient = locus::norm(*mObjectiveGradient);
        this->increaseObjectiveGradientEvaluationCounter();

        // Compute inequality constraint gradient: \frac{\partial h_i}{\partial\mathbf{z}}
        const ScalarType tOneOverPenalty = static_cast<ScalarType>(1.) / mCurrentLagrangeMultipliersPenalty;
        const OrdinalType tNumConstraintVectors = mCurrentConstraintValues->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
        {
            const OrdinalType tNumConstraints = (*mCurrentConstraintValues)[tVectorIndex].size();
            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                assert(mConstraintGradientOperators.get() != nullptr);
                assert(mConstraintGradientOperators->ptr(tConstraintIndex).get() != nullptr);

                // Add contribution from: \lambda_i\frac{\partial h_i}{\partial\mathbf{z}} to Lagrangian gradient
                locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintGradient = mCostraintGradients->operator[](tVectorIndex);
                locus::fill(static_cast<ScalarType>(0), tMyConstraintGradient);
                mConstraintGradientOperators->operator[](tConstraintIndex).compute(mState.operator*(), aControl, tMyConstraintGradient);
                this->increaseConstraintGradientEvaluationCounter(tConstraintIndex);
                locus::update((*mLagrangeMultipliers)(tVectorIndex, tConstraintIndex), tMyConstraintGradient, static_cast<ScalarType>(1), aOutput);

                // Add contribution from \mu*h_i(\mathbf{u}(\mathbf{z}),\mathbf{z})\frac{\partial h_i}{\partial\mathbf{z}}.
                ScalarType tAlpha = tOneOverPenalty * mCurrentConstraintValues->operator()(tVectorIndex, tConstraintIndex);
                locus::update(tAlpha, tMyConstraintGradient, static_cast<ScalarType>(1), aOutput);
            }
        }
        // Compute Augmented Lagrangian gradient
        locus::update(static_cast<ScalarType>(1), *mObjectiveGradient, static_cast<ScalarType>(1), aOutput);
    }
    /*! Reduced space interface: Assemble the reduced space gradient operator. \n
        In: \n
            aControl = design variable vector, unchanged on exist. \n
            aVector = trial descent direction, unchanged on exist. \n
        Out: \n
            aOutput = application of the trial descent direction to the Hessian operator.
    */
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mObjectiveHessian.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mObjectiveHessian->apply(*mState, aControl, aVector, aOutput);
        this->increaseObjectiveHessianEvaluationCounter();

        // Apply vector to inequality constraint Hessian operator and add contribution to total Hessian
        const ScalarType tOneOverPenalty = static_cast<ScalarType>(1.) / mCurrentLagrangeMultipliersPenalty;
        const OrdinalType tNumConstraintVectors = mCurrentConstraintValues->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
        {
            const OrdinalType tNumConstraints = (*mCurrentConstraintValues)[tVectorIndex].size();
            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                assert(mConstraintHessians.get() != nullptr);
                assert(mConstraintHessians->ptr(tConstraintIndex).get() != nullptr);
                // Add contribution from: \lambda_i\frac{\partial^2 h_i}{\partial\mathbf{z}^2}
                locus::fill(static_cast<ScalarType>(0), *mControlWorkVec);
                (*mConstraintHessians)[tConstraintIndex].apply(*mState, aControl, aVector, *mControlWorkVec);
                this->increaseConstraintHessianEvaluationCounter(tConstraintIndex);
                locus::update((*mLagrangeMultipliers)(tVectorIndex, tConstraintIndex),
                              *mControlWorkVec,
                              static_cast<ScalarType>(1),
                              aOutput);

                // Add contribution from: \mu\frac{\partial^2 h_i}{\partial\mathbf{z}^2}\h_i(\mathbf{z})
                ScalarType tAlpha = tOneOverPenalty * (*mCurrentConstraintValues)(tVectorIndex, tConstraintIndex);
                locus::update(tAlpha, *mControlWorkVec, static_cast<ScalarType>(1), aOutput);

                // Compute Jacobian, i.e. \frac{\partial h_i}{\partial\mathbf{z}}
                locus::fill(static_cast<ScalarType>(0), *mControlWorkVec);
                (*mConstraintGradientOperators)[tConstraintIndex].compute(*mState, aControl, *mControlWorkVec);
                this->increaseConstraintGradientEvaluationCounter(tConstraintIndex);

                ScalarType tJacobianDotTrialDirection = locus::dot(*mControlWorkVec, aVector);
                ScalarType tBeta = tOneOverPenalty * tJacobianDotTrialDirection;
                // Add contribution from: \mu\left(\frac{\partial h_i}{\partial\mathbf{z}}^{T}
                //                        \frac{\partial h_i}{\partial\mathbf{z}}\right)
                locus::update(tBeta, *mControlWorkVec, static_cast<ScalarType>(1), aOutput);
            }
        }
    }
    void applyVectorToPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                     const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                     locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mPreconditioner.get() != nullptr);
        mPreconditioner->applyPreconditioner(aControl, aVector, aOutput);
    }
    void applyVectorToInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                        locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mPreconditioner.get() != nullptr);
        mPreconditioner->applyInvPreconditioner(aControl, aVector, aOutput);
    }

    void evaluateConstraint(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        // Evaluate inequality constraints, h(\mathbf{u}(\mathbf{z}),\mathbf{z})
        const OrdinalType tNumConstraintVectors = mWorkConstraintValues->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
        {
            const OrdinalType tNumConstraints = (*mWorkConstraintValues)[tVectorIndex].size();
            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                (*mWorkConstraintValues)(tVectorIndex, tConstraintIndex) = (*mConstraints)[tConstraintIndex].value(*mState, aControl);
                this->increaseConstraintEvaluationCounter(tConstraintIndex);
            }
        }
    }
    bool updateLagrangeMultipliers()
    {
        bool tIsPenaltyBelowTolerance = false;
        ScalarType tPreviousPenalty = mCurrentLagrangeMultipliersPenalty;
        mCurrentLagrangeMultipliersPenalty = mPenaltyScaleFactor * mCurrentLagrangeMultipliersPenalty;
        if(mCurrentLagrangeMultipliersPenalty >= mMinPenaltyValue)
        {
            const OrdinalType tNumConstraintVectors = mCurrentConstraintValues->getNumVectors();
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
            {
                const OrdinalType tNumConstraints = (*mCurrentConstraintValues)[tVectorIndex].size();
                for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
                {
                    ScalarType tAlpha = static_cast<ScalarType>(1.) / tPreviousPenalty;
                    ScalarType tBeta = tAlpha * (*mCurrentConstraintValues)(tVectorIndex, tConstraintIndex);
                    (*mLagrangeMultipliers)(tVectorIndex, tConstraintIndex) =
                            (*mLagrangeMultipliers)(tVectorIndex, tConstraintIndex) + tBeta;
                }
            }
        }
        else
        {
            tIsPenaltyBelowTolerance = true;
        }

        return (tIsPenaltyBelowTolerance);
    }
    void updateCurrentConstraintValues()
    {
        locus::update(static_cast<ScalarType>(1), *mWorkConstraintValues, static_cast<ScalarType>(0), *mCurrentConstraintValues);
    }
    void computeCurrentFeasibilityMeasure()
    {
        locus::update(static_cast<ScalarType>(1), *mCurrentConstraintValues, static_cast<ScalarType>(0), *mDualWorkVec);
        const OrdinalType tNumVectors = mDualWorkVec->getNumVectors();
        std::vector<ScalarType> tMaxValues(tNumVectors, 0.);
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyVector = (*mDualWorkVec)[tVectorIndex];
            tMyVector.modulus();
            tMaxValues[tVectorIndex] = mDualReductionOperations->max(tMyVector);
        }

        mCurrentFeasibilityMeasure = *std::max_element(tMaxValues.begin(), tMaxValues.end());
    }
    ScalarType getCurrentFeasibilityMeasure() const
    {
        return (mCurrentFeasibilityMeasure);
    }

private:
    void initialize()
    {
        const OrdinalType tVECTOR_INDEX = 0;
        const OrdinalType tNumConstraints = mCurrentConstraintValues->operator[](tVECTOR_INDEX).size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            mCostraintGradients->add(mControlWorkVec.operator*());
        }
    }
    void increaseObjectiveFunctionEvaluationCounter()
    {
        mNumObjectiveFunctionEvaluations++;
    }
    void increaseObjectiveGradientEvaluationCounter()
    {
        mNumObjectiveGradientEvaluations++;
    }
    void increaseObjectiveHessianEvaluationCounter()
    {
        mNumObjectiveHessianEvaluations++;
    }
    void increaseConstraintEvaluationCounter(const OrdinalType & aIndex)
    {
        mNumConstraintEvaluations[aIndex] = mNumConstraintEvaluations[aIndex] + static_cast<OrdinalType>(1);
    }
    void increaseConstraintGradientEvaluationCounter(const OrdinalType & aIndex)
    {
        mNumConstraintGradientEvaluations[aIndex] = mNumConstraintGradientEvaluations[aIndex] + static_cast<OrdinalType>(1);
    }
    void increaseConstraintHessianEvaluationCounter(const OrdinalType & aIndex)
    {
        mNumConstraintHessianEvaluations[aIndex] = mNumConstraintHessianEvaluations[aIndex] + static_cast<OrdinalType>(1);
    }

private:
    OrdinalType mNumObjectiveFunctionEvaluations;
    OrdinalType mNumObjectiveGradientEvaluations;
    OrdinalType mNumObjectiveHessianEvaluations;

    ScalarType mMinPenaltyValue;
    ScalarType mPenaltyScaleFactor;
    ScalarType mCurrentFeasibilityMeasure;
    ScalarType mCurrentLagrangeMultipliersPenalty;
    ScalarType mNormObjectiveFunctionGradient;

    std::vector<OrdinalType> mNumConstraintEvaluations;
    std::vector<OrdinalType> mNumConstraintGradientEvaluations;
    std::vector<OrdinalType> mNumConstraintHessianEvaluations;

    std::shared_ptr<locus::StateData<ScalarType, OrdinalType>> mStateData;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mState;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualWorkVec;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWorkVec;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mLagrangeMultipliers;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mWorkConstraintValues;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintValues;

    std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> mCostraintGradients;

    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mObjective;
    std::shared_ptr<locus::CriterionList<ScalarType, OrdinalType>> mConstraints;

    std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> mPreconditioner;
    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> mObjectiveHessian;
    std::shared_ptr<locus::LinearOperatorList<ScalarType, OrdinalType>> mConstraintHessians;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;

    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> mObjectiveGradientOperator;
    std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> mConstraintGradientOperators;

private:
    AugmentedLagrangianStageMng(const locus::AugmentedLagrangianStageMng<ScalarType, OrdinalType>&);
    locus::AugmentedLagrangianStageMng<ScalarType, OrdinalType> & operator=(const locus::AugmentedLagrangianStageMng<ScalarType, OrdinalType>&);
};

} // namespace locus

#endif /* LOCUS_AUGMENTEDLAGRANGIANSTAGEMNG_HPP_ */
