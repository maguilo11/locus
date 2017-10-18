/*
 * Locus_VectorTest.cpp
 *
 *  Created on: Jun 14, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <mpi.h>

#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <sstream>
#include <numeric>
#include <iterator>
#include <iostream>
#include <algorithm>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_CriterionList.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_StandardVector.hpp"
#include "Locus_OptimalityCriteria.hpp"
#include "Locus_ReductionOperations.hpp"
#include "Locus_StandardMultiVector.hpp"
#include "Locus_OptimalityCriteriaDataMng.hpp"
#include "Locus_OptimalityCriteriaStageMng.hpp"
#include "Locus_DistributedReductionOperations.hpp"
#include "Locus_SynthesisOptimizationSubProblem.hpp"
#include "Locus_SingleConstraintSubProblemTypeLP.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"
#include "Locus_OptimalityCriteriaTestObjectiveOne.hpp"
#include "Locus_OptimalityCriteriaTestObjectiveTwo.hpp"
#include "Locus_OptimalityCriteriaTestInequalityOne.hpp"
#include "Locus_OptimalityCriteriaTestInequalityTwo.hpp"

namespace locus
{

/**********************************************************************************************************/
/*********************************************** STATE DATA ***********************************************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class StateData
{
public:
    explicit StateData(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mCurrentControl(aDataFactory.control().create()),
            mCurrentTrialStep(aDataFactory.control().create()),
            mCurrentObjectiveGradient(aDataFactory.control().create()),
            mCurrentConstraintGradient(aDataFactory.control().create())
    {
    }
    ~StateData()
    {
    }

    ScalarType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ScalarType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);
        return (mCurrentControl.operator*());
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentControl.operator*());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentTrialStep() const
    {
        assert(mCurrentTrialStep.get() != nullptr);
        return (mCurrentTrialStep.operator*());
    }
    void setCurrentTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentTrialStep.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentTrialStep->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentTrialStep.operator*());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentObjectiveGradient() const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        return (mCurrentObjectiveGradient);
    }
    void setCurrentObjectiveGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentObjectiveGradient->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentObjectiveGradient.operator*());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentConstraintGradient() const
    {
        assert(mCurrentConstraintGradient.get() != nullptr);
        return (mCurrentConstraintGradient);
    }
    void setCurrentConstraintGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradient.get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentConstraintGradient->getNumVectors());

        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentConstraintGradient.operator*());
    }

private:
    ScalarType mCurrentObjectiveFunctionValue;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintGradient;

private:
    StateData(const locus::StateData<ScalarType, OrdinalType>&);
    locus::StateData<ScalarType, OrdinalType> & operator=(const locus::StateData<ScalarType, OrdinalType>&);
};

/**********************************************************************************************************/
/************************************* GRAD, HESS AND PREC OPERATORS **************************************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class GradientOperator
{
public:
    virtual ~GradientOperator()
    {
    }

    virtual void update(const locus::StateData<ScalarType, OrdinalType> & aStateData) = 0;
    virtual void compute(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                         const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> create() const = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class GradientOperatorList
{
public:
    GradientOperatorList() :
            mList()
    {
    }
    ~GradientOperatorList()
    {
    }

    OrdinalType size() const
    {
        return (mList.size());
    }
    void add(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::GradientOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::GradientOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> create() const
    {
        assert(this->size() > static_cast<OrdinalType>(0));

        std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::GradientOperatorList<ScalarType, OrdinalType>>();
        const OrdinalType tNumGradientOperators = this->size();
        for(OrdinalType tIndex = 0; tIndex < tNumGradientOperators; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);

            const std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> & tGradientOperator = mList[tIndex];
            tOutput->add(tGradientOperator);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> & ptr(const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>>> mList;

private:
    GradientOperatorList(const locus::GradientOperatorList<ScalarType, OrdinalType>&);
    locus::GradientOperatorList<ScalarType, OrdinalType> & operator=(const locus::GradientOperatorList<ScalarType, OrdinalType>&);
};

template<typename ScalarType, typename OrdinalType = size_t>
class AnalyticalGradient : public locus::GradientOperator<ScalarType, OrdinalType>
{
public:
    explicit AnalyticalGradient(const locus::Criterion<ScalarType, OrdinalType> & aCriterion) :
            mCriterion(aCriterion.create())
    {
    }
    explicit AnalyticalGradient(const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & aCriterion) :
            mCriterion(aCriterion)
    {
    }
    virtual ~AnalyticalGradient()
    {
    }

    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void compute(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                 const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mCriterion->gradient(aState, aControl, aOutput);
    }
    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::AnalyticalGradient<ScalarType, OrdinalType>>(mCriterion);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mCriterion;

private:
    AnalyticalGradient(const locus::AnalyticalGradient<ScalarType, OrdinalType> & aRhs);
    locus::AnalyticalGradient<ScalarType, OrdinalType> & operator=(const locus::AnalyticalGradient<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class LinearOperator
{
public:
    virtual ~LinearOperator()
    {
    }

    virtual void update(const locus::StateData<ScalarType, OrdinalType> & aStateData) = 0;
    virtual void apply(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                       const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                       const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                       locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> create() const = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class LinearOperatorList
{
public:
    LinearOperatorList() :
            mList()
    {
    }
    ~LinearOperatorList()
    {
    }

    OrdinalType size() const
    {
        return (mList.size());
    }
    void add(const locus::LinearOperator<ScalarType, OrdinalType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::LinearOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::LinearOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::LinearOperatorList<ScalarType, OrdinalType>> create() const
    {
        assert(this->size() > static_cast<OrdinalType>(0));

        std::shared_ptr<locus::LinearOperatorList<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::LinearOperatorList<ScalarType, OrdinalType>>();
        const OrdinalType tNumLinearOperators = this->size();
        for(OrdinalType tIndex = 0; tIndex < tNumLinearOperators; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);

            const std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> & tLinearOperator = mList[tIndex];
            tOutput->add(tLinearOperator);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> & ptr(const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>>> mList;

private:
    LinearOperatorList(const locus::LinearOperatorList<ScalarType, OrdinalType>&);
    locus::LinearOperatorList<ScalarType, OrdinalType> & operator=(const locus::LinearOperatorList<ScalarType, OrdinalType>&);
};

template<typename ScalarType, typename OrdinalType = size_t>
class AnalyticalHessian : public locus::LinearOperator<ScalarType, OrdinalType>
{
public:
    explicit AnalyticalHessian(const locus::Criterion<ScalarType, OrdinalType> & aCriterion) :
            mCriterion(aCriterion.create())
    {
    }
    explicit AnalyticalHessian(const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & aCriterion) :
            mCriterion(aCriterion)
    {
    }
    virtual ~AnalyticalHessian()
    {
    }

    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void apply(const locus::MultiVector<ScalarType, OrdinalType> & aState,
               const locus::MultiVector<ScalarType, OrdinalType> & aControl,
               const locus::MultiVector<ScalarType, OrdinalType> & aVector,
               locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mCriterion->hessian(aState, aControl, aVector, aOutput);
    }
    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::AnalyticalHessian<ScalarType, OrdinalType>>(mCriterion);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mCriterion;

private:
    AnalyticalHessian(const locus::AnalyticalHessian<ScalarType, OrdinalType> & aRhs);
    locus::AnalyticalHessian<ScalarType, OrdinalType> & operator=(const locus::AnalyticalHessian<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class IdentityHessian : public locus::LinearOperator<ScalarType, OrdinalType>
{
public:
    IdentityHessian()
    {
    }
    virtual ~IdentityHessian()
    {
    }

    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void apply(const locus::MultiVector<ScalarType, OrdinalType> & aState,
               const locus::MultiVector<ScalarType, OrdinalType> & aControl,
               const locus::MultiVector<ScalarType, OrdinalType> & aVector,
               locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::update(static_cast<ScalarType>(1), aVector, static_cast<ScalarType>(0), aOutput);
    }
    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::IdentityHessian<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    IdentityHessian(const locus::IdentityHessian<ScalarType, OrdinalType> & aRhs);
    locus::IdentityHessian<ScalarType, OrdinalType> & operator=(const locus::IdentityHessian<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Preconditioner
{
public:
    virtual ~Preconditioner()
    {
    }

    virtual void update(const locus::StateData<ScalarType, OrdinalType> & aStateData) = 0;
    virtual void applyPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                     const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                     locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                        locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> create() const = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class IdentityPreconditioner : public Preconditioner<ScalarType, OrdinalType>
{
public:
    IdentityPreconditioner()
    {
    }
    virtual ~IdentityPreconditioner()
    {
    }
    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void applyPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                             const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                             locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() == aOutput.getNumVectors());
        locus::update(1., aVector, 0., aOutput);
    }
    void applyInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() == aOutput.getNumVectors());
        locus::update(1., aVector, 0., aOutput);
    }
    std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::IdentityPreconditioner<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    IdentityPreconditioner(const locus::IdentityPreconditioner<ScalarType, OrdinalType> & aRhs);
    locus::IdentityPreconditioner<ScalarType, OrdinalType> & operator=(const locus::IdentityPreconditioner<ScalarType, OrdinalType> & aRhs);
};

/**********************************************************************************************************/
/******************************************** MULTI VECTOR LIST *******************************************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class MultiVectorList
{
public:
    MultiVectorList() :
            mList()
    {
    }
    ~MultiVectorList()
    {
    }

    OrdinalType size() const
    {
        return (mList.size());
    }
    void add(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::MultiVector<ScalarType, OrdinalType> & operator [](const OrdinalType & aInput)
    {
        assert(aInput < mList.size());
        assert(mList[aInput].get() != nullptr);
        return (mList[aInput].operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & operator [](const OrdinalType & aInput) const
    {
        assert(aInput < mList.size());
        assert(mList[aInput].get() != nullptr);
        return (mList[aInput].operator*());
    }
    locus::Vector<ScalarType, OrdinalType> & operator ()(const OrdinalType & aListIndex, const OrdinalType & aVectorIndex)
    {
        assert(aListIndex < mList.size());
        assert(mList[aListIndex].get() != nullptr);
        return (mList[aListIndex]->operator[](aVectorIndex));
    }
    const locus::Vector<ScalarType, OrdinalType> & operator ()(const OrdinalType & aListIndex,
                                                              const OrdinalType & aVectorIndex) const
    {
        assert(aListIndex < mList.size());
        assert(mList[aListIndex].get() != nullptr);
        return (mList[aListIndex]->operator[](aVectorIndex));
    }
    std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> create() const
    {
        assert(this->size() > static_cast<OrdinalType>(0));
        std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::MultiVectorList<ScalarType, OrdinalType>>();
        const OrdinalType tListSize = this->size();
        for(OrdinalType tIndex = 0; tIndex < tListSize; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);
            const std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> & tMultiVector = mList[tIndex];
            tOutput->add(tMultiVector);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> & ptr(const OrdinalType & aInput) const
    {
        assert(aInput < mList.size());
        assert(mList[aInput].get() != nullptr);
        return(mList[aInput]);
    }

private:
    std::vector<std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>>> mList;

private:
    MultiVectorList(const locus::MultiVectorList<ScalarType, OrdinalType>&);
    locus::MultiVectorList<ScalarType, OrdinalType> & operator=(const locus::MultiVectorList<ScalarType, OrdinalType>&);
};

/**********************************************************************************************************/
/*************** AUGMENTED LAGRANGIAN IMPLEMENTATION OF KELLEY-SACHS TRUST REGION ALGORITHM ***************/
/**********************************************************************************************************/

struct algorithm
{
    enum stop_t
    {
        NaN_NORM_TRIAL_STEP = 1,
        NaN_NORM_GRADIENT = 2,
        NORM_GRADIENT = 3,
        NORM_STEP = 4,
        OBJECTIVE_STAGNATION = 5,
        MAX_NUMBER_ITERATIONS = 6,
        OPTIMALITY_AND_FEASIBILITY = 7,
        ACTUAL_REDUCTION_TOLERANCE = 8,
        CONTROL_STAGNATION = 9,
        NaN_OBJECTIVE_GRADIENT = 10,
        NaN_FEASIBILITY_VALUE = 11,
        NOT_CONVERGED = 12
    };
};

namespace bounds
{

template<typename ScalarType, typename OrdinalType = size_t>
void checkBounds(const locus::MultiVector<ScalarType, OrdinalType> & aLowerBounds,
                 const locus::MultiVector<ScalarType, OrdinalType> & aUpperBounds,
                 bool aPrintMessage = false)
{
    assert(aLowerBounds.getNumVectors() == aUpperBounds.getNumVectors());

    try
    {
        OrdinalType tNumVectors = aLowerBounds.getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            assert(aLowerBounds[tVectorIndex].size() == aUpperBounds[tVectorIndex].size());

            OrdinalType tNumElements = aLowerBounds[tVectorIndex].size();
            for(OrdinalType tElemIndex = 0; tElemIndex < tNumElements; tElemIndex++)
            {
                if(aLowerBounds(tVectorIndex, tElemIndex) >= aUpperBounds(tVectorIndex, tElemIndex))
                {
                    std::ostringstream tErrorMessage;
                    tErrorMessage << "\n\n**** ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                            << __PRETTY_FUNCTION__ << ", MESSAGE: LOWER BOUND AT ELEMENT INDEX " << tElemIndex
                            << " EXCEEDS/MATCHES UPPER BOUND WITH VALUE " << aLowerBounds(tVectorIndex, tElemIndex)
                            << ". UPPER BOUND AT ELEMENT INDEX " << tElemIndex << " HAS A VALUE OF "
                            << aUpperBounds(tVectorIndex, tElemIndex) << ": ABORT ****\n\n";
                    throw std::invalid_argument(tErrorMessage.str().c_str());
                }
            }
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        if(aPrintMessage == true)
        {
            std::cout << tErrorMsg.what() << std::flush;
        }
        throw tErrorMsg;
    }
}

template<typename ScalarType, typename OrdinalType = size_t>
void project(const locus::MultiVector<ScalarType, OrdinalType> & aLowerBound,
             const locus::MultiVector<ScalarType, OrdinalType> & aUpperBound,
             locus::MultiVector<ScalarType, OrdinalType> & aInput)
{
    assert(aInput.getNumVectors() == aUpperBound.getNumVectors());
    assert(aLowerBound.getNumVectors() == aUpperBound.getNumVectors());

    OrdinalType tNumVectors = aInput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        locus::Vector<ScalarType, OrdinalType> & tVector = aInput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tLowerBound = aLowerBound[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tUpperBound = aUpperBound[tVectorIndex];

        assert(tVector.size() == tLowerBound.size());
        assert(tUpperBound.size() == tLowerBound.size());

        OrdinalType tNumElements = tVector.size();
        for(OrdinalType tIndex = 0; tIndex < tNumElements; tIndex++)
        {
            tVector[tIndex] = std::max(tVector[tIndex], tLowerBound[tIndex]);
            tVector[tIndex] = std::min(tVector[tIndex], tUpperBound[tIndex]);
        }
    }
} // function project

template<typename ScalarType, typename OrdinalType = size_t>
void computeProjectedVector(const locus::MultiVector<ScalarType, OrdinalType> & aTrialControl,
                            const locus::MultiVector<ScalarType, OrdinalType> & aCurrentControl,
                            locus::MultiVector<ScalarType, OrdinalType> & aProjectedVector)
{
    assert(aTrialControl.getNumVectors() == aCurrentControl.getNumVectors());
    assert(aCurrentControl.getNumVectors() == aProjectedVector.getNumVectors());

    locus::update(static_cast<ScalarType>(1), aTrialControl, static_cast<ScalarType>(0), aProjectedVector);
    locus::update(static_cast<ScalarType>(-1), aCurrentControl, static_cast<ScalarType>(1), aProjectedVector);
} // function computeProjectedVector

template<typename ScalarType, typename OrdinalType = size_t>
void computeActiveAndInactiveSets(const locus::MultiVector<ScalarType, OrdinalType> & aInput,
                                  const locus::MultiVector<ScalarType, OrdinalType> & aLowerBound,
                                  const locus::MultiVector<ScalarType, OrdinalType> & aUpperBound,
                                  locus::MultiVector<ScalarType, OrdinalType> & aActiveSet,
                                  locus::MultiVector<ScalarType, OrdinalType> & aInactiveSet)
{
    assert(aInput.getNumVectors() == aLowerBound.getNumVectors());
    assert(aInput.getNumVectors() == aInactiveSet.getNumVectors());
    assert(aActiveSet.getNumVectors() == aInactiveSet.getNumVectors());
    assert(aLowerBound.getNumVectors() == aUpperBound.getNumVectors());

    OrdinalType tNumVectors = aInput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        locus::Vector<ScalarType, OrdinalType> & tActiveSet = aActiveSet[tVectorIndex];
        locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aInactiveSet[tVectorIndex];

        const locus::Vector<ScalarType, OrdinalType> & tVector = aInput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tLowerBound = aLowerBound[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tUpperBound = aUpperBound[tVectorIndex];

        assert(tVector.size() == tLowerBound.size());
        assert(tVector.size() == tInactiveSet.size());
        assert(tActiveSet.size() == tInactiveSet.size());
        assert(tUpperBound.size() == tLowerBound.size());

        tActiveSet.fill(0.);
        tInactiveSet.fill(0.);

        OrdinalType tNumElements = tVector.size();
        for(OrdinalType tIndex = 0; tIndex < tNumElements; tIndex++)
        {
            tActiveSet[tIndex] = static_cast<OrdinalType>((tVector[tIndex] >= tUpperBound[tIndex])
                    || (tVector[tIndex] <= tLowerBound[tIndex]));
            tInactiveSet[tIndex] = static_cast<OrdinalType>((tVector[tIndex] < tUpperBound[tIndex])
                    && (tVector[tIndex] > tLowerBound[tIndex]));
        }
    }
} // function computeActiveAndInactiveSets

} // namespace bounds

template<typename ScalarType, typename OrdinalType = size_t>
class StandardAlgorithmDataMng
{
public:
    virtual ~StandardAlgorithmDataMng()
    {
    }

    virtual OrdinalType getNumControlVectors() const = 0;

    // NOTE: OBJECTIVE FUNCTION VALUE
    virtual ScalarType getCurrentObjectiveFunctionValue() const = 0;
    virtual void setCurrentObjectiveFunctionValue(const ScalarType & aInput) = 0;
    virtual ScalarType getPreviousObjectiveFunctionValue() const = 0;
    virtual void setPreviousObjectiveFunctionValue(const ScalarType & aInput) = 0;

    // NOTE: SET INITIAL GUESS
    virtual void setInitialGuess(const ScalarType & aValue) = 0;
    virtual void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue) = 0;
    virtual void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInitialGuess) = 0;
    virtual void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInitialGuess) = 0;

    // NOTE: TRIAL STEP
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getTrialStep(const OrdinalType & aVectorIndex) const = 0;
    virtual void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aTrialStep) = 0;
    virtual void setTrialStep(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aTrialStep) = 0;

    // NOTE: CURRENT CONTROL
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getCurrentControl(const OrdinalType & aVectorIndex) const = 0;
    virtual void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aControl) = 0;

    // NOTE: PREVIOUS CONTROL
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getPreviousControl() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getPreviousControl(const OrdinalType & aVectorIndex) const = 0;
    virtual void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aControl) = 0;

    // NOTE: CURRENT GRADIENT
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getCurrentGradient(const OrdinalType & aVectorIndex) const = 0;
    virtual void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aGradient) = 0;
    virtual void setCurrentGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aGradient) = 0;

    // NOTE: PREVIOUS GRADIENT
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getPreviousGradient() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getPreviousGradient(const OrdinalType & aVectorIndex) const = 0;
    virtual void setPreviousGradient(const locus::MultiVector<ScalarType, OrdinalType> & aGradient) = 0;
    virtual void setPreviousGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aGradient) = 0;

    // NOTE: SET CONTROL LOWER BOUNDS
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const = 0;
    virtual void setControlLowerBounds(const ScalarType & aValue) = 0;
    virtual void setControlLowerBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue) = 0;
    virtual void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aLowerBound) = 0;
    virtual void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aLowerBound) = 0;

    // NOTE: SET CONTROL UPPER BOUNDS
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const = 0;
    virtual const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const = 0;
    virtual void setControlUpperBounds(const ScalarType & aValue) = 0;
    virtual void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue) = 0;
    virtual void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aUpperBound) = 0;
    virtual void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aUpperBound) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class TrustRegionAlgorithmDataMng : public StandardAlgorithmDataMng<ScalarType, OrdinalType>
{
public:
    explicit TrustRegionAlgorithmDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mNumDualVectors(aDataFactory.dual().getNumVectors()),
            mNumControlVectors(aDataFactory.control().getNumVectors()),
            mStagnationMeasure(0),
            mStationarityMeasure(0),
            mNormProjectedGradient(0),
            mGradientInexactnessTolerance(0),
            mObjectiveInexactnessTolerance(0),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mIsInitialGuessSet(false),
            mGradientInexactnessToleranceExceeded(false),
            mObjectiveInexactnessToleranceExceeded(false),
            mDual(aDataFactory.dual().create()),
            mTrialStep(aDataFactory.control().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mCurrentControl(aDataFactory.control().create()),
            mPreviousControl(aDataFactory.control().create()),
            mCurrentGradient(aDataFactory.control().create()),
            mPreviousGradient(aDataFactory.control().create()),
            mControlLowerBounds(aDataFactory.control().create()),
            mControlUpperBounds(aDataFactory.control().create()),
            mControlWorkVector(),
            mControlWorkMultiVector(aDataFactory.control().create()),
            mDualReductionOperations(aDataFactory.getDualReductionOperations().create()),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize(aDataFactory);
    }
    virtual ~TrustRegionAlgorithmDataMng()
    {
    }

    // NOTE: NUMBER OF CONTROL VECTORS
    OrdinalType getNumControlVectors() const
    {
        return (mNumControlVectors);
    }
    // NOTE: NUMBER OF DUAL VECTORS
    OrdinalType getNumDualVectors() const
    {
        return (mNumDualVectors);
    }

    // NOTE: OBJECTIVE FUNCTION VALUE
    ScalarType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ScalarType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    ScalarType getPreviousObjectiveFunctionValue() const
    {
        return (mPreviousObjectiveFunctionValue);
    }
    void setPreviousObjectiveFunctionValue(const ScalarType & aInput)
    {
        mPreviousObjectiveFunctionValue = aInput;
    }

    // NOTE: SET INITIAL GUESS
    void setInitialGuess(const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }

    // NOTE: DUAL VECTOR
    const locus::MultiVector<ScalarType, OrdinalType> & getDual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getDual(const OrdinalType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        return (mDual->operator [](aVectorIndex));
    }
    void setDual(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mDual->getNumVectors());
        locus::update(1., aInput, 0., *mDual);
    }
    void setDual(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        mDual->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: TRIAL STEP
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const
    {
        assert(mTrialStep.get() != nullptr);

        return (mTrialStep.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getTrialStep(const OrdinalType & aVectorIndex) const
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());

        return (mTrialStep->operator [](aVectorIndex));
    }
    void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mTrialStep->getNumVectors());
        locus::update(1., aInput, 0., *mTrialStep);
    }
    void setTrialStep(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());

        mTrialStep->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: ACTIVE SET FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getActiveSet() const
    {
        assert(mActiveSet.get() != nullptr);

        return (mActiveSet.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getActiveSet(const OrdinalType & aVectorIndex) const
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        return (mActiveSet->operator [](aVectorIndex));
    }
    void setActiveSet(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mActiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mActiveSet);
    }
    void setActiveSet(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        mActiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: INACTIVE SET FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getInactiveSet() const
    {
        assert(mInactiveSet.get() != nullptr);

        return (mInactiveSet.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getInactiveSet(const OrdinalType & aVectorIndex) const
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        return (mInactiveSet->operator [](aVectorIndex));
    }
    void setInactiveSet(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mInactiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mInactiveSet);
    }
    void setInactiveSet(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        mInactiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);

        return (mCurrentControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentControl(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mCurrentControl);
    }
    void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);

        return (mPreviousControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousControl(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousControl->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousControl);
    }
    void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        mPreviousControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const
    {
        assert(mCurrentGradient.get() != nullptr);

        return (mCurrentGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());

        return (mCurrentGradient->operator [](aVectorIndex));
    }
    void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentGradient->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentGradient);
    }
    void setCurrentGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());

        mCurrentGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousGradient() const
    {
        assert(mPreviousGradient.get() != nullptr);

        return (mPreviousGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());

        return (mPreviousGradient->operator [](aVectorIndex));
    }
    void setPreviousGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousGradient->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousGradient);
    }
    void setPreviousGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());

        mPreviousGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);

        return (mControlLowerBounds.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        return (mControlLowerBounds->operator [](aVectorIndex));
    }
    void setControlLowerBounds(const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mControlLowerBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlLowerBounds);
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);

        return (mControlUpperBounds.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        return (mControlUpperBounds->operator [](aVectorIndex));
    }
    void setControlUpperBounds(const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mControlUpperBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlUpperBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlUpperBounds);
    }

    // NOTE: OBJECTIVE AND GRADIENT INEXACTNESS VIOLATION FLAGS
    bool isInitialGuessSet() const
    {
        return (mIsInitialGuessSet);
    }
    void setGradientInexactnessFlag(const bool & aInput)
    {
        mGradientInexactnessToleranceExceeded = aInput;
    }
    bool isGradientInexactnessToleranceExceeded() const
    {
        return (mGradientInexactnessToleranceExceeded);
    }
    void setObjectiveInexactnessFlag(const bool & aInput)
    {
        mObjectiveInexactnessToleranceExceeded = aInput;
    }
    bool isObjectiveInexactnessToleranceExceeded() const
    {
        return (mObjectiveInexactnessToleranceExceeded);
    }

    // NOTE: STAGNATION MEASURE CRITERION
    void computeStagnationMeasure()
    {
        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkVector->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkVector->update(-1., tMyPreviousControl, 1.);
            mControlWorkVector->modulus();
            storage[tIndex] = mControlReductionOperations->max(*mControlWorkVector);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: NORM OF CURRENT PROJECTED GRADIENT
    ScalarType computeProjectedVectorNorm(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = aInput.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyInputVector = aInput[tIndex];

            mControlWorkVector->update(1., tMyInputVector, 0.);
            mControlWorkVector->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVector->dot(*mControlWorkVector);
        }
        ScalarType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormProjectedGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = mCurrentGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = (*mCurrentGradient)[tIndex];

            mControlWorkVector->update(1., tMyGradient, 0.);
            mControlWorkVector->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVector->dot(*mControlWorkVector);
        }
        mNormProjectedGradient = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormProjectedGradient() const
    {
        return (mNormProjectedGradient);
    }

    // NOTE: STATIONARITY MEASURE CALCULATION
    void computeStationarityMeasure()
    {
        assert(mInactiveSet.get() != nullptr);
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentGradient.get() != nullptr);
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlUpperBounds.get() != nullptr);

        locus::update(1., *mCurrentControl, 0., *mControlWorkMultiVector);
        locus::update(-1., *mCurrentGradient, 1., *mControlWorkMultiVector);
        locus::bounds::project(*mControlLowerBounds, *mControlUpperBounds, *mControlWorkMultiVector);
        locus::update(1., *mCurrentControl, -1., *mControlWorkMultiVector);
        locus::entryWiseProduct(*mInactiveSet, *mControlWorkMultiVector);
        mStationarityMeasure = locus::norm(*mControlWorkMultiVector);
    }
    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    // NOTE: RESET AND STORE STAGE DATA
    void resetCurrentStageDataToPreviousStageData()
    {
        OrdinalType tNumVectors = mCurrentGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = (*mCurrentControl)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = (*mPreviousControl)[tIndex];
            tMyCurrentControl.update(1., tMyPreviousControl, 0.);

            locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = (*mCurrentGradient)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousGradient = (*mPreviousGradient)[tIndex];
            tMyCurrentGradient.update(1., tMyPreviousGradient, 0.);
        }
        mCurrentObjectiveFunctionValue = mPreviousObjectiveFunctionValue;
    }
    void storeCurrentStageData()
    {
        const ScalarType tCurrentObjectiveValue = this->getCurrentObjectiveFunctionValue();
        this->setPreviousObjectiveFunctionValue(tCurrentObjectiveValue);

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = this->getCurrentControl(tVectorIndex);
            this->setPreviousControl(tVectorIndex, tMyCurrentControl);

            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = this->getCurrentGradient(tVectorIndex);
            this->setPreviousGradient(tVectorIndex, tMyCurrentGradient);
        }
    }

private:
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        assert(aDataFactory.control().getNumVectors() > 0);

        const OrdinalType tVectorIndex = 0;
        mControlWorkVector = aDataFactory.control(tVectorIndex).create();
        locus::fill(static_cast<ScalarType>(0), *mActiveSet);
        locus::fill(static_cast<ScalarType>(1), *mInactiveSet);

        ScalarType tScalarValue = std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlUpperBounds);
        tScalarValue = -std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlLowerBounds);
    }

private:
    OrdinalType mNumDualVectors;
    OrdinalType mNumControlVectors;

    ScalarType mStagnationMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mNormProjectedGradient;
    ScalarType mGradientInexactnessTolerance;
    ScalarType mObjectiveInexactnessTolerance;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;

    bool mIsInitialGuessSet;
    bool mGradientInexactnessToleranceExceeded;
    bool mObjectiveInexactnessToleranceExceeded;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWorkMultiVector;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    TrustRegionAlgorithmDataMng(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aRhs);
    locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & operator=(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aRhs);
};

struct solver
{
    enum stop_t
    {
        NaN_CURVATURE = 1,
        ZERO_CURVATURE = 2,
        NEGATIVE_CURVATURE = 3,
        INF_CURVATURE = 4,
        TOLERANCE = 5,
        TRUST_REGION_RADIUS = 6,
        MAX_ITERATIONS = 7,
        NaN_NORM_RESIDUAL = 8,
        INF_NORM_RESIDUAL = 9,
        INEXACTNESS_MEASURE = 10,
        ORTHOGONALITY_MEASURE = 11,
    };
};

struct preconditioner
{
    enum method_t
    {
        IDENTITY = 1,
    };
};

struct operators
{
    enum hessian_t
    {
        REDUCED = 1, SECANT = 2, USER_DEFINED = 3
    };
};

template<typename ScalarType, typename OrdinalType = size_t>
class TrustRegionStageMng
{
public:
    virtual ~TrustRegionStageMng()
    {
    }

    virtual void update(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                          ScalarType aTolerance = std::numeric_limits<ScalarType>::max()) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                             const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                             locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                                const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                                locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

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

template<typename ScalarType, typename OrdinalType = size_t>
class SteihaugTointSolver
{
public:
    SteihaugTointSolver() :
            mMaxNumIterations(200),
            mNumIterationsDone(0),
            mTolerance(1e-8),
            mNormResidual(0),
            mTrustRegionRadius(0),
            mRelativeTolerance(1e-1),
            mRelativeToleranceExponential(0.5),
            mStoppingCriterion(locus::solver::stop_t::MAX_ITERATIONS)
    {
    }
    virtual ~SteihaugTointSolver()
    {
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setNumIterationsDone(const OrdinalType & aInput)
    {
        mNumIterationsDone = aInput;
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    void setSolverTolerance(const ScalarType & aInput)
    {
        mTolerance = aInput;
    }
    ScalarType getSolverTolerance() const
    {
        return (mTolerance);
    }
    void setTrustRegionRadius(const ScalarType & aInput)
    {
        mTrustRegionRadius = aInput;
    }
    ScalarType getTrustRegionRadius() const
    {
        return (mTrustRegionRadius);
    }
    void setNormResidual(const ScalarType & aInput)
    {
        mNormResidual = aInput;
    }
    ScalarType getNormResidual() const
    {
        return (mNormResidual);
    }
    void setRelativeTolerance(const ScalarType & aInput)
    {
        mRelativeTolerance = aInput;
    }
    ScalarType getRelativeTolerance() const
    {
        return (mRelativeTolerance);
    }
    void setRelativeToleranceExponential(const ScalarType & aInput)
    {
        mRelativeToleranceExponential = aInput;
    }
    ScalarType getRelativeToleranceExponential() const
    {
        return (mRelativeToleranceExponential);
    }
    void setStoppingCriterion(const locus::solver::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }
    locus::solver::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    ScalarType computeSteihaugTointStep(const locus::MultiVector<ScalarType, OrdinalType> & aNewtonStep,
                                         const locus::MultiVector<ScalarType, OrdinalType> & aConjugateDir,
                                         const locus::MultiVector<ScalarType, OrdinalType> & aPrecTimesNewtonStep,
                                         const locus::MultiVector<ScalarType, OrdinalType> & aPrecTimesConjugateDir)
    {
        assert(aNewtonStep.getNumVectors() == aConjugateDir.getNumVectors());
        assert(aNewtonStep.getNumVectors() == aPrecTimesNewtonStep.getNumVectors());
        assert(aNewtonStep.getNumVectors() == aPrecTimesConjugateDir.getNumVectors());

        // Dogleg trust region step
        OrdinalType tNumVectors = aNewtonStep.getNumVectors();
        ScalarType tNewtonStepDotPrecTimesNewtonStep = 0;
        ScalarType tNewtonStepDotPrecTimesConjugateDir = 0;
        ScalarType tConjugateDirDotPrecTimesConjugateDir = 0;
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            tNewtonStepDotPrecTimesNewtonStep += aNewtonStep[tVectorIndex].dot(aPrecTimesNewtonStep[tVectorIndex]);
            tNewtonStepDotPrecTimesConjugateDir += aNewtonStep[tVectorIndex].dot(aPrecTimesConjugateDir[tVectorIndex]);
            tConjugateDirDotPrecTimesConjugateDir += aConjugateDir[tVectorIndex].dot(aPrecTimesConjugateDir[tVectorIndex]);
        }

        ScalarType tTrustRegionRadius = this->getTrustRegionRadius();
        ScalarType tAlpha = tNewtonStepDotPrecTimesConjugateDir * tNewtonStepDotPrecTimesConjugateDir;
        ScalarType tBeta = tConjugateDirDotPrecTimesConjugateDir
                * (tTrustRegionRadius * tTrustRegionRadius - tNewtonStepDotPrecTimesNewtonStep);
        ScalarType tAlphaPlusBeta = tAlpha + tBeta;
        ScalarType tNumerator = -tNewtonStepDotPrecTimesConjugateDir + std::sqrt(tAlphaPlusBeta);
        ScalarType tStep = tNumerator / tConjugateDirDotPrecTimesConjugateDir;

        return (tStep);
    }
    bool invalidCurvatureDetected(const ScalarType & aInput)
    {
        bool tInvalidCurvatureDetected = false;

        if(aInput < static_cast<ScalarType>(0.))
        {
            this->setStoppingCriterion(locus::solver::stop_t::NEGATIVE_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::abs(aInput) <= std::numeric_limits<ScalarType>::min())
        {
            this->setStoppingCriterion(locus::solver::stop_t::ZERO_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::isinf(aInput))
        {
            this->setStoppingCriterion(locus::solver::stop_t::INF_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::isnan(aInput))
        {
            this->setStoppingCriterion(locus::solver::stop_t::NaN_CURVATURE);
            tInvalidCurvatureDetected = true;
        }

        return (tInvalidCurvatureDetected);
    }
    bool toleranceSatisfied(const ScalarType & aNormDescentDirection)
    {
        this->setNormResidual(aNormDescentDirection);
        ScalarType tStoppingTolerance = this->getSolverTolerance();

        bool tToleranceCriterionSatisfied = false;
        if(aNormDescentDirection < tStoppingTolerance)
        {
            this->setStoppingCriterion(locus::solver::stop_t::TOLERANCE);
            tToleranceCriterionSatisfied = true;
        }
        else if(std::isinf(aNormDescentDirection))
        {
            this->setStoppingCriterion(locus::solver::stop_t::INF_NORM_RESIDUAL);
            tToleranceCriterionSatisfied = true;
        }
        else if(std::isnan(aNormDescentDirection))
        {
            this->setStoppingCriterion(locus::solver::stop_t::NaN_NORM_RESIDUAL);
            tToleranceCriterionSatisfied = true;
        }

        return (tToleranceCriterionSatisfied);
    }

    virtual void solve(locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                       locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng) = 0;

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mTolerance;
    ScalarType mNormResidual;
    ScalarType mTrustRegionRadius;
    ScalarType mRelativeTolerance;
    ScalarType mRelativeToleranceExponential;

    locus::solver::stop_t mStoppingCriterion;

private:
    SteihaugTointSolver(const locus::SteihaugTointSolver<ScalarType, OrdinalType> & aRhs);
    locus::SteihaugTointSolver<ScalarType, OrdinalType> & operator=(const locus::SteihaugTointSolver<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ProjectedSteihaugTointPcg : public locus::SteihaugTointSolver<ScalarType, OrdinalType>
{
public:
    explicit ProjectedSteihaugTointPcg(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            locus::SteihaugTointSolver<ScalarType, OrdinalType>(),
            mResidual(aDataFactory.control().create()),
            mNewtonStep(aDataFactory.control().create()),
            mCauchyStep(aDataFactory.control().create()),
            mWorkVector(aDataFactory.control().create()),
            mActiveVector(aDataFactory.control().create()),
            mInactiveVector(aDataFactory.control().create()),
            mConjugateDirection(aDataFactory.control().create()),
            mPrecTimesNewtonStep(aDataFactory.control().create()),
            mInvPrecTimesResidual(aDataFactory.control().create()),
            mPrecTimesConjugateDirection(aDataFactory.control().create()),
            mHessTimesConjugateDirection(aDataFactory.control().create())
    {
    }
    virtual ~ProjectedSteihaugTointPcg()
    {
    }

    void solve(locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
               locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        OrdinalType tNumVectors = aDataMng.getNumControlVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            (*mNewtonStep)[tVectorIndex].fill(0);
            (*mConjugateDirection)[tVectorIndex].fill(0);

            const locus::Vector<ScalarType, OrdinalType> & tCurrentGradient = aDataMng.getCurrentGradient(tVectorIndex);
            (*mResidual)[tVectorIndex].update(static_cast<ScalarType>(-1.), tCurrentGradient, static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mResidual)[tVectorIndex].entryWiseProduct(tInactiveSet);
        }
        ScalarType tNormResidual = locus::norm(*mResidual);
        this->setNormResidual(tNormResidual);

        this->iterate(aDataMng, aStageMng);

        ScalarType tNormNewtonStep = locus::norm(*mNewtonStep);
        if(tNormNewtonStep <= static_cast<ScalarType>(0.))
        {
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                const locus::Vector<ScalarType, OrdinalType> & tCurrentGradient = aDataMng.getCurrentGradient(tVectorIndex);
                (*mNewtonStep)[tVectorIndex].update(static_cast<ScalarType>(-1.), tCurrentGradient, static_cast<ScalarType>(0.));
                const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
                (*mNewtonStep)[tVectorIndex].entryWiseProduct(tInactiveSet);
            }
        }
        aDataMng.setTrialStep(*mNewtonStep);
    }

private:
    void iterate(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                 locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        ScalarType tPreviousTau = 0;
        ScalarType tNormResidual = this->getNormResidual();
        ScalarType tCurrentTrustRegionRadius = this->getTrustRegionRadius();

        OrdinalType tIteration = 0;
        OrdinalType tMaxNumIterations = this->getMaxNumIterations();
        while(this->toleranceSatisfied(tNormResidual) == false)
        {
            tIteration++;
            if(tIteration > tMaxNumIterations)
            {
                tIteration = tIteration - static_cast<OrdinalType>(1);
                this->setStoppingCriterion(locus::solver::stop_t::MAX_ITERATIONS);
                break;
            }
            this->applyVectorToInvPreconditioner(aDataMng, *mResidual, aStageMng, *mInvPrecTimesResidual);
            //compute scaling
            ScalarType tCurrentTau = locus::dot(*mResidual, *mInvPrecTimesResidual);
            if(tIteration > 1)
            {
                ScalarType tBeta = tCurrentTau / tPreviousTau;
                locus::update(static_cast<ScalarType>(1.), *mInvPrecTimesResidual, tBeta, *mConjugateDirection);
            }
            else
            {
                locus::update(static_cast<ScalarType>(1.), *mInvPrecTimesResidual, static_cast<ScalarType>(0.), *mConjugateDirection);
            }
            this->applyVectorToHessian(aDataMng, *mConjugateDirection, aStageMng, *mHessTimesConjugateDirection);
            ScalarType tCurvature = locus::dot(*mConjugateDirection, *mHessTimesConjugateDirection);
            if(this->invalidCurvatureDetected(tCurvature) == true)
            {
                // compute scaled inexact trial step
                ScalarType tScaling = this->step(aDataMng, aStageMng);
                locus::update(tScaling, *mConjugateDirection, static_cast<ScalarType>(1.), *mNewtonStep);
                break;
            }
            ScalarType tRayleighQuotient = tCurrentTau / tCurvature;
            locus::update(-tRayleighQuotient, *mHessTimesConjugateDirection, static_cast<ScalarType>(1.), *mResidual);
            tNormResidual = locus::norm(*mResidual);
            locus::update(tRayleighQuotient, *mConjugateDirection, static_cast<ScalarType>(1.), *mNewtonStep);
            if(tIteration == static_cast<OrdinalType>(1))
            {
                locus::update(static_cast<ScalarType>(1.), *mNewtonStep, static_cast<ScalarType>(0.), *mCauchyStep);
            }
            ScalarType tNormNewtonStep = locus::norm(*mNewtonStep);
            if(tNormNewtonStep > tCurrentTrustRegionRadius)
            {
                // compute scaled inexact trial step
                ScalarType tScaleFactor = this->step(aDataMng, aStageMng);
                locus::update(tScaleFactor, *mConjugateDirection, static_cast<ScalarType>(1), *mNewtonStep);
                this->setStoppingCriterion(locus::solver::stop_t::TRUST_REGION_RADIUS);
                break;
            }
            tPreviousTau = tCurrentTau;
        }
        this->setNumIterationsDone(tIteration);
    }
    ScalarType step(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                     locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        this->applyVectorToPreconditioner(aDataMng, *mNewtonStep, aStageMng, *mPrecTimesNewtonStep);
        this->applyVectorToPreconditioner(aDataMng, *mConjugateDirection, aStageMng, *mPrecTimesConjugateDirection);

        ScalarType tScaleFactor = this->computeSteihaugTointStep(*mNewtonStep,
                                                                  *mConjugateDirection,
                                                                  *mPrecTimesNewtonStep,
                                                                  *mPrecTimesConjugateDirection);

        return (tScaleFactor);
    }
    void applyVectorToHessian(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tNumVectors = aVector.getNumVectors();

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            // Set Active Vector
            (*mActiveVector)[tVectorIndex].update(static_cast<ScalarType>(1.), aVector[tVectorIndex], static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tActiveSet = aDataMng.getActiveSet(tVectorIndex);
            (*mActiveVector)[tVectorIndex].entryWiseProduct(tActiveSet);
            // Set Inactive Vector
            (*mInactiveVector)[tVectorIndex].update(static_cast<ScalarType>(1.), aVector[tVectorIndex], static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mInactiveVector)[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].fill(0);
        }

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToHessian(tCurrentControl, *mInactiveVector, aOutput);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            aOutput[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].update(static_cast<ScalarType>(1.), (*mActiveVector)[tVectorIndex], static_cast<ScalarType>(1.));
        }
    }
    void applyVectorToPreconditioner(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                                     const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                     locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                                     locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tNumVectors = aVector.getNumVectors();

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            // Set Active Vector
            (*mActiveVector)[tVectorIndex].update(static_cast<ScalarType>(1.), aVector[tVectorIndex], static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tActiveSet = aDataMng.getActiveSet(tVectorIndex);
            (*mActiveVector)[tVectorIndex].entryWiseProduct(tActiveSet);
            // Set Inactive Vector
            (*mInactiveVector)[tVectorIndex].update(static_cast<ScalarType>(1.), aVector[tVectorIndex], static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mInactiveVector)[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].fill(0);
        }

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToPreconditioner(tCurrentControl, *mInactiveVector, aOutput);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            aOutput[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].update(static_cast<ScalarType>(1.), (*mActiveVector)[tVectorIndex], static_cast<ScalarType>(1.));
        }
    }
    void applyVectorToInvPreconditioner(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                        locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                                        locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tNumVectors = aVector.getNumVectors();

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            // Set Active Vector
            (*mActiveVector)[tVectorIndex].update(static_cast<ScalarType>(1.), aVector[tVectorIndex], static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tActiveSet = aDataMng.getActiveSet(tVectorIndex);
            (*mActiveVector)[tVectorIndex].entryWiseProduct(tActiveSet);
            // Set Inactive Vector
            (*mInactiveVector)[tVectorIndex].update(static_cast<ScalarType>(1.), aVector[tVectorIndex], static_cast<ScalarType>(0.));
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mInactiveVector)[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].fill(0);
        }

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToInvPreconditioner(tCurrentControl, *mInactiveVector, aOutput);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            aOutput[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].update(static_cast<ScalarType>(1.), (*mActiveVector)[tVectorIndex], static_cast<ScalarType>(1.));
        }
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mResidual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mNewtonStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCauchyStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mWorkVector;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveVector;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConjugateDirection;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPrecTimesNewtonStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInvPrecTimesResidual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPrecTimesConjugateDirection;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mHessTimesConjugateDirection;

private:
    ProjectedSteihaugTointPcg(const locus::ProjectedSteihaugTointPcg<ScalarType, OrdinalType> & aRhs);
    locus::ProjectedSteihaugTointPcg<ScalarType, OrdinalType> & operator=(const locus::ProjectedSteihaugTointPcg<ScalarType, OrdinalType> & aRhs);
};

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

template<typename ScalarType, typename OrdinalType = size_t>
class KelleySachsAugmentedLagrangian : public locus::KelleySachsAlgorithm<ScalarType, OrdinalType>
{
public:
    KelleySachsAugmentedLagrangian(const std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> & aDataFactory,
                                   const std::shared_ptr<locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType>> & aDataMng,
                                   const std::shared_ptr<locus::AugmentedLagrangianStageMng<ScalarType, OrdinalType>> & aStageMng) :
            locus::KelleySachsAlgorithm<ScalarType, OrdinalType>(*aDataFactory),
            mGammaConstant(1e-3),
            mOptimalityTolerance(1e-5),
            mFeasibilityTolerance(1e-4),
            mGradient(aDataFactory->control().create()),
            mStepMng(std::make_shared<locus::KelleySachsStepMng<ScalarType, OrdinalType>>(*aDataFactory)),
            mSolver(std::make_shared<locus::ProjectedSteihaugTointPcg<ScalarType, OrdinalType>>(*aDataFactory)),
            mDataMng(aDataMng),
            mStageMng(aStageMng)
    {
    }
    virtual ~KelleySachsAugmentedLagrangian()
    {
    }

    void setOptimalityTolerance(const ScalarType & aInput)
    {
        mOptimalityTolerance = aInput;
    }
    void setFeasibilityTolerance(const ScalarType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }

    void solve()
    {
        assert(mDataMng->isInitialGuessSet() == true);

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
        ScalarType tTolerance = mStepMng->getObjectiveInexactnessTolerance();
        ScalarType tCurrentObjectiveFunctionValue = mStageMng->evaluateObjective(tCurrentControl, tTolerance);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mStageMng->updateCurrentConstraintValues();

        mStageMng->computeGradient(tCurrentControl, *mGradient);
        mDataMng->setCurrentGradient(*mGradient);
        mDataMng->computeNormProjectedGradient();
        mDataMng->storeCurrentStageData();

        if(mStepMng->isInitialTrustRegionRadiusSetToNormProjectedGradient() == true)
        {
            ScalarType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
            mStepMng->setTrustRegionRadius(tNormProjectedGradient);
        }
        mDataMng->computeStationarityMeasure();

        OrdinalType tIteration = 0;
        while(1)
        {
            tIteration++;
            this->setNumIterationsDone(tIteration);
            // Compute adaptive constants to ensure superlinear convergence
            ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
            ScalarType tValue = std::pow(tStationarityMeasure, static_cast<ScalarType>(0.75));
            ScalarType tEpsilon = std::min(static_cast<ScalarType>(1e-3), tValue);
            mStepMng->setEpsilonConstant(tEpsilon);
            tValue = std::pow(tStationarityMeasure, static_cast<ScalarType>(0.95));
            ScalarType tEta = static_cast<ScalarType>(0.1) * std::min(static_cast<ScalarType>(1e-1), tValue);
            mStepMng->setEtaConstant(tEta);
            // Solve trust region subproblem
            mStepMng->solveSubProblem(*mDataMng, *mStageMng, *mSolver);
            // Update mid objective, control, and gradient information if necessary
            this->updateDataManager();
            // Update stage manager data
            mStageMng->update(mDataMng.operator*());
            if(this->checkStoppingCriteria() == true)
            {
                break;
            }
        }
    }

private:
    void updateDataManager()
    {
        // Store current objective function, control, and gradient values
        mDataMng->storeCurrentStageData();

        // Update inequality constraint values at mid point
        mStageMng->updateCurrentConstraintValues();
        // Compute gradient at new midpoint
        const locus::MultiVector<ScalarType, OrdinalType> & tMidControl = mStepMng->getMidPointControls();
        mStageMng->computeGradient(tMidControl, *mGradient);

        if(this->updateControl(*mGradient, *mStepMng, *mDataMng, *mStageMng) == true)
        {
            // Update new gradient and inequality constraint values since control
            // was successfully updated; else, keep mid gradient and thus mid control.
            mStageMng->updateCurrentConstraintValues();
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
            mStageMng->computeGradient(tCurrentControl, *mGradient);
            mDataMng->setCurrentGradient(*mGradient);
        }
        else
        {
            // Keep current objective function, control, and gradient values at mid point
            const ScalarType tMidObjectiveFunctionValue = mStepMng->getMidPointObjectiveFunctionValue();
            mDataMng->setCurrentObjectiveFunctionValue(tMidObjectiveFunctionValue);
            mDataMng->setCurrentControl(tMidControl);
            mDataMng->setCurrentGradient(*mGradient);
        }

        // Compute feasibility measure
        mStageMng->computeCurrentFeasibilityMeasure();
        // Compute norm of projected gradient
        mDataMng->computeNormProjectedGradient();
        // Compute stationarity measure
        mDataMng->computeStationarityMeasure();
        // Compute stagnation measure
        mDataMng->computeStagnationMeasure();
        // compute gradient inexactness bound
        ScalarType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
        mStepMng->updateGradientInexactnessTolerance(tNormProjectedGradient);
    }

    bool checkStoppingCriteria()
    {
        bool tStop = false;
        ScalarType tCurrentLagrangeMultipliersPenalty = mStageMng->getCurrentLagrangeMultipliersPenalty();
        ScalarType tTolerance = mGammaConstant * tCurrentLagrangeMultipliersPenalty;
        ScalarType tNormAugmentedLagrangianGradient = mDataMng->getNormProjectedGradient();
        if(tNormAugmentedLagrangianGradient <= tTolerance)
        {
            if(this->checkPrimaryStoppingCriteria() == true)
            {
                tStop = true;
            }
            else
            {
                // Update Lagrange multipliers and stop if penalty is below defined threshold/tolerance
                tStop = mStageMng->updateLagrangeMultipliers();
            }
        }
        else
        {
            const OrdinalType tIterationCount = this->getNumIterationsDone();
            const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
            const ScalarType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
            const ScalarType tOptimalityMeasure = mDataMng->getNormProjectedGradient();
            if( (tOptimalityMeasure < mOptimalityTolerance) && (tFeasibilityMeasure < mFeasibilityTolerance) )
            {
                this->setStoppingCriterion(locus::algorithm::stop_t::OPTIMALITY_AND_FEASIBILITY);
                tStop = true;
            }
            else if( tStagnationMeasure < this->getStagnationTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::CONTROL_STAGNATION);
            }
            else if( tIterationCount >= this->getMaxNumIterations() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS);
            }
        }

        return (tStop);
    }

    bool checkPrimaryStoppingCriteria()
    {
        bool tStop = false;
        if(this->checkNaN() == true)
        {
            // Stop optimization algorithm: NaN number detected
            tStop = true;
            mDataMng->resetCurrentStageDataToPreviousStageData();
        }
        else
        {
            const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
            const ScalarType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
            const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
            const ScalarType tOptimalityMeasure = mDataMng->getNormProjectedGradient();

            if( tStationarityMeasure <= this->getTrialStepTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::NORM_STEP);
            }
            else if( tStagnationMeasure < this->getStagnationTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::CONTROL_STAGNATION);
            }
            else if( (tOptimalityMeasure < mOptimalityTolerance) && (tFeasibilityMeasure < mFeasibilityTolerance) )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::OPTIMALITY_AND_FEASIBILITY);
            }
            else if( this->getNumIterationsDone() >= this->getMaxNumIterations() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS);
            }
        }

        return (tStop);
    }

    bool checkNaN()
    {
        const ScalarType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
        const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ScalarType tOptimalityMeasure = mDataMng->getNormProjectedGradient();
        const ScalarType tNormProjectedAugmentedLagrangianGradient = mDataMng->getNormProjectedGradient();

        bool tNaN_ValueDetected = false;
        if(std::isfinite(tStationarityMeasure) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_NORM_TRIAL_STEP);
        }
        else if(std::isfinite(tNormProjectedAugmentedLagrangianGradient) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_NORM_GRADIENT);
        }
        else if(std::isfinite(tOptimalityMeasure) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_OBJECTIVE_GRADIENT);
        }
        else if(std::isfinite(tFeasibilityMeasure) == false)
        {
            tNaN_ValueDetected = true;
            this->setStoppingCriterion(locus::algorithm::stop_t::NaN_FEASIBILITY_VALUE);
        }

        return (tNaN_ValueDetected);
    }

private:
    ScalarType mGammaConstant;
    ScalarType mOptimalityTolerance;
    ScalarType mFeasibilityTolerance;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mGradient;

    std::shared_ptr<locus::KelleySachsStepMng<ScalarType, OrdinalType>> mStepMng;
    std::shared_ptr<locus::ProjectedSteihaugTointPcg<ScalarType,OrdinalType>> mSolver;
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<ScalarType,OrdinalType>> mDataMng;
    std::shared_ptr<locus::AugmentedLagrangianStageMng<ScalarType,OrdinalType>> mStageMng;

private:
    KelleySachsAugmentedLagrangian(const locus::KelleySachsAugmentedLagrangian<ScalarType, OrdinalType> & aRhs);
    locus::KelleySachsAugmentedLagrangian<ScalarType, OrdinalType> & operator=(const locus::KelleySachsAugmentedLagrangian<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Rosenbrock : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    Rosenbrock()
    {
    }
    virtual ~Rosenbrock()
    {
    }

    /*!
     * Evaluate Rosenbrock function:
     *      f(\mathbf{x}) = 100 * \left(x_2 - x_1^2\right)^2 + \left(1 - x_1\right)^2
     * */
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                     const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tVectorIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        ScalarType tOutput = static_cast<ScalarType>(100.)
                * std::pow((tMyControl[1] - tMyControl[0] * tMyControl[0]), static_cast<ScalarType>(2))
                + std::pow(static_cast<ScalarType>(1) - tMyControl[0], static_cast<ScalarType>(2));

        return (tOutput);
    }
    /*!
     * Compute Rosenbrock gradient:
     *      \frac{\partial{f}}{\partial x_1} = -400 * \left(x_2 - x_1^2\right) * x_1 +
     *                                          \left(2 * \left(1 - x_1\right) \right)
     *      \frac{\partial{f}}{\partial x_2} = 200 * \left(x_2 - x_1^2\right)
     * */
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aControl.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tMyOutput = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        tMyOutput[0] = static_cast<ScalarType>(-400) * (tMyControl[1] - (tMyControl[0] * tMyControl[0])) * tMyControl[0]
                + static_cast<ScalarType>(2) * tMyControl[0] - static_cast<ScalarType>(2);
        tMyOutput[1] = static_cast<ScalarType>(200) * (tMyControl[1] - (tMyControl[0] * tMyControl[0]));
    }
    /*!
     * Compute Rosenbrock Hessian times vector:
     * */
    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                 const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aVector.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aControl.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tMyOutput = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyVector = aVector[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        tMyOutput[0] = ((static_cast<ScalarType>(2)
                - static_cast<ScalarType>(400) * (tMyControl[1] - (tMyControl[0] * tMyControl[0]))
                + static_cast<ScalarType>(800) * (tMyControl[0] * tMyControl[0])) * tMyVector[0])
                - (static_cast<ScalarType>(400) * tMyControl[0] * tMyVector[1]);
        tMyOutput[1] = (static_cast<ScalarType>(-400) * tMyControl[0] * tMyVector[0])
                + (static_cast<ScalarType>(200) * tMyVector[1]);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::Rosenbrock<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    Rosenbrock(const locus::Rosenbrock<ScalarType, OrdinalType> & aRhs);
    locus::Rosenbrock<ScalarType, OrdinalType> & operator=(const locus::Rosenbrock<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Circle : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    Circle()
    {
    }
    virtual ~Circle()
    {
    }

    /// \left(\mathbf{z}(0) - 1.\right)^2 + 2\left(\mathbf{z}(1) - 2\right)^2
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                     const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));

        const OrdinalType tVectorIndex = 0;
        ScalarType tAlpha = aControl(tVectorIndex, 0) - static_cast<ScalarType>(1.);
        ScalarType tBeta = aControl(tVectorIndex, 1) - static_cast<ScalarType>(2);
        tBeta = static_cast<ScalarType>(2.) * std::pow(tBeta, static_cast<ScalarType>(2));
        ScalarType tOutput = std::pow(tAlpha, static_cast<ScalarType>(2)) + tBeta;
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) =
                static_cast<ScalarType>(2.) * (aControl(tVectorIndex, 0) - static_cast<ScalarType>(1.));
        aOutput(tVectorIndex, 1) =
                static_cast<ScalarType>(4.) * (aControl(tVectorIndex, 1) - static_cast<ScalarType>(2.));

    }
    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                 const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ScalarType>(2.) * aVector(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ScalarType>(4.) * aVector(tVectorIndex, 1);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::Circle<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    Circle(const locus::Circle<ScalarType, OrdinalType> & aRhs);
    locus::Circle<ScalarType, OrdinalType> & operator=(const locus::Circle<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Radius : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    Radius() :
            mLimit(1)
    {
    }
    virtual ~Radius()
    {
    }

    /// \left(\mathbf{z}(0) - 1.\right)^2 + 2\left(\mathbf{z}(1) - 2\right)^2
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                      const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));

        const OrdinalType tVectorIndex = 0;
        ScalarType tOutput = std::pow(aControl(tVectorIndex, 0), static_cast<ScalarType>(2.)) +
                std::pow(aControl(tVectorIndex, 1), static_cast<ScalarType>(2.));
        tOutput = tOutput - mLimit;
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ScalarType>(2.) * aControl(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ScalarType>(2.) * aControl(tVectorIndex, 1);

    }
    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                 const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ScalarType>(2.) * aVector(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ScalarType>(2.) * aVector(tVectorIndex, 1);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::Radius<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    ScalarType mLimit;

private:
    Radius(const locus::Radius<ScalarType, OrdinalType> & aRhs);
    locus::Radius<ScalarType, OrdinalType> & operator=(const locus::Radius<ScalarType, OrdinalType> & aRhs);
};

/**********************************************************************************************************/
/******************************** NONLINEAR CONJUGATE GRADIENT ALGORITHM **********************************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientDataMng : public locus::StandardAlgorithmDataMng<ScalarType, OrdinalType>
{
public:
    NonlinearConjugateGradientDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
        mIsInitialGuessSet(false),
        mNormGradient(std::numeric_limits<ScalarType>::max()),
        mStagnationMeasure(std::numeric_limits<ScalarType>::max()),
        mStationarityMeasure(std::numeric_limits<ScalarType>::max()),
        mObjectiveStagnationMeasure(std::numeric_limits<ScalarType>::max()),
        mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
        mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
        mControlWork(),
        mTrialStep(aDataFactory.control().create()),
        mCurrentControl(aDataFactory.control().create()),
        mPreviousControl(aDataFactory.control().create()),
        mCurrentGradient(aDataFactory.control().create()),
        mPreviousGradient(aDataFactory.control().create()),
        mControlLowerBounds(aDataFactory.control().create()),
        mControlUpperBounds(aDataFactory.control().create()),
        mControlReductions(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize();
    }
    ~NonlinearConjugateGradientDataMng()
    {
    }

    bool isInitialGuessSet() const
    {
        return (mIsInitialGuessSet);
    }
    OrdinalType getNumControlVectors() const
    {
        return (mCurrentControl->getNumVectors());
    }

    // NOTE: OBJECTIVE FUNCTION VALUE
    ScalarType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ScalarType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    ScalarType getPreviousObjectiveFunctionValue() const
    {
        return (mPreviousObjectiveFunctionValue);
    }
    void setPreviousObjectiveFunctionValue(const ScalarType & aInput)
    {
        mPreviousObjectiveFunctionValue = aInput;
    }

    // NOTE: SET INITIAL GUESS
    void setInitialGuess(const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }

    // NOTE: TRIAL STEP
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const
    {
        assert(mTrialStep.get() != nullptr);
        return (mTrialStep.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getTrialStep(const OrdinalType & aVectorIndex) const
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());
        return (mTrialStep->operator [](aVectorIndex));
    }
    void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mTrialStep->getNumVectors());
        locus::update(1., aInput, 0., *mTrialStep);
    }
    void setTrialStep(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());
        mTrialStep->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);
        return (mCurrentControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentControl(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());
        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mCurrentControl);
    }
    void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());
        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);
        return (mPreviousControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousControl(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());
        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousControl->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousControl);
    }
    void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());
        mPreviousControl->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }

    // NOTE: CURRENT GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const
    {
        assert(mCurrentGradient.get() != nullptr);
        return (mCurrentGradient.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());
        return (mCurrentGradient->operator[](aVectorIndex));
    }
    void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentGradient->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mCurrentGradient.operator*());
    }
    void setCurrentGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());
        mCurrentGradient->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }

    // NOTE: PREVIOUS GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousGradient() const
    {
        assert(mPreviousGradient.get() != nullptr);
        return (mPreviousGradient.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());
        return (mPreviousGradient->operator[](aVectorIndex));
    }
    void setPreviousGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousGradient->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mPreviousGradient.operator*());
    }
    void setPreviousGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());
        mPreviousGradient->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);
        return (mControlLowerBounds.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());
        return (mControlLowerBounds->operator[](aVectorIndex));
    }
    void setControlLowerBounds(const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<OrdinalType>(0));
        OrdinalType tNumVectors = mControlLowerBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());
        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());
        mControlLowerBounds->operator [](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mControlLowerBounds.operator*());
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);
        return (mControlUpperBounds.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());
        return (mControlUpperBounds->operator[](aVectorIndex));
    }
    void setControlUpperBounds(const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<OrdinalType>(0));
        OrdinalType tNumVectors = mControlUpperBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator[](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());
        mControlUpperBounds->operator[](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());
        mControlUpperBounds->operator[](aVectorIndex).update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0));
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlUpperBounds->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), mControlUpperBounds.operator*());
    }

    // NOTE: CONTROL STAGNATION MEASURE CALCULATION
    void computeStagnationMeasure()
    {
        const OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWork->update(static_cast<ScalarType>(1), tMyCurrentControl, static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWork->update(static_cast<ScalarType>(-1), tMyPreviousControl, static_cast<ScalarType>(1));
            mControlWork->modulus();
            storage[tIndex] = mControlReductions->max(*mControlWork);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeObjectiveStagnationMeasure()
    {
        mObjectiveStagnationMeasure = mPreviousObjectiveFunctionValue - mCurrentObjectiveFunctionValue;
        mObjectiveStagnationMeasure = std::abs(mObjectiveStagnationMeasure);
    }
    ScalarType getObjectiveStagnationMeasure() const
    {
        return (mObjectiveStagnationMeasure);
    }

    // NOTE: NORM OF GRADIENT MEASURE CALCULATION
    void computeNormGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        const OrdinalType tNumVectors = mCurrentGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = mCurrentGradient->operator[](tIndex);
            tCummulativeDotProduct += tMyGradient.dot(tMyGradient);
        }
        mNormGradient = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormGradient() const
    {
        return (mNormGradient);
    }

    // NOTE: COMPUTE STATIONARITY MEASURE
    void computeStationarityMeasure()
    {
        ScalarType tCummulativeDotProduct = 0.;
        const OrdinalType tNumVectors = mTrialStep->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyTrialStep = mTrialStep->operator[](tIndex);
            tCummulativeDotProduct += tMyTrialStep.dot(tMyTrialStep);
        }
        mStationarityMeasure = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    // NOTE: STORE PREVIOUS STATE (SIDENOTE: THE PREVIOUS TRIAL STEP IS ALWAYS STORED IN mTrialStep)
    void storePreviousState()
    {
        mPreviousObjectiveFunctionValue = mCurrentObjectiveFunctionValue;
        locus::update(static_cast<ScalarType>(1), *mCurrentControl, static_cast<ScalarType>(0), *mPreviousControl);
        locus::update(static_cast<ScalarType>(1), *mCurrentGradient, static_cast<ScalarType>(0), *mPreviousGradient);
    }

private:
    void initialize()
    {
        const OrdinalType tVectorIndex = 0;
        mControlWork = mCurrentControl->operator[](tVectorIndex).create();

        ScalarType tValue = -std::numeric_limits<ScalarType>::max();
        locus::fill(tValue, mControlLowerBounds.operator*());
        tValue = std::numeric_limits<ScalarType>::max();
        locus::fill(tValue, mControlUpperBounds.operator*());
    }

private:
    bool mIsInitialGuessSet;

    ScalarType mNormGradient;
    ScalarType mStagnationMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mObjectiveStagnationMeasure;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWork;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductions;

private:
    NonlinearConjugateGradientDataMng(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStageMng
{
public:
    virtual ~NonlinearConjugateGradientStageMng()
    {
    }

    virtual void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                         ScalarType aTolerance = std::numeric_limits<ScalarType>::max()) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStandardStageMng : public locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType>
{
public:
    NonlinearConjugateGradientStandardStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory,
                                               const locus::Criterion<ScalarType, OrdinalType> & aObjective) :
            mNumHessianEvaluations(0),
            mNumFunctionEvaluations(0),
            mNumGradientEvaluations(0),
            mState(aDataFactory.state().create()),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory)),
            mObjective(aObjective.create()),
            mHessian(std::make_shared<locus::AnalyticalHessian<ScalarType, OrdinalType>>(aObjective)),
            mGradient(std::make_shared<locus::AnalyticalGradient<ScalarType, OrdinalType>>(aObjective))

    {
    }
    ~NonlinearConjugateGradientStandardStageMng()
    {
    }

    OrdinalType getNumObjectiveFunctionEvaluations() const
    {
        return (mNumFunctionEvaluations);
    }
    void setNumObjectiveFunctionEvaluations(const ScalarType & aInput)
    {
        mNumFunctionEvaluations = aInput;
    }
    OrdinalType getNumObjectiveGradientEvaluations() const
    {
        return (mNumGradientEvaluations);
    }
    void setNumObjectiveGradientEvaluations(const ScalarType & aInput)
    {
        mNumGradientEvaluations = aInput;
    }
    OrdinalType getNumObjectiveHessianEvaluations() const
    {
        return (mNumHessianEvaluations);
    }
    void setNumObjectiveHessianEvaluations(const ScalarType & aInput)
    {
        mNumHessianEvaluations = aInput;
    }

    void setGradient(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mGradient = aInput.create();
    }
    void setHessian(const locus::LinearOperator<ScalarType, OrdinalType> & aInput)
    {
        mHessian = aInput.create();
    }

    void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mStateData->setCurrentTrialStep(aDataMng.getTrialStep());
        mStateData->setCurrentControl(aDataMng.getCurrentControl());
        mStateData->setCurrentObjectiveGradient(aDataMng.getCurrentGradient());
        mStateData->setCurrentObjectiveFunctionValue(aDataMng.getCurrentObjectiveFunctionValue());

        mHessian->update(mStateData.operator*());
        mGradient->update(mStateData.operator*());
    }
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 ScalarType aTolerance = std::numeric_limits<ScalarType>::max())
    {
        assert(mObjective.get() != nullptr);
        ScalarType tObjectiveFunctionValue = mObjective->value(mState.operator*(), aControl);
        mNumFunctionEvaluations++;
        return (tObjectiveFunctionValue);
    }
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mGradient.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mGradient->compute(mState.operator*(), aControl, aOutput);
        mNumGradientEvaluations++;
    }
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mHessian.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mHessian->apply(mState.operator*(), aControl, aVector, aOutput);
        mNumHessianEvaluations++;
    }

private:
    OrdinalType mNumHessianEvaluations;
    OrdinalType mNumFunctionEvaluations;
    OrdinalType mNumGradientEvaluations;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mState;
    std::shared_ptr<locus::StateData<ScalarType, OrdinalType>> mStateData;
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mObjective;

    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> mHessian;
    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> mGradient;

private:
    NonlinearConjugateGradientStandardStageMng(const locus::NonlinearConjugateGradientStandardStageMng<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientStandardStageMng<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientStandardStageMng<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStep
{
public:
    virtual ~NonlinearConjugateGradientStep()
    {
    }

    virtual void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                               locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class PolakRibiere : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    explicit PolakRibiere(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~PolakRibiere()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousGradient());
        tBeta = tBeta / locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    PolakRibiere(const locus::PolakRibiere<ScalarType, OrdinalType> & aRhs);
    locus::PolakRibiere<ScalarType, OrdinalType> & operator=(const locus::PolakRibiere<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class FletcherReeves : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    FletcherReeves(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~FletcherReeves()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                / locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    FletcherReeves(const locus::FletcherReeves<ScalarType, OrdinalType> & aRhs);
    locus::FletcherReeves<ScalarType, OrdinalType> & operator=(const locus::FletcherReeves<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class HestenesStiefel : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    HestenesStiefel(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~HestenesStiefel()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousGradient());
        ScalarType tDenominator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getTrialStep())
                - locus::dot(aDataMng.getPreviousGradient(), aDataMng.getTrialStep());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    HestenesStiefel(const locus::HestenesStiefel<ScalarType, OrdinalType> & aRhs);
    locus::HestenesStiefel<ScalarType, OrdinalType> & operator=(const locus::HestenesStiefel<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConjugateDescent : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    ConjugateDescent(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~ConjugateDescent()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = static_cast<ScalarType>(-1)
                * (locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                        / locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient()));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    ConjugateDescent(const locus::ConjugateDescent<ScalarType, OrdinalType> & aRhs);
    locus::ConjugateDescent<ScalarType, OrdinalType> & operator=(const locus::ConjugateDescent<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DaiYuan : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiYuan(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiYuan()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tDenominator = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tBeta = tNumerator / tDenominator;
        //tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiYuan(const locus::DaiYuan<ScalarType, OrdinalType> & aRhs);
    locus::DaiYuan<ScalarType, OrdinalType> & operator=(const locus::DaiYuan<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class LiuStorey : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    LiuStorey(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~LiuStorey()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                - locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousGradient());
        ScalarType tDenominator = static_cast<ScalarType>(-1)
                * locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    LiuStorey(const locus::LiuStorey<ScalarType, OrdinalType> & aRhs);
    locus::LiuStorey<ScalarType, OrdinalType> & operator=(const locus::LiuStorey<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class Daniels : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    Daniels(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mHessianTimesVector(aDataFactory.control().create()),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~Daniels()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        aStageMng.applyVectorToHessian(aDataMng.getCurrentControl(),
                                       aDataMng.getTrialStep(),
                                       mHessianTimesVector.operator*());

        ScalarType tNumerator = locus::dot(aDataMng.getCurrentGradient(), mHessianTimesVector.operator*());
        ScalarType tDenominator = locus::dot(aDataMng.getTrialStep(), mHessianTimesVector.operator*());
        ScalarType tBeta = tNumerator / tDenominator;
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mHessianTimesVector;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    Daniels(const locus::Daniels<ScalarType, OrdinalType> & aRhs);
    locus::Daniels<ScalarType, OrdinalType> & operator=(const locus::Daniels<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DaiLiao : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiLiao(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaleFactor(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiLiao()
    {
    }

    void setScaleFactor(const ScalarType & aInput)
    {
        mScaleFactor = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentControlDotCurrentGradient = locus::dot(aDataMng.getCurrentControl(), aDataMng.getCurrentGradient());
        ScalarType tPreviousControlDotCurrentGradient = locus::dot(aDataMng.getPreviousControl(), aDataMng.getCurrentGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType tDeltaControlDotCurrentGradient = tCurrentControlDotCurrentGradient
                - tPreviousControlDotCurrentGradient;

        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (mScaleFactor * tDeltaControlDotCurrentGradient));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mScaleFactor;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiLiao(const locus::DaiLiao<ScalarType, OrdinalType> & aRhs);
    locus::DaiLiao<ScalarType, OrdinalType> & operator=(const locus::DaiLiao<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DaiYuanHybrid : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    DaiYuanHybrid(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mWolfeConstant(static_cast<ScalarType>(1) / static_cast<ScalarType>(3)),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~DaiYuanHybrid()
    {
    }

    void setWolfeConstant(const ScalarType & aInput)
    {
        mWolfeConstant = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());

        ScalarType tHestenesStiefelBeta = (tCurrentGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);

        ScalarType tDaiYuanBeta = tCurrentGradientDotCurrentGradient
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tScaleFactor = (static_cast<ScalarType>(1) - mWolfeConstant)
                / (static_cast<ScalarType>(1) + mWolfeConstant);
        tScaleFactor = static_cast<ScalarType>(-1) * tScaleFactor;
        ScalarType tScaledDaiYuanBeta = tScaleFactor * tDaiYuanBeta;

        ScalarType tBeta = std::max(tScaledDaiYuanBeta, std::min(tDaiYuanBeta, tHestenesStiefelBeta));
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mWolfeConstant;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    DaiYuanHybrid(const locus::DaiYuanHybrid<ScalarType, OrdinalType> & aRhs);
    locus::DaiYuanHybrid<ScalarType, OrdinalType> & operator=(const locus::DaiYuanHybrid<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class HagerZhang : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    HagerZhang(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mLowerBound(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~HagerZhang()
    {
    }

    void setLowerBound(const ScalarType & aInput)
    {
        mLowerBound = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType DeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;
        ScalarType tScaleFactor = static_cast<ScalarType>(2) * DeltaGradientDotDeltaGradient
                * tOneOverTrialStepDotDeltaGradient;

        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (tScaleFactor * tTrialStepDotCurrentGradient));
        ScalarType tTrialStepDotTrialStep = locus::dot(aDataMng.getTrialStep(), aDataMng.getTrialStep());
        ScalarType tNormTrialStep = std::sqrt(tTrialStepDotTrialStep);
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradientDotPreviousGradient);
        ScalarType tLowerBound = static_cast<ScalarType>(-1)
                / (tNormTrialStep * std::min(tNormPreviousGradient, mLowerBound));
        tBeta = std::max(tBeta, tLowerBound);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType mLowerBound;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    HagerZhang(const locus::HagerZhang<ScalarType, OrdinalType> & aRhs);
    locus::HagerZhang<ScalarType, OrdinalType> & operator=(const locus::HagerZhang<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class PerryShanno : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    PerryShanno(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mLowerBound(0.1),
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~PerryShanno()
    {
    }

    void setLowerBound(const ScalarType & aInput)
    {
        mLowerBound = aInput;
    }
    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = this->computeBeta(aDataMng);
        ScalarType tAlpha = this->computeAlpha(aDataMng);
        ScalarType tTheta = this->computeTheta(aDataMng);

        locus::scale(tBeta, mScaledDescentDirection.operator*());
        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::update(tAlpha,
                      aDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::update(-tAlpha,
                      aDataMng.getPreviousGradient(),
                      static_cast<ScalarType>(1),
                      mScaledDescentDirection.operator*());
        locus::scale(tTheta, mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    ScalarType computeBeta(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tOneOverTrialStepDotDeltaGradient = static_cast<ScalarType>(1)
                / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        ScalarType tDeltaGradientDotCurrentGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient;
        ScalarType tDeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;

        ScalarType tScaleFactor = static_cast<ScalarType>(2) * tDeltaGradientDotDeltaGradient
                * tOneOverTrialStepDotDeltaGradient;
        ScalarType tBeta = tOneOverTrialStepDotDeltaGradient
                * (tDeltaGradientDotCurrentGradient - (tScaleFactor * tTrialStepDotCurrentGradient));
        ScalarType tNormTrialStep = locus::dot(aDataMng.getTrialStep(), aDataMng.getTrialStep());
        tNormTrialStep = std::sqrt(tNormTrialStep);
        ScalarType tNormPreviousGradient = std::sqrt(tPreviousGradientDotPreviousGradient);
        ScalarType tLowerBound = static_cast<ScalarType>(-1)
                / (tNormTrialStep * std::min(tNormPreviousGradient, mLowerBound));

        tBeta = std::max(tBeta, tLowerBound);
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());
        return (tBeta);
    }
    ScalarType computeAlpha(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tTrialStepDotCurrentGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getCurrentGradient());
        ScalarType tTrialStepDotPreviousGradient = locus::dot(aDataMng.getTrialStep(), aDataMng.getPreviousGradient());
        ScalarType tAlpha = tTrialStepDotCurrentGradient / (tTrialStepDotCurrentGradient - tTrialStepDotPreviousGradient);
        tAlpha = std::max(tAlpha, std::numeric_limits<ScalarType>::min());
        return (tAlpha);
    }
    ScalarType computeTheta(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        ScalarType tCurrentGradientDotCurrentControl = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentControl());
        ScalarType tPreviousGradientDotCurrentControl = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentControl());
        ScalarType tCurrentGradientDotPreviousControl = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getPreviousControl());
        ScalarType tPreviousGradientDotPreviousControl = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousControl());

        ScalarType tCurrentGradientDotCurrentGradient = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotCurrentGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getCurrentGradient());
        ScalarType tPreviousGradientDotPreviousGradient = locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());

        ScalarType tDeltaGradientDotDeltaGradient = tCurrentGradientDotCurrentGradient
                - tPreviousGradientDotCurrentGradient - tPreviousGradientDotCurrentGradient
                + tPreviousGradientDotPreviousGradient;
        ScalarType tDeltaGradientDotDeltaControl = tCurrentGradientDotCurrentControl
                - tPreviousGradientDotCurrentControl - tCurrentGradientDotPreviousControl
                + tPreviousGradientDotPreviousControl;

        ScalarType tTheta = tDeltaGradientDotDeltaControl / tDeltaGradientDotDeltaGradient;
        tTheta = std::max(tTheta, std::numeric_limits<ScalarType>::min());
        return (tTheta);
    }

private:
    ScalarType mLowerBound;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    PerryShanno(const locus::PerryShanno<ScalarType, OrdinalType> & aRhs);
    locus::PerryShanno<ScalarType, OrdinalType> & operator=(const locus::PerryShanno<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class StateManager
{
public:
    virtual ~StateManager()
    {
    }

    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;

    virtual ScalarType getCurrentObjectiveValue() const = 0;
    virtual void setCurrentObjectiveValue(const ScalarType & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const = 0;
    virtual void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const = 0;
    virtual void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const = 0;
    virtual void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const = 0;
    virtual void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const = 0;
    virtual void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientStateMng : public locus::StateManager<ScalarType, OrdinalType>
{
public:
    NonlinearConjugateGradientStateMng(const std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> & aDataMng,
                                       const std::shared_ptr<locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType>> & aStageMng) :
            mDataMng(aDataMng),
            mStageMng(aStageMng)
    {
    }
    virtual ~NonlinearConjugateGradientStateMng()
    {
    }

    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        ScalarType tOutput = mStageMng->evaluateObjective(aControl);
        return (tOutput);
    }
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        mStageMng->computeGradient(aControl, aOutput);
    }
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        mStageMng->applyVectorToHessian(aControl, aVector, aOutput);
    }

    ScalarType getCurrentObjectiveValue() const
    {
        return (mDataMng->getCurrentObjectiveFunctionValue());
    }
    void setCurrentObjectiveValue(const ScalarType & aInput)
    {
        mDataMng->setCurrentObjectiveFunctionValue(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const
    {
        return (mDataMng->getTrialStep());
    }
    void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setTrialStep(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        return (mDataMng->getCurrentControl());
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setCurrentControl(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const
    {
        return (mDataMng->getCurrentGradient());
    }
    void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setCurrentGradient(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        return (mDataMng->getControlLowerBounds());
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setControlLowerBounds(aInput);
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        return (mDataMng->getControlUpperBounds());
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDataMng->setControlUpperBounds(aInput);
    }

    locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & getDataMng()
    {
        return (mDataMng.operator*());
    }
    locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & getStageMng()
    {
        return (mStageMng.operator*());
    }

private:
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> mDataMng;
    std::shared_ptr<locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType>> mStageMng;

private:
    NonlinearConjugateGradientStateMng(const locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class LineSearch
{
public:
    virtual ~LineSearch()
    {
    }

    virtual OrdinalType getNumIterationsDone() const = 0;
    virtual void setMaxNumIterations(const OrdinalType & aInput) = 0;
    virtual void setContractionFactor(const ScalarType & aInput) = 0;
    virtual void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class CubicLineSearch : public locus::LineSearch<ScalarType, OrdinalType>
{
public:
    explicit CubicLineSearch(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mContractionFactor(0.5),
            mArmijoRuleConstant(1e-4),
            mStagnationTolerance(1e-8),
            mInitialGradientDotTrialStep(0),
            mStepValues(3, static_cast<ScalarType>(0)),
            mTrialControl(aDataFactory.control().create()),
            mProjectedTrialStep(aDataFactory.control().create())
    {
    }
    virtual ~CubicLineSearch()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    ScalarType getStepValue() const
    {
        return (mStepValues[2]);
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStagnationTolerance(const OrdinalType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    void setContractionFactor(const ScalarType & aInput)
    {
        mContractionFactor = aInput;
    }

    void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng)
    {
        ScalarType tInitialStep = 1;
        locus::update(static_cast<ScalarType>(1),
                      aStateMng.getCurrentControl(),
                      static_cast<ScalarType>(0),
                      mTrialControl.operator*());
        locus::update(tInitialStep, aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

        ScalarType tTrialObjectiveValue = aStateMng.evaluateObjective(mTrialControl.operator*());
        // tTrialObjectiveValue[0] = current, tTrialObjectiveValue[1] = old, tTrialObjectiveValue[2] = trial
        const OrdinalType tSize = 3;
        std::vector<ScalarType> tObjectiveFunctionValues(tSize, 0.);
        tObjectiveFunctionValues[0] = aStateMng.getCurrentObjectiveValue();
        tObjectiveFunctionValues[2] = tTrialObjectiveValue;
        // step[0] = old, step[1] = current, step[2] = new
        mStepValues[2] = tInitialStep;
        mStepValues[1] = mStepValues[2];

        mNumIterationsDone = 1;
        mInitialGradientDotTrialStep = locus::dot(aStateMng.getCurrentGradient(), aStateMng.getTrialStep());
        while(mNumIterationsDone <= mMaxNumIterations)
        {
            ScalarType tSufficientDecreaseCondition = tObjectiveFunctionValues[0]
                    + (mArmijoRuleConstant * mStepValues[1] * mInitialGradientDotTrialStep);
            bool tSufficientDecreaseConditionSatisfied =
                    tObjectiveFunctionValues[2] < tSufficientDecreaseCondition ? true : false;
            bool tStepIsLessThanTolerance = mStepValues[2] < mStagnationTolerance ? true : false;
            if(tSufficientDecreaseConditionSatisfied || tStepIsLessThanTolerance)
            {
                break;
            }
            mStepValues[0] = mStepValues[1];
            mStepValues[1] = mStepValues[2];
            if(mNumIterationsDone == static_cast<OrdinalType>(1))
            {
                // first backtrack: do a quadratic fit
                ScalarType tDenominator = static_cast<ScalarType>(2)
                        * (tObjectiveFunctionValues[2] - tObjectiveFunctionValues[0] - mInitialGradientDotTrialStep);
                mStepValues[2] = -mInitialGradientDotTrialStep / tDenominator;
            }
            else
            {
                this->interpolate(tObjectiveFunctionValues);
            }
            this->checkCurrentStepValue();
            locus::update(static_cast<ScalarType>(1),
                          aStateMng.getCurrentControl(),
                          static_cast<ScalarType>(0),
                          mTrialControl.operator*());
            locus::update(mStepValues[2], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

            tTrialObjectiveValue = aStateMng.evaluateObjective(mTrialControl.operator*());
            tObjectiveFunctionValues[1] = tObjectiveFunctionValues[2];
            tObjectiveFunctionValues[2] = tTrialObjectiveValue;
            mNumIterationsDone++;
        }

        aStateMng.setCurrentObjectiveValue(tTrialObjectiveValue);
        aStateMng.setCurrentControl(mTrialControl.operator*());
    }

private:
    void checkCurrentStepValue()
    {
        const ScalarType tGamma = 0.1;
        if(mStepValues[2] > mContractionFactor * mStepValues[1])
        {
            mStepValues[2] = mContractionFactor * mStepValues[1];
        }
        else if(mStepValues[2] < tGamma * mStepValues[1])
        {
            mStepValues[2] = tGamma * mStepValues[1];
        }
        if(std::isfinite(mStepValues[2]) == false)
        {
            mStepValues[2] = tGamma * mStepValues[1];
        }
    }
    void interpolate(const std::vector<ScalarType> & aObjectiveFunctionValues)
    {
        ScalarType tPointOne = aObjectiveFunctionValues[2] - aObjectiveFunctionValues[0]
                - mStepValues[1] * mInitialGradientDotTrialStep;
        ScalarType tPointTwo = aObjectiveFunctionValues[1] - aObjectiveFunctionValues[0]
                - mStepValues[0] * mInitialGradientDotTrialStep;
        ScalarType tPointThree = static_cast<ScalarType>(1.) / (mStepValues[1] - mStepValues[0]);

        // find cubic unique minimum
        ScalarType tPointA = tPointThree
                * ((tPointOne / (mStepValues[1] * mStepValues[1])) - (tPointTwo / (mStepValues[0] * mStepValues[0])));
        ScalarType tPointB = tPointThree
                * ((tPointTwo * mStepValues[1] / (mStepValues[0] * mStepValues[0]))
                        - (tPointOne * mStepValues[0] / (mStepValues[1] * mStepValues[1])));
        ScalarType tPointC = tPointB * tPointB - static_cast<ScalarType>(3) * tPointA * mInitialGradientDotTrialStep;

        // cubic equation has unique minimum
        ScalarType tValueOne = (-tPointB + std::sqrt(tPointC)) / (static_cast<ScalarType>(3) * tPointA);
        // cubic equation is a quadratic
        ScalarType tValueTwo = -mInitialGradientDotTrialStep / (static_cast<ScalarType>(2) * tPointB);
        mStepValues[2] = tPointA != 0 ? mStepValues[2] = tValueOne : mStepValues[2] = tValueTwo;
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mContractionFactor;
    ScalarType mArmijoRuleConstant;
    ScalarType mStagnationTolerance;
    ScalarType mInitialGradientDotTrialStep;

    std::vector<ScalarType> mStepValues;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mProjectedTrialStep;

private:
    CubicLineSearch(const locus::CubicLineSearch<ScalarType, OrdinalType> & aRhs);
    locus::CubicLineSearch<ScalarType, OrdinalType> & operator=(const locus::CubicLineSearch<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class QuadraticLineSearch : public locus::LineSearch<ScalarType, OrdinalType>
{
public:
    explicit QuadraticLineSearch(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mStepLowerBound(1e-3),
            mStepUpperBound(0.5),
            mContractionFactor(0.5),
            mInitialTrialStepDotCurrentGradient(0),
            mStepValues(2, static_cast<ScalarType>(0)),
            mTrialControl(aDataFactory.control().create())
    {
    }
    virtual ~QuadraticLineSearch()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    ScalarType getStepValue() const
    {
        return (mStepValues[1]);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStepLowerBound(const ScalarType & aInput)
    {
        mStepLowerBound = aInput;
    }
    void setStepUpperBound(const ScalarType & aInput)
    {
        mStepUpperBound = aInput;
    }
    void setContractionFactor(const ScalarType & aInput)
    {
        mContractionFactor = aInput;
    }

    void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng)
    {
        OrdinalType tSize = 3;
        // tObjectiveFunction[0] = current trial value
        // tObjectiveFunction[1] = old trial value
        // tObjectiveFunction[2] = new trial value
        std::vector<ScalarType> tObjectiveFunction(tSize);
        tObjectiveFunction[0] = aStateMng.getCurrentObjectiveValue();

        ScalarType tNormTrialStep = locus::dot(aStateMng.getTrialStep(), aStateMng.getTrialStep());
        tNormTrialStep = std::sqrt(tNormTrialStep);
        mStepValues[1] = std::min(static_cast<ScalarType>(1),
                                  static_cast<ScalarType>(100) / (static_cast<ScalarType>(1) + tNormTrialStep));

        locus::update(static_cast<ScalarType>(1),
                      aStateMng.getCurrentControl(),
                      static_cast<ScalarType>(0),
                      *mTrialControl);
        locus::update(mStepValues[1], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

        tObjectiveFunction[2] = aStateMng.evaluateObjective(*mTrialControl);
        mInitialTrialStepDotCurrentGradient = locus::dot(aStateMng.getTrialStep(), aStateMng.getCurrentGradient());
        const ScalarType tAlpha = 1e-4;
        ScalarType tTargetObjectiveValue = tObjectiveFunction[0]
                - (tAlpha * mStepValues[1] * mInitialTrialStepDotCurrentGradient);

        mNumIterationsDone = 1;
        while(tObjectiveFunction[2] > tTargetObjectiveValue)
        {
            mStepValues[0] = mStepValues[1];
            ScalarType tStep = this->interpolate(tObjectiveFunction, mStepValues);
            mStepValues[1] = tStep;

            locus::update(static_cast<ScalarType>(1),
                          aStateMng.getCurrentControl(),
                          static_cast<ScalarType>(0),
                          *mTrialControl);
            locus::update(mStepValues[1], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

            tObjectiveFunction[1] = tObjectiveFunction[2];
            tObjectiveFunction[2] = aStateMng.evaluateObjective(*mTrialControl);

            mNumIterationsDone++;
            if(mNumIterationsDone >= mMaxNumIterations)
            {
                break;
            }
            tTargetObjectiveValue = tObjectiveFunction[0]
                    - (tAlpha * mStepValues[1] * mInitialTrialStepDotCurrentGradient);
        }

        aStateMng.setCurrentObjectiveValue(tObjectiveFunction[2]);
        aStateMng.setCurrentControl(*mTrialControl);
    }

private:
    ScalarType interpolate(const std::vector<ScalarType> & aObjectiveFunction, const std::vector<ScalarType> & aStepValues)
    {
        ScalarType tStepLowerBound = aStepValues[1] * mStepLowerBound;
        ScalarType tStepUpperBound = aStepValues[1] * mStepUpperBound;
        ScalarType tDenominator = static_cast<ScalarType>(2) * aStepValues[1]
                * (aObjectiveFunction[2] - aObjectiveFunction[0] - mInitialTrialStepDotCurrentGradient);

        ScalarType tStep = -mInitialTrialStepDotCurrentGradient / tDenominator;
        if(tStep < tStepLowerBound)
        {
            tStep = tStepLowerBound;
        }
        if(tStep > tStepUpperBound)
        {
            tStep = tStepUpperBound;
        }

        return (tStep);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mStepLowerBound;
    ScalarType mStepUpperBound;
    ScalarType mContractionFactor;
    ScalarType mInitialTrialStepDotCurrentGradient;

    // mStepValues[0] = old trial value
    // mStepValues[1] = new trial value
    std::vector<ScalarType> mStepValues;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;

private:
    QuadraticLineSearch(const locus::QuadraticLineSearch<ScalarType, OrdinalType> & aRhs);
    locus::QuadraticLineSearch<ScalarType, OrdinalType> & operator=(const locus::QuadraticLineSearch<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradient
{
public:
    NonlinearConjugateGradient(const std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> & aDataFactory,
                               const std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> & aDataMng,
                               const std::shared_ptr<locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType>> & aStageMng) :
            mMaxNumIterations(100),
            mNumIterationsDone(0),
            mGradientTolerance(1e-8),
            mStationarityTolerance(1e-8),
            mControlStagnationTolerance(std::numeric_limits<ScalarType>::epsilon()),
            mObjectiveStagnationTolerance(std::numeric_limits<ScalarType>::epsilon()),
            mStoppingCriteria(locus::algorithm::stop_t::NOT_CONVERGED),
            mControlWork(aDataFactory->control().create()),
            mTrialControl(aDataFactory->control().create()),
            mLineSearch(std::make_shared<locus::CubicLineSearch<ScalarType, OrdinalType>>(aDataFactory.operator*())),
            mStep(std::make_shared<locus::PolakRibiere<ScalarType, OrdinalType>>(aDataFactory.operator*())),
            mStateMng(std::make_shared<locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType>>(aDataMng, aStageMng))
    {
    }
    ~NonlinearConjugateGradient()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    locus::algorithm::stop_t getStoppingCriteria() const
    {
        return (mStoppingCriteria);
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setGradientTolerance(const ScalarType & aInput)
    {
        mGradientTolerance = aInput;
    }
    void setStationarityTolerance(const ScalarType & aInput)
    {
        mStationarityTolerance = aInput;
    }
    void setControlStagnationTolerance(const ScalarType & aInput)
    {
        mControlStagnationTolerance = aInput;
    }
    void setObjectiveStagnationTolerance(const ScalarType & aInput)
    {
        mObjectiveStagnationTolerance = aInput;
    }

    void setContractionFactor(const ScalarType & aInput)
    {
        mLineSearch->setContractionFactor(aInput);
    }
    void setDanielsMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::Daniels<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiLiaoMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiLiao<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiYuanMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiYuan<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setLiuStoreyMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::LiuStorey<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setHagerZhangMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::HagerZhang<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setPerryShannoMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::PerryShanno<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiYuanHybridMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiYuanHybrid<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setFletcherReevesMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::FletcherReeves<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setHestenesStiefelMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::HestenesStiefel<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setConjugateDescentMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::ConjugateDescent<ScalarType, OrdinalType>>(aDataFactory);
    }

    void solve()
    {
        assert(mStep.get() != nullptr);

        this->computeInitialState();
        // Perform first iteration (i.e. x_0)
        this->computeInitialDescentDirection();
        this->computeProjectedStep();
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        tDataMng.storePreviousState();

        mLineSearch->step(mStateMng.operator*());
        locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
        mStateMng->computeGradient(tDataMng.getCurrentControl(), mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
        this->computeProjectedGradient();

        bool tStop = false;
        if(this->checkStoppingCriteria() == true)
        {
            tStop = true;
        }

        mNumIterationsDone = 1;
        locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType> & tStageMng = mStateMng->getStageMng();
        while(tStop != true)
        {

            mStep->computeScaledDescentDirection(tDataMng, tStageMng);
            this->computeProjectedStep();
            tDataMng.storePreviousState();

            mLineSearch->step(mStateMng.operator*());
            locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
            mStateMng->computeGradient(tDataMng.getCurrentControl(), mControlWork.operator*());
            tDataMng.setCurrentGradient(mControlWork.operator*());
            this->computeProjectedGradient();

            mNumIterationsDone++;
            if(this->checkStoppingCriteria() == true)
            {
                tStop = true;
                break;
            }
        }
    }

private:
    void computeInitialState()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        ScalarType tValue = mStateMng->evaluateObjective(tControl);
        tDataMng.setCurrentObjectiveFunctionValue(tValue);

        mStateMng->computeGradient(tControl, mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
        this->computeProjectedGradient();
    }
    void computeProjectedStep()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(0), mTrialControl.operator*());
        locus::update(static_cast<ScalarType>(1),
                      tDataMng.getTrialStep(),
                      static_cast<ScalarType>(1),
                      mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = tDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = tDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());

        // Compute projected trial step
        locus::update(static_cast<ScalarType>(1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(-1), tControl, static_cast<ScalarType>(1), mControlWork.operator*());
        tDataMng.setTrialStep(mControlWork.operator*());
    }
    void computeProjectedGradient()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(0), mTrialControl.operator*());
        locus::update(static_cast<ScalarType>(1),
                      tDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = tDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = tDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());

        // Compute projected gradient
        locus::update(static_cast<ScalarType>(1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(-1), tControl, static_cast<ScalarType>(1), mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
    }
    void computeInitialDescentDirection()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        locus::update(static_cast<ScalarType>(-1),
                      tDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        tDataMng.setTrialStep(mControlWork.operator*());
    }
    bool checkStoppingCriteria()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();

        tDataMng.computeNormGradient();
        tDataMng.computeStagnationMeasure();
        tDataMng.computeStationarityMeasure();
        tDataMng.computeObjectiveStagnationMeasure();

        const ScalarType tNormGradient = tDataMng.getNormGradient();
        const ScalarType tStagnationMeasure = tDataMng.getStagnationMeasure();
        const ScalarType tStationarityMeasure = tDataMng.getStationarityMeasure();
        const ScalarType tObjectiveStagnationMeasure = tDataMng.getObjectiveStagnationMeasure();

        bool tStop = false;
        if(tStagnationMeasure < mControlStagnationTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::CONTROL_STAGNATION;
        }
        else if(tNormGradient < mGradientTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::NORM_GRADIENT;
        }
        else if(tStationarityMeasure < mStationarityTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::NORM_STEP;
        }
        else if(mNumIterationsDone >= mMaxNumIterations)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS;
        }
        else if(tObjectiveStagnationMeasure < mObjectiveStagnationTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::OBJECTIVE_STAGNATION;
        }

        return (tStop);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mGradientTolerance;
    ScalarType mStationarityTolerance;
    ScalarType mControlStagnationTolerance;
    ScalarType mObjectiveStagnationTolerance;

    locus::algorithm::stop_t mStoppingCriteria;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;

    std::shared_ptr<locus::LineSearch<ScalarType, OrdinalType>> mLineSearch;
    std::shared_ptr<locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>> mStep;
    std::shared_ptr<locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType>> mStateMng;

private:
    NonlinearConjugateGradient(const locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & aRhs);
};

/**********************************************************************************************************/
/************************* CONSERVATIVE CONVEX SEPARABLE APPROXIMATION ALGORITHM **************************/
/**********************************************************************************************************/

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxDataMng
{
public:
    explicit ConservativeConvexSeparableAppxDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mIsInitialGuessSet(false),
            mStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mFeasibilityMeasure(std::numeric_limits<ScalarType>::max()),
            mStationarityMeasure(std::numeric_limits<ScalarType>::max()),
            mNormProjectedGradient(std::numeric_limits<ScalarType>::max()),
            mObjectiveStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mDualProblemBoundsScaleFactor(0.5),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mDualObjectiveGlobalizationFactor(1),
            mKarushKuhnTuckerConditionsInexactness(std::numeric_limits<ScalarType>::max()),
            mDualWorkOne(),
            mDualWorkTwo(),
            mControlWorkOne(),
            mControlWorkTwo(),
            mDual(aDataFactory.dual().create()),
            mTrialStep(aDataFactory.control().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mCurrentSigma(aDataFactory.control().create()),
            mCurrentControl(aDataFactory.control().create()),
            mPreviousControl(aDataFactory.control().create()),
            mControlLowerBounds(aDataFactory.control().create()),
            mControlUpperBounds(aDataFactory.control().create()),
            mControlWorkMultiVector(aDataFactory.control().create()),
            mCurrentConstraintValues(aDataFactory.dual().create()),
            mCurrentObjectiveGradient(aDataFactory.control().create()),
            mConstraintGlobalizationFactors(aDataFactory.dual().create()),
            mDualReductions(aDataFactory.getDualReductionOperations().create()),
            mControlReductions(aDataFactory.getControlReductionOperations().create()),
            mCurrentConstraintGradients(std::make_shared<locus::MultiVectorList<ScalarType, OrdinalType>>())
    {
        this->initialize();
    }
    ~ConservativeConvexSeparableAppxDataMng()
    {
    }

    bool isInitialGuessSet() const
    {
        return (mIsInitialGuessSet);
    }

    // NOTE: NUMBER OF CONTROL VECTORS
    OrdinalType getNumControlVectors() const
    {
        return (mCurrentControl->getNumVectors());
    }
    // NOTE: NUMBER OF DUAL VECTORS
    OrdinalType getNumDualVectors() const
    {
        return (mDual->getNumVectors());
    }
    // NOTE :GET NUMBER OF CONSTRAINTS
    OrdinalType getNumConstraints() const
    {
        OrdinalType tNumConstraints = mCurrentConstraintGradients.size();
        return (tNumConstraints);
    }

    // NOTE: DUAL PROBLEM PARAMETERS
    ScalarType getDualProblemBoundsScaleFactor() const
    {
        return (mDualProblemBoundsScaleFactor);
    }
    void setDualProblemBoundsScaleFactor(const ScalarType & aInput)
    {
        mDualProblemBoundsScaleFactor = aInput;
    }
    ScalarType getDualObjectiveGlobalizationFactor() const
    {
        return (mDualObjectiveGlobalizationFactor);
    }
    void setDualObjectiveGlobalizationFactor(const ScalarType & aInput) const
    {
        mDualObjectiveGlobalizationFactor = aInput;
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintGlobalizationFactors() const
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        return (mConstraintGlobalizationFactors.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintGlobalizationFactors(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintGlobalizationFactors->getNumVectors());
        return (mConstraintGlobalizationFactors->operator [](aVectorIndex));
    }
    void setConstraintGlobalizationFactors(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintGlobalizationFactors->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintGlobalizationFactors);
    }
    void setConstraintGlobalizationFactors(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintGlobalizationFactors->getNumVectors());
        mConstraintGlobalizationFactors->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: OBJECTIVE FUNCTION VALUE
    ScalarType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ScalarType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    ScalarType getPreviousObjectiveFunctionValue() const
    {
        return (mPreviousObjectiveFunctionValue);
    }
    void setPreviousObjectiveFunctionValue(const ScalarType & aInput)
    {
        mPreviousObjectiveFunctionValue = aInput;
    }

    // NOTE: SET INITIAL GUESS
    void setInitialGuess(const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }

    // NOTE: DUAL VECTOR
    const locus::MultiVector<ScalarType, OrdinalType> & getDual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getDual(const OrdinalType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        return (mDual->operator [](aVectorIndex));
    }
    void setDual(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mDual->getNumVectors());
        locus::update(1., aInput, 0., *mDual);
    }
    void setDual(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        mDual->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: TRIAL STEP FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const
    {
        assert(mTrialStep.get() != nullptr);

        return (mTrialStep.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getTrialStep(const OrdinalType & aVectorIndex) const
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());

        return (mTrialStep->operator [](aVectorIndex));
    }
    void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mTrialStep->getNumVectors());
        locus::update(1., aInput, 0., *mTrialStep);
    }
    void setTrialStep(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());

        mTrialStep->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: ACTIVE SET FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getActiveSet() const
    {
        assert(mActiveSet.get() != nullptr);

        return (mActiveSet.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getActiveSet(const OrdinalType & aVectorIndex) const
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        return (mActiveSet->operator [](aVectorIndex));
    }
    void setActiveSet(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mActiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mActiveSet);
    }
    void setActiveSet(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        mActiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: INACTIVE SET FUNCTIONS
    const locus::MultiVector<ScalarType, OrdinalType> & getInactiveSet() const
    {
        assert(mInactiveSet.get() != nullptr);

        return (mInactiveSet.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getInactiveSet(const OrdinalType & aVectorIndex) const
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        return (mInactiveSet->operator [](aVectorIndex));
    }
    void setInactiveSet(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mInactiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mInactiveSet);
    }
    void setInactiveSet(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        mInactiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);

        return (mCurrentControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentControl(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mCurrentControl);
    }
    void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS CONTROL
    const locus::MultiVector<ScalarType, OrdinalType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);

        return (mPreviousControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getPreviousControl(const OrdinalType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousControl->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousControl);
    }
    void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        mPreviousControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT OBJECTIVE GRADIENT
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentObjectiveGradient() const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);

        return (mCurrentObjectiveGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentObjectiveGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentObjectiveGradient->getNumVectors());

        return (mCurrentObjectiveGradient->operator [](aVectorIndex));
    }
    void setCurrentObjectiveGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentObjectiveGradient->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentObjectiveGradient);
    }
    void setCurrentObjectiveGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentObjectiveGradient->getNumVectors());

        mCurrentObjectiveGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT SIGMA VECTOR
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentSigma() const
    {
        assert(mCurrentSigma.get() != nullptr);

        return (mCurrentSigma.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentSigma(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentSigma.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentSigma->getNumVectors());

        return (mCurrentSigma->operator [](aVectorIndex));
    }
    void setCurrentSigma(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentSigma->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentSigma);
    }
    void setCurrentSigma(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentSigma.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentSigma->getNumVectors());

        mCurrentSigma->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);

        return (mControlLowerBounds.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlLowerBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        return (mControlLowerBounds->operator [](aVectorIndex));
    }
    void setControlLowerBounds(const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mControlLowerBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlLowerBounds);
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);

        return (mControlUpperBounds.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getControlUpperBounds(const OrdinalType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        return (mControlUpperBounds->operator [](aVectorIndex));
    }
    void setControlUpperBounds(const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mControlUpperBounds->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlUpperBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlUpperBounds);
    }

    // NOTE: CURRENT CONSTRAINT VALUES
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentConstraintValues() const
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        return (mCurrentConstraintValues.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentConstraintValues(const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentConstraintValues->getNumVectors());
        return (mCurrentConstraintValues->operator [](aVectorIndex));
    }
    void setCurrentConstraintValues(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentConstraintValues->getNumVectors());
        locus::update(static_cast<ScalarType>(1), aInput, static_cast<ScalarType>(0), *mCurrentConstraintValues);
    }
    void setCurrentConstraintValues(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentConstraintValues->getNumVectors());
        mCurrentConstraintValues->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONSTRAINT GRADIENTS
    const locus::MultiVectorList<ScalarType, OrdinalType> & getCurrentConstraintGradients() const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        return (mCurrentConstraintGradients.operator*());
    }
    const locus::MultiVector<ScalarType, OrdinalType> & getCurrentConstraintGradients(const OrdinalType & aConstraintIndex) const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);

        return (mCurrentConstraintGradients->operator[](aConstraintIndex));
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentConstraintGradients(const OrdinalType & aConstraintIndex,
                                                                                const OrdinalType & aVectorIndex) const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);
        assert(aVectorIndex < mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());

        return (mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex));
    }
    void getCurrentConstraintGradients(locus::MultiVectorList<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aInput.size() == mCurrentConstraintGradients->size());

        const OrdinalType tNumConstraints = aInput.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            assert(aInput[tConstraintIndex].get() != nullptr);
            assert(mCurrentConstraintGradients->ptr(tConstraintIndex).get() != nullptr);
            locus::update(static_cast<ScalarType>(1),
                          mCurrentConstraintGradients->operator[](tConstraintIndex),
                          static_cast<ScalarType>(0),
                          aInput[tConstraintIndex]);
        }
    }
    void setCurrentConstraintGradients(const OrdinalType & aConstraintIndex,
                                       const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());

        locus::update(static_cast<ScalarType>(1),
                      aInput,
                      static_cast<ScalarType>(0),
                      mCurrentConstraintGradients->operator[](aConstraintIndex));
    }
    void setCurrentConstraintGradients(const OrdinalType & aConstraintIndex,
                                       const OrdinalType & aVectorIndex,
                                       const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex >= static_cast<OrdinalType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(aVectorIndex < mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());
        assert(aInput.size() == mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex).size());

        const ScalarType tAlpha = 1;
        const ScalarType tBeta = 0;
        mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex).update(tAlpha, aInput, tBeta);
    }

    // NOTE: STAGNATION MEASURE CRITERION
    void computeStagnationMeasure()
    {
        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkOne->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ScalarType, OrdinalType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkOne->update(-1., tMyPreviousControl, 1.);
            mControlWorkOne->modulus();
            storage[tIndex] = mControlReductions->max(*mControlWorkOne);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: NORM OF CURRENT PROJECTED GRADIENT
    ScalarType computeProjectedVectorNorm(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = aInput.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyInputVector = aInput[tIndex];

            mControlWorkOne->update(1., tMyInputVector, 0.);
            mControlWorkOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkOne->dot(*mControlWorkOne);
        }
        ScalarType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormProjectedGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = mCurrentObjectiveGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = (*mCurrentObjectiveGradient)[tIndex];

            mControlWorkOne->update(1., tMyGradient, 0.);
            mControlWorkOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkOne->dot(*mControlWorkOne);
        }
        mNormProjectedGradient = std::sqrt(tCummulativeDotProduct);
    }
    ScalarType getNormProjectedGradient() const
    {
        return (mNormProjectedGradient);
    }

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeObjectiveStagnationMeasure()
    {
        mObjectiveStagnationMeasure = mPreviousObjectiveFunctionValue - mCurrentObjectiveFunctionValue;
        mObjectiveStagnationMeasure = std::abs(mObjectiveStagnationMeasure);
    }
    ScalarType getObjectiveStagnationMeasure() const
    {
        return (mObjectiveStagnationMeasure);
    }

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeFeasibilityMeasure()
    {
        const OrdinalType tNumVectors = mCurrentConstraintValues->getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors, static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tDual = mDual->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tWork = mDualWorkOne->operator[](tVectorIndex);
            locus::update(static_cast<ScalarType>(1), tDual, static_cast<ScalarType>(0), tWork);
            tWork.modulus();
            tStorage[tVectorIndex] = mDualReductions->max(tWork);
        }
        const ScalarType tInitialValue = 0;
        mFeasibilityMeasure = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
    }
    ScalarType getFeasibilityMeasure() const
    {
        return (mFeasibilityMeasure);
    }

    // NOTE: STATIONARITY MEASURE CALCULATION
    void computeStationarityMeasure()
    {
        assert(mInactiveSet.get() != nullptr);
        assert(mCurrentControl.get() != nullptr);
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlUpperBounds.get() != nullptr);
        assert(mCurrentObjectiveGradient.get() != nullptr);

        locus::update(static_cast<ScalarType>(1), *mCurrentControl, static_cast<ScalarType>(0), *mControlWorkMultiVector);
        locus::update(static_cast<ScalarType>(-1), *mCurrentObjectiveGradient, static_cast<ScalarType>(1), *mControlWorkMultiVector);
        locus::bounds::project(*mControlLowerBounds, *mControlUpperBounds, *mControlWorkMultiVector);
        locus::update(static_cast<ScalarType>(1), *mCurrentControl, static_cast<ScalarType>(-1), *mControlWorkMultiVector);
        locus::update(static_cast<ScalarType>(1), *mControlWorkMultiVector, static_cast<ScalarType>(0), *mTrialStep);

        locus::entryWiseProduct(*mInactiveSet, *mControlWorkMultiVector);
        mStationarityMeasure = locus::norm(*mControlWorkMultiVector);
    }
    ScalarType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    /*! Check inexactness in the Karush-Kuhn-Tucker (KKT) conditions (i.e. KKT residual) and compute
     * the norm of the KKT residual, where r(x,\lambda) = \{C1, C2, C3, C4\} denotes the residual vector
     * and C# denotes the corresponding Condition. The KKT conditions are given by:
     *
     * Condition 1: \left(1 + x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{+} = 0,\quad{j}=1,\dots,n_x
     * Condition 2: \left(1 - x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{-} = 0,\quad{j}=1,\dots,n_x
     * Condition 3: f_i(x)^{+} = 0,\quad{i}=1,\dots,N_c
     * Condition 4: \lambda_{i}f_i(x)^{-} = 0,\quad{i}=1,\dots,N_c.
     *
     * The nomenclature is given as follows: x denotes the control vector, \lambda denotes the dual
     * vector, N_c is the number of constraints, n_x is the number of controls, f_0 is the objective
     * function and f_i is the i-th constraint. Finally, a^{+} = max{0, a} and a^{-} = max{0, −a}.
     **/
    void computeKarushKuhnTuckerConditionsInexactness(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                                      const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aDual.getNumVectors() == mDual->getNumVectors());
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));
        assert(aDual[tVectorIndex].size() == mDual->operator[](tVectorIndex).size());
        assert(aControl.getNumVectors() == mCurrentControl->getNumVectors());
        assert(aControl[tVectorIndex].size() == mCurrentControl->operator[](tVectorIndex).size());

        locus::fill(static_cast<ScalarType>(0), mControlWorkMultiVector.operator*());
        const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tVectorIndex];
        const OrdinalType tNumConstraints = tDual.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> tConstraintGradients =
                    mCurrentConstraintGradients[tConstraintIndex].operator*();
            locus::MultiVector<ScalarType, OrdinalType> tConstraintGradientsTimesDual =
                    mControlWorkMultiVector.operator*();
            locus::update(tDual[tConstraintIndex],
                          tConstraintGradients,
                          static_cast<ScalarType>(1),
                          tConstraintGradientsTimesDual);
        }

        ScalarType tConditioneOne = std::numeric_limits<ScalarType>::max();
        ScalarType tConditioneTwo = std::numeric_limits<ScalarType>::max();
        this->computeConditionsOneAndTwo(aControl, aDual, tConditioneOne, tConditioneTwo);

        const ScalarType tConditioneThree = std::numeric_limits<ScalarType>::max();
        const ScalarType tConditioneFour = std::numeric_limits<ScalarType>::max();
        this->computeConditionsThreeAndFour(aControl, aDual, tConditioneThree, tConditioneFour);

        ScalarType tNumControls = aControl[tVectorIndex].size();
        ScalarType tSum = tConditioneOne + tConditioneTwo + tConditioneThree + tConditioneFour;
        mKarushKuhnTuckerConditionsInexactness = (static_cast<ScalarType>(1) / tNumControls) * std::sqrt(tSum);
    }
    ScalarType getKarushKuhnTuckerConditionsInexactness() const
    {
        return (mKarushKuhnTuckerConditionsInexactness);
    }


private:
    void initialize()
    {
        const OrdinalType tControlVectorIndex = 0;
        mControlWorkOne = mCurrentControl->operator[](tControlVectorIndex).create();
        mControlWorkTwo = mCurrentControl->operator[](tControlVectorIndex).create();
        locus::fill(static_cast<ScalarType>(0), *mActiveSet);
        locus::fill(static_cast<ScalarType>(1), *mInactiveSet);

        assert(mDual->getNumVectors() == static_cast<OrdinalType>(1));
        const OrdinalType tDualVectorIndex = 0;
        mDualWorkOne = mDual->operator[](tDualVectorIndex).create();
        mDualWorkTwo = mDual->operator[](tDualVectorIndex).create();

        const OrdinalType tNumConstraints = mDual->operator[](tDualVectorIndex).size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            mCurrentConstraintGradients->add(mCurrentControl.operator*());
        }

        ScalarType tScalarValue = std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlUpperBounds);
        tScalarValue = -std::numeric_limits<ScalarType>::max();
        locus::fill(tScalarValue, *mControlLowerBounds);

        tScalarValue = 1;
        locus::fill(tScalarValue, mConstraintGlobalizationFactors.operator*());
    }
    /*!
     * Compute the following Karush-Kuhn-Tucker (KKT) conditions:
     *
     * Condition 3: f_i(x)^{+} = 0,\quad{i}=1,\dots,N_c
     * Condition 4: \lambda_{i}f_i(x)^{-} = 0,\quad{i}=1,\dots,N_c.
     *
     * where the nomenclature is given as follows: \lambda denotes the dual vector, N_c is the
     * number of constraints, n_x is the number of controls and f_i is the i-th constraint.
     * Finally, a^{+} = max{0, a} and a^{-} = max{0, −a}.
     **/
    void computeConditionsOneAndTwo(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                    const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                                    ScalarType & aConditionOne,
                                    ScalarType & aConditionTwo)
    {
        const OrdinalType tNumControlVectors = aControl.getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumControlVectors, static_cast<ScalarType>(0));
        std::vector<ScalarType> tStorageTwo(tNumControlVectors, static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tWorkOne = mControlWorkOne.operator*();
            locus::Vector<ScalarType, OrdinalType> & tWorkTwo = mControlWorkTwo.operator*();
            const locus::Vector<ScalarType, OrdinalType> & tControl = aControl[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tObjectiveGradient =
                    mCurrentObjectiveGradient->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tConstraintGradientTimesDual =
                    mControlWorkMultiVector->operator[](tVectorIndex);

            const OrdinalType tNumControls = tControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                tWorkOne[tControlIndex] = tObjectiveGradient[tControlIndex] + tConstraintGradientTimesDual[tControlIndex];
                tWorkOne[tControlIndex] = std::max(static_cast<ScalarType>(0), tWorkOne[tControlIndex]);
                tWorkOne[tControlIndex] = (static_cast<ScalarType>(1) + tControl[tControlIndex]) * tWorkOne[tControlIndex];
                tWorkOne[tControlIndex] = tWorkOne[tControlIndex] * tWorkOne[tControlIndex];

                tWorkTwo[tControlIndex] = tObjectiveGradient[tControlIndex] + tConstraintGradientTimesDual[tControlIndex];
                tWorkTwo[tControlIndex] = std::max(static_cast<ScalarType>(0), -tWorkTwo[tControlIndex]);
                tWorkTwo[tControlIndex] = (static_cast<ScalarType>(1) - tControl[tControlIndex]) * tWorkTwo[tControlIndex];
                tWorkTwo[tControlIndex] = tWorkTwo[tControlIndex] * tWorkTwo[tControlIndex];
            }

            tStorageOne[tVectorIndex] = mControlReductions->sum(tWorkOne);
            tStorageTwo[tVectorIndex] = mControlReductions->sum(tWorkTwo);
        }

        const ScalarType tInitialValue = 0;
        aConditionOne = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        aConditionTwo = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
    }
    /*!
     * Compute the following Karush-Kuhn-Tucker (KKT) conditions:
     *
     * Condition 1: \left(1 + x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{+} = 0,\quad{j}=1,\dots,n_x
     * Condition 2: \left(1 - x_j\right)\left(\frac{\partial{f}_0}{\partial{x}_j} + \sum_{i=1}^{N_c}
     *              \lambda_i\frac{\partial{f}_i}{\partial{x}_j}\right)^{-} = 0,\quad{j}=1,\dots,n_x,
     *
     * where the nomenclature is given as follows: x denotes the control vector, \lambda denotes
     * the dual vector, N_c is the number of constraints, n_x is the number of controls, f_0 is
     * the objective function and f_i is the i-th constraint. Finally, a^{+} = max{0, a} and a^{-}
     * = max{0, −a}.
     **/
    void computeConditionsThreeAndFour(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                    const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                                    ScalarType & aConditionThree,
                                    ScalarType & aConditionFour)
    {
        const OrdinalType tNumDualVectors = aDual.getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumDualVectors, static_cast<ScalarType>(0));
        std::vector<ScalarType> tStorageTwo(tNumDualVectors, static_cast<ScalarType>(0));
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tWorkOne = mDualWorkOne.operator*();
            locus::Vector<ScalarType, OrdinalType> & tWorkTwo = mDualWorkTwo.operator*();
            const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tConstraintValues =
                    mCurrentConstraintValues->operator[](tVectorIndex);

            const OrdinalType tNumConstraints = tDual.size();
            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                tWorkOne[tConstraintIndex] = std::max(static_cast<ScalarType>(0), tConstraintValues[tConstraintIndex]);
                tWorkOne[tConstraintIndex] = tWorkOne[tConstraintIndex] * tWorkOne[tConstraintIndex];

                tWorkTwo[tConstraintIndex] = tDual[tConstraintIndex]
                        * std::max(static_cast<ScalarType>(0), -tConstraintValues[tConstraintIndex]);
                tWorkTwo[tConstraintIndex] = tWorkTwo[tConstraintIndex] * tWorkTwo[tConstraintIndex];
            }

            tStorageOne[tVectorIndex] = mDualReductions->sum(tWorkOne);
            tStorageTwo[tVectorIndex] = mDualReductions->sum(tWorkTwo);
        }

        const ScalarType tInitialValue = 0;
        aConditionThree = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        aConditionFour = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
    }

private:
    bool mIsInitialGuessSet;

    ScalarType mStagnationMeasure;
    ScalarType mFeasibilityMeasure;
    ScalarType mStationarityMeasure;
    ScalarType mNormProjectedGradient;
    ScalarType mObjectiveStagnationMeasure;
    ScalarType mDualProblemBoundsScaleFactor;
    ScalarType mCurrentObjectiveFunctionValue;
    ScalarType mPreviousObjectiveFunctionValue;
    ScalarType mDualObjectiveGlobalizationFactor;
    ScalarType mKarushKuhnTuckerConditionsInexactness;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkTwo;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentSigma;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWorkMultiVector;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintValues;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintGlobalizationFactors;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductions;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductions;

    std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> mCurrentConstraintGradients;

private:
    ConservativeConvexSeparableAppxDataMng(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxStageMng
{
public:
    virtual ~ConservativeConvexSeparableAppxStageMng()
    {
    }

    virtual void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class PrimalProblemStageMng : public locus::ConservativeConvexSeparableAppxStageMng<ScalarType, OrdinalType>
{
public:
    PrimalProblemStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumConstraintEvaluations(),
            mNumConstraintGradientEvaluations(),
            mState(aDataFactory.state().create()),
            mObjective(),
            mConstraints(),
            mObjectiveGradient(),
            mConstraintGradients(),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory))
    {
    }
    PrimalProblemStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory,
                          const locus::Criterion<ScalarType, OrdinalType> & aObjective,
                          const locus::CriterionList<ScalarType, OrdinalType> & aConstraints) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumConstraintEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mNumConstraintGradientEvaluations(std::vector<OrdinalType>(aConstraints.size())),
            mState(aDataFactory.state().create()),
            mObjective(aObjective.create()),
            mConstraints(aConstraints.create()),
            mObjectiveGradient(),
            mConstraintGradients(),
            mStateData(std::make_shared<locus::StateData<ScalarType, OrdinalType>>(aDataFactory))
    {
    }
    virtual ~PrimalProblemStageMng()
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

    void setObjectiveGradient(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mObjectiveGradient = aInput.create();
    }
    void setConstraintGradients(const locus::GradientOperatorList<ScalarType, OrdinalType> & aInput)
    {
        mConstraintGradients = aInput.create();
    }

    void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mStateData->setCurrentTrialStep(aDataMng.getTrialStep());
        mStateData->setCurrentControl(aDataMng.getCurrentControl());
        mStateData->setCurrentObjectiveGradient(aDataMng.getCurrentObjectiveGradient());
        mStateData->setCurrentObjectiveFunctionValue(aDataMng.getCurrentObjectiveFunctionValue());

        mObjectiveGradient->update(mStateData.operator*());

        const OrdinalType tNumConstraints = mConstraintGradients->size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintGradient =
                    aDataMng.getCurrentConstraintGradients(tConstraintIndex);
            mStateData->setCurrentConstraintGradient(tMyConstraintGradient);
            mConstraintGradients->operator[](tConstraintIndex).update(mStateData.operator*());
        }
    }
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(mObjective.get() != nullptr);

        ScalarType tObjectiveFunctionValue = mObjective->value(mState.operator*(), aControl);
        mNumObjectiveFunctionEvaluations++;

        return (tObjectiveFunctionValue);
    }
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mObjectiveGradient.get() != nullptr);
        locus::fill(static_cast<ScalarType>(0), aOutput);
        mObjectiveGradient->compute(mState.operator*(), aControl, aOutput);
        mNumObjectiveGradientEvaluations++;
    }
    void evaluateConstraints(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                             locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(mConstraints.get() != nullptr);

        locus::fill(static_cast<ScalarType>(0), aOutput);
        const OrdinalType tNumVectors = aOutput.getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyOutput = aOutput[tVectorIndex];
            const OrdinalType tNumConstraints = tMyOutput.size();
            assert(tNumConstraints == mConstraints->size());

            for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                assert(mConstraints->ptr(tConstraintIndex).get() != nullptr);

                tMyOutput[tConstraintIndex] = mConstraints->operator[](tConstraintIndex).value(*mState, aControl);
                mNumConstraintEvaluations[tConstraintIndex] =
                        mNumConstraintEvaluations[tConstraintIndex] + static_cast<OrdinalType>(1);
            }
        }
    }
    void computeConstraintGradients(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                    locus::MultiVectorList<ScalarType, OrdinalType> & aOutput)
    {
        assert(mConstraintGradients.get() != nullptr);
        assert(mConstraintGradients->size() == aOutput.size());

        const OrdinalType tNumConstraints = aOutput.size();
        for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            assert(mConstraints->ptr(tIndex).get() != nullptr);

            locus::MultiVector<ScalarType, OrdinalType> & tMyOutput = aOutput[tIndex];
            locus::fill(static_cast<ScalarType>(0), tMyOutput);
            mConstraints->operator[](tIndex).computeConstraintGradients(*mState, aControl, tMyOutput);
            mNumConstraintGradientEvaluations[tIndex] =
                    mNumConstraintGradientEvaluations[tIndex] + static_cast<OrdinalType>(1);
        }
    }

private:
    OrdinalType mNumObjectiveFunctionEvaluations;
    OrdinalType mNumObjectiveGradientEvaluations;

    std::vector<OrdinalType> mNumConstraintEvaluations;
    std::vector<OrdinalType> mNumConstraintGradientEvaluations;

    std::shared_ptr<MultiVector<ScalarType, OrdinalType>> mState;
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> mObjective;
    std::shared_ptr<locus::CriterionList<ScalarType, OrdinalType>> mConstraints;
    std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> mObjectiveGradient;
    std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> mConstraintGradients;

    std::shared_ptr<locus::StateData<ScalarType, OrdinalType>> mStateData;

private:
    PrimalProblemStageMng(const locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aRhs);
    locus::PrimalProblemStageMng<ScalarType, OrdinalType> & operator=(const locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class DualProblemStageMng : public locus::NonlinearConjugateGradientStageMng<ScalarType, OrdinalType>
{
public:
    explicit DualProblemStageMng(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mEpsilon(1e-6),
            mObjectiveCoefficientA(1),
            mObjectiveCoefficientR(1),
            mTrialAuxiliaryVariableZ(0),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mDualWorkVector(),
            mControlWorkVectorOne(),
            mControlWorkVectorTwo(),
            mTermA(aDataFactory.control().create()),
            mTermB(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mLowerAsymptotes(aDataFactory.control().create()),
            mUpperAsymptotes(aDataFactory.control().create()),
            mDualTimesCoefficientsP(aDataFactory.control().create()),
            mDualTimesCoefficientsQ(aDataFactory.control().create()),
            mObjectiveCoefficientsP(aDataFactory.control().create()),
            mObjectiveCoefficientsQ(aDataFactory.control().create()),
            mTrialControlLowerBounds(aDataFactory.control().create()),
            mTrialControlUpperBounds(aDataFactory.control().create()),
            mConstraintCoefficientsA(aDataFactory.dual().create()),
            mConstraintCoefficientsC(aDataFactory.dual().create()),
            mConstraintCoefficientsD(aDataFactory.dual().create()),
            mConstraintCoefficientsR(aDataFactory.dual().create()),
            mTrialAuxiliaryVariableY(aDataFactory.dual().create()),
            mCurrentConstraintValues(aDataFactory.dual().create()),
            mDualReductionOperations(aDataFactory.getDualReductionOperations().create()),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create()),
            mConstraintCoefficientsP(),
            mConstraintCoefficientsQ()
    {
        this->initialize(aDataFactory);
    }
    virtual ~DualProblemStageMng()
    {
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS A
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsA() const
    {
        assert(mConstraintCoefficientsA.get() != nullptr);

        return (mConstraintCoefficientsA.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintCoefficientsA(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        return (mConstraintCoefficientsA->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsA(const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(mConstraintCoefficientsA->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mConstraintCoefficientsA->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsA->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsA(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        mConstraintCoefficientsA->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsA(const OrdinalType & aVectorIndex,
                                    const OrdinalType & aElementIndex,
                                    const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());
        assert(aElementIndex >= static_cast<OrdinalType>(0));
        assert(aElementIndex < mConstraintCoefficientsA->operator [](aVectorIndex).size());

        mConstraintCoefficientsA->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsA(const OrdinalType & aVectorIndex,
                                    const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        mConstraintCoefficientsA->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsA(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsA->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsA);
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS C
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsC() const
    {
        assert(mConstraintCoefficientsC.get() != nullptr);

        return (mConstraintCoefficientsC.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintCoefficientsC(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        return (mConstraintCoefficientsC->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsC(const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(mConstraintCoefficientsC->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mConstraintCoefficientsC->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsC->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsC(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        mConstraintCoefficientsC->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsC(const OrdinalType & aVectorIndex,
                                    const OrdinalType & aElementIndex,
                                    const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());
        assert(aElementIndex >= static_cast<OrdinalType>(0));
        assert(aElementIndex < mConstraintCoefficientsC->operator [](aVectorIndex).size());

        mConstraintCoefficientsC->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsC(const OrdinalType & aVectorIndex,
                                    const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        mConstraintCoefficientsC->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsC(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsC->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsC);
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS D
    const locus::MultiVector<ScalarType, OrdinalType> & getConstraintCoefficientsD() const
    {
        assert(mConstraintCoefficientsD.get() != nullptr);

        return (mConstraintCoefficientsD.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getConstraintCoefficientsD(const OrdinalType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        return (mConstraintCoefficientsD->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsD(const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(mConstraintCoefficientsD->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mConstraintCoefficientsD->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsD->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsD(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        mConstraintCoefficientsD->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsD(const OrdinalType & aVectorIndex,
                                    const OrdinalType & aElementIndex,
                                    const ScalarType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());
        assert(aElementIndex >= static_cast<OrdinalType>(0));
        assert(aElementIndex < mConstraintCoefficientsD->operator [](aVectorIndex).size());

        mConstraintCoefficientsD->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsD(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        mConstraintCoefficientsD->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsD(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsD->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsD);
    }
    // UPDATE DUAL PROBLEM DATA
    void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        // Update Current Objective Function Value
        mCurrentObjectiveFunctionValue = aDataMng.getCurrentObjectiveFunctionValue();

        // Update Current Constraint Values
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getCurrentConstraintValues(),
                      static_cast<ScalarType>(0),
                      mCurrentConstraintValues.operator*());

        // Update Moving Asymptotes
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma();
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mLowerAsymptotes);
        locus::update(static_cast<ScalarType>(-1), tCurrentSigma, static_cast<ScalarType>(1), *mLowerAsymptotes);
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mUpperAsymptotes);
        locus::update(static_cast<ScalarType>(1), tCurrentSigma, static_cast<ScalarType>(1), *mUpperAsymptotes);

        // Update Trial Control Bounds
        const ScalarType tScaleFactor = aDataMng.getDualProblemBoundsScaleFactor();
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mTrialControlLowerBounds);
        locus::update(-tScaleFactor, tCurrentSigma, static_cast<ScalarType>(1), *mTrialControlLowerBounds);
        locus::update(static_cast<ScalarType>(1), tCurrentControl, static_cast<ScalarType>(0), *mTrialControlUpperBounds);
        locus::update(tScaleFactor, tCurrentSigma, static_cast<ScalarType>(1), *mTrialControlUpperBounds);
    }

    // UPDATE PRIMAL PROBLEM DATA
    void update(const locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        return;
    }
    // EVALUATE DUAL OBJECTIVE FUNCTION
    ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                                 ScalarType aTolerance = std::numeric_limits<ScalarType>::max())
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        this->computeDualTimesConstraintCoefficientTerms(aDual);
        this->computeTrialControl(aDual);
        this->computeTrialAuxiliaryVariables(aDual);

        const ScalarType tObjectiveTerm = mObjectiveCoefficientR + (mTrialAuxiliaryVariableZ * mObjectiveCoefficientA)
                + (mEpsilon * mTrialAuxiliaryVariableZ * mTrialAuxiliaryVariableZ);

        const ScalarType tConstraintSummationTerm = this->computeConstraintContribution(aDual);

        ScalarType tMovingAsymptotesTerm = this->computeMovingAsymptotesContribution();

        // Add all contributions to dual objective function
        ScalarType tOutput = static_cast<ScalarType>(-1)
                * (tObjectiveTerm + tConstraintSummationTerm + tMovingAsymptotesTerm);

        return (tOutput);
    }
    // COMPUTE DUAL GRADIENT
    void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                         locus::MultiVector<ScalarType, OrdinalType> & aDualGradient)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tDualVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tDualGradient = aDualGradient[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tAuxiliaryVariableY = (*mTrialAuxiliaryVariableY)[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tCoefficientsR = (*mConstraintCoefficientsR)[tDualVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tCoefficientsA = (*mConstraintCoefficientsA)[tDualVectorIndex];

        OrdinalType tNumConstraints = tDual.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            tDualGradient[tConstraintIndex] = tCoefficientsR[tConstraintIndex] - tAuxiliaryVariableY[tConstraintIndex]
                    - (tCoefficientsA[tConstraintIndex] * mTrialAuxiliaryVariableZ);

            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();

            const OrdinalType tNumControlVectors = mTrialControl->getNumVectors();
            std::vector<ScalarType> tMyStorageOne(tNumControlVectors);
            std::vector<ScalarType> tMyStorageTwo(tNumControlVectors);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
                mControlWorkVectorTwo->fill(static_cast<ScalarType>(0));
                const locus::Vector<ScalarType, OrdinalType> & tMyTrialControl = (*mTrialControl)[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyLowerAsymptotes = (*mLowerAsymptotes)[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyUpperAsymptotes = (*mUpperAsymptotes)[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsP = tMyConstraintCoefficientsP[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsQ = tMyConstraintCoefficientsQ[tVectorIndex];

                const OrdinalType tNumControls = tMyTrialControl.size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    (*mControlWorkVectorOne)[tControlIndex] = tMyCoefficientsP[tControlIndex]
                            / (tMyUpperAsymptotes[tControlIndex] - tMyTrialControl[tControlIndex]);

                    (*mControlWorkVectorTwo)[tControlIndex] = tMyCoefficientsQ[tControlIndex]
                            / (tMyTrialControl[tControlIndex] - tMyLowerAsymptotes[tControlIndex]);
                }

                tMyStorageOne[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
                tMyStorageTwo[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorTwo.operator*());
            }

            const ScalarType tInitialValue = 0;
            const ScalarType tGlobalSumP = std::accumulate(tMyStorageOne.begin(), tMyStorageOne.end(), tInitialValue);
            const ScalarType tGlobalSumQ = std::accumulate(tMyStorageTwo.begin(), tMyStorageTwo.end(), tInitialValue);
            // Add contribution to dual gradient
            tDualGradient[tConstraintIndex] = static_cast<ScalarType>(-1)
                    * (tDualGradient[tConstraintIndex] + tGlobalSumP + tGlobalSumQ);
        }
    }
    // APPLY VECTOR TO DUAL HESSIAN
    void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aDual,
                              const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                              locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        return;
    }

    // GET OPTIMAL TRIAL CONTROL FROM DUAL PROBLEM
    void getTrialControl(locus::MultiVector<ScalarType, OrdinalType> & aInput) const
    {
        locus::update(static_cast<ScalarType>(1), *mTrialControl, static_cast<ScalarType>(0), aInput);
    }
    void updateObjectiveCoefficients(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mObjectiveCoefficientR = mCurrentObjectiveFunctionValue;
        const ScalarType tGlobalizationFactor = aDataMng.getDualObjectiveGlobalizationFactor();
        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma();

        const OrdinalType tNumVectors = tCurrentSigma.getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentSigma = tCurrentSigma[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCurrentObjectiveGradient =
                    aDataMng.getCurrentObjectiveGradient(tVectorIndex);

            OrdinalType tNumControls = tMyCurrentSigma.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                        * tMyCurrentSigma[tControlIndex];
                (*mObjectiveCoefficientsP)(tVectorIndex, tControlIndex) = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ScalarType>(0), tMyCurrentObjectiveGradient[tControlIndex]);
                +((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ScalarType>(4));

                (*mObjectiveCoefficientsQ)(tVectorIndex, tControlIndex) = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ScalarType>(0), -tMyCurrentObjectiveGradient[tControlIndex])
                        + ((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ScalarType>(4));
                (*mControlWorkVectorOne)[tControlIndex] = ((*mObjectiveCoefficientsP)(tVectorIndex, tControlIndex)
                        + (*mObjectiveCoefficientsQ)(tVectorIndex, tControlIndex)) / tMyCurrentSigma[tControlIndex];
            }
            tStorage[tVectorIndex] = mControlReductionOperations->sum(*mControlWorkVectorOne);
        }

        const ScalarType tInitialValue = 0;
        const ScalarType tValue = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
        mObjectiveCoefficientR = mObjectiveCoefficientR - tValue;
    }
    void updateConstraintCoefficients(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tDualVectorIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tCurrentConstraintValues =
                mCurrentConstraintValues->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType, OrdinalType> & tGlobalizationFactor =
                aDataMng.getConstraintGlobalizationFactors(tDualVectorIndex);
        locus::Vector<ScalarType, OrdinalType> & tConstraintCoefficientsR =
                mConstraintCoefficientsR.operator*()[tDualVectorIndex];

        const OrdinalType tNumConstraints = tConstraintCoefficientsR.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            tConstraintCoefficientsR[tConstraintIndex] = tCurrentConstraintValues[tConstraintIndex];
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentConstraintGradients =
                    aDataMng.getCurrentConstraintGradients(tConstraintIndex);
            locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoeffP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoeffQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();
            assert(tCurrentConstraintGradients.getNumVectors() == aDataMng.getNumControlVectors());

            const OrdinalType tNumControlVectors = tCurrentConstraintGradients.getNumVectors();
            std::vector<ScalarType> tStorage(tNumControlVectors);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
                locus::Vector<ScalarType, OrdinalType> & tMyCoeffP = tMyConstraintCoeffP[tVectorIndex];
                locus::Vector<ScalarType, OrdinalType> & tMyCoeffQ = tMyConstraintCoeffQ[tVectorIndex];
                const locus::Vector<ScalarType, OrdinalType> & tMyCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tMyCurrentGradient = tCurrentConstraintGradients[tVectorIndex];

                const OrdinalType tNumControls = tMyCurrentGradient.size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ScalarType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                            * tMyCurrentSigma[tControlIndex];
                    tMyCoeffP[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                            * std::max(static_cast<ScalarType>(0), tMyCurrentGradient[tControlIndex])
                            + ((tGlobalizationFactor[tConstraintIndex] * tMyCurrentSigma[tControlIndex])
                                    / static_cast<ScalarType>(4));

                    tMyCoeffQ[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                            * std::max(static_cast<ScalarType>(0), -tMyCurrentGradient[tControlIndex])
                            + ((tGlobalizationFactor[tConstraintIndex] * tMyCurrentSigma[tControlIndex])
                                    / static_cast<ScalarType>(4));

                    (*mControlWorkVectorOne)[tControlIndex] = (tMyCoeffP[tControlIndex] + tMyCoeffQ[tControlIndex])
                            / tMyCurrentSigma[tControlIndex];
                }
                tStorage[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
            }

            const ScalarType tInitialValue = 0;
            const ScalarType tValue = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
            tConstraintCoefficientsR[tConstraintIndex] = tConstraintCoefficientsR[tConstraintIndex] - tValue;
        }
    }
    void initializeAuxiliaryVariables(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tDualVectorIndex = 0;
        locus::Vector<ScalarType> & tAuxiliaryVariablesY =
                mTrialAuxiliaryVariableY->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType> & tCoefficientsA =
                mConstraintCoefficientsA->operator[](tDualVectorIndex);
        const locus::Vector<ScalarType, OrdinalType> & tCurrentConstraintValues =
                aDataMng.getCurrentConstraintValues(tDualVectorIndex);

        const OrdinalType tNumConstraints = mDualWorkVector->size();
        const ScalarType tMaxCoefficientA = mDualReductionOperations->max(tCoefficientsA);
        if(tMaxCoefficientA > static_cast<ScalarType>(0))
        {
            mDualWorkVector->fill(static_cast<ScalarType>(0));
            for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
            {
                if(tCoefficientsA[tIndex] > static_cast<ScalarType>(0))
                {
                    ScalarType tValue = std::max(static_cast<ScalarType>(0), tCurrentConstraintValues[tIndex]);
                    (*mDualWorkVector)[tIndex] = tValue / tCoefficientsA[tIndex];
                    tAuxiliaryVariablesY[tIndex] = 0;
                }
                else
                {
                    tAuxiliaryVariablesY[tIndex] =
                            std::max(static_cast<ScalarType>(0), tCurrentConstraintValues[tIndex]);
                }
            }
            mTrialAuxiliaryVariableZ = mDualReductionOperations->max(mDualWorkVector.operator*());
        }
        else
        {
            for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
            {
                tAuxiliaryVariablesY[tIndex] = tCurrentConstraintValues[tIndex];
            }
            mTrialAuxiliaryVariableZ = 0;
        }
    }
    void checkConstraintCoefficients()
    {
        const OrdinalType tDualVectorIndex = 0;
        ScalarType tMinCoeffA = mDualReductionOperations->min(mConstraintCoefficientsA->operator[](tDualVectorIndex));
        assert(tMinCoeffA >= static_cast<ScalarType>(0));
        ScalarType tMinCoeffC = mDualReductionOperations->min(mConstraintCoefficientsC->operator[](tDualVectorIndex));
        assert(tMinCoeffC >= static_cast<ScalarType>(0));
        ScalarType tMinCoeffD = mDualReductionOperations->min(mConstraintCoefficientsD->operator[](tDualVectorIndex));
        assert(tMinCoeffD >= static_cast<ScalarType>(0));
    }

private:
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        const OrdinalType tDualVectorIndex = 0;
        mDualWorkVector = aDataFactory.dual(tDualVectorIndex).create();

        const OrdinalType tControlVectorIndex = 0;
        mControlWorkVectorOne = aDataFactory.control(tControlVectorIndex).create();
        mControlWorkVectorTwo = aDataFactory.control(tControlVectorIndex).create();

        const OrdinalType tNumConstraints = aDataFactory.dual(tDualVectorIndex).size();
        mConstraintCoefficientsP.resize(tNumConstraints);
        mConstraintCoefficientsQ.resize(tNumConstraints);
        for(OrdinalType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            mConstraintCoefficientsP[tIndex] = aDataFactory.control().create();
            mConstraintCoefficientsQ[tIndex] = aDataFactory.control().create();
        }

        locus::fill(static_cast<ScalarType>(0), mConstraintCoefficientsA.operator*());
        locus::fill(static_cast<ScalarType>(1), mConstraintCoefficientsD.operator*());
        locus::fill(static_cast<ScalarType>(1e3), mConstraintCoefficientsC.operator*());
    }
    ScalarType computeMovingAsymptotesContribution()
    {
        const OrdinalType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ScalarType> tMySumP(tNumVectors);
        std::vector<ScalarType> tMySumQ(tNumVectors);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlWorkVectorOne->fill(static_cast<ScalarType>(0));
            mControlWorkVectorTwo->fill(static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyTrialControl = mTrialControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyLowerAsymptotes =
                    mLowerAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyUpperAsymptotes =
                    mUpperAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyObjectiveCoefficientsP =
                    mObjectiveCoefficientsP->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyObjectiveCoefficientsQ =
                    mObjectiveCoefficientsQ->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsP =
                    mDualTimesCoefficientsP->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsQ =
                    mDualTimesCoefficientsQ->operator[](tVectorIndex);

            const OrdinalType tNumControls = tMyTrialControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tNumerator = tMyObjectiveCoefficientsP[tControlIndex]
                        + tMyDualTimesCoefficientsP[tControlIndex];
                ScalarType tDenominator = tMyUpperAsymptotes[tControlIndex] - tMyTrialControl[tControlIndex];
                (*mControlWorkVectorOne)[tControlIndex] = tNumerator / tDenominator;

                tNumerator = tMyObjectiveCoefficientsQ[tControlIndex] + tMyDualTimesCoefficientsQ[tControlIndex];
                tDenominator = tMyTrialControl[tControlIndex] - tMyLowerAsymptotes[tControlIndex];
                (*mControlWorkVectorTwo)[tControlIndex] = tNumerator / tDenominator;
            }

            tMySumP[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
            tMySumQ[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorTwo.operator*());
        }

        const ScalarType tInitialValue = 0;
        const ScalarType tGlobalSumP = std::accumulate(tMySumP.begin(), tMySumP.end(), tInitialValue);
        const ScalarType tGlobalSumQ = std::accumulate(tMySumQ.begin(), tMySumQ.end(), tInitialValue);
        const ScalarType tMovingAsymptotesTerm = tGlobalSumP + tGlobalSumQ;

        return (tMovingAsymptotesTerm);
    }
    // Compute trial controls based on the following explicit expression:
    // \[ x(\lambda)=\frac{u_j^k\mathtt{b}^{1/2}+l_j^k\mathtt{a}^{1/2}}{(\mathtt{a}^{1/2}+\mathtt{b}^{1/2})} \],
    // where
    //      \[ \mathtt{a}=(p_{0j}+\lambda^{\intercal}p_j) \] and [ \mathtt{b}=(q_{0j}+\lambda^{\intercal}q_j) ]
    //      and j=1\dots,n_{x}
    // Here, x denotes the trial control vector
    void computeTrialControl(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        locus::update(static_cast<ScalarType>(1), *mObjectiveCoefficientsP, static_cast<ScalarType>(0), *mTermA);
        locus::update(static_cast<ScalarType>(1), *mDualTimesCoefficientsP, static_cast<ScalarType>(1), *mTermA);
        locus::update(static_cast<ScalarType>(1), *mObjectiveCoefficientsQ, static_cast<ScalarType>(0), *mTermB);
        locus::update(static_cast<ScalarType>(1), *mDualTimesCoefficientsQ, static_cast<ScalarType>(1), *mTermB);

        const OrdinalType tNumControlVectors = mTrialControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tMyTrialControl = mTrialControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyLowerAsymptotes =
                    mLowerAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyUpperAsymptotes =
                    mUpperAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyLowerBounds =
                    mTrialControlLowerBounds->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tMyUpperBounds =
                    mTrialControlUpperBounds->operator[](tVectorIndex);

            OrdinalType tNumControls = tMyTrialControl.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tSqrtTermA = std::sqrt(mTermA->operator()(tVectorIndex, tControlIndex));
                ScalarType tSqrtTermB = std::sqrt(mTermB->operator()(tVectorIndex, tControlIndex));
                ScalarType tNumerator = (tMyLowerAsymptotes[tControlIndex] * tSqrtTermA)
                        + (tMyUpperAsymptotes[tControlIndex] * tSqrtTermB);
                ScalarType tDenominator = (tSqrtTermA + tSqrtTermB);
                tMyTrialControl[tControlIndex] = tNumerator / tDenominator;
                // Project trial control to feasible set
                tMyTrialControl[tControlIndex] =
                        std::max(tMyTrialControl[tControlIndex], tMyLowerBounds[tControlIndex]);
                tMyTrialControl[tControlIndex] =
                        std::min(tMyTrialControl[tControlIndex], tMyUpperBounds[tControlIndex]);
            }
        }
    }
    /*!
     * Update auxiliary variables based on the following expression:
     *  \[ y_i(\lambda)=\frac{\lambda_i-c_i}{2d_i} \]
     *  and
     *  \[ z(\lambda)=\frac{\lambda^{\intercal}a-a_0}{2\varepsilon} \]
     */
    void computeTrialAuxiliaryVariables(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        OrdinalType tNumVectors = aDual.getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyDual = aDual[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsC = (*mConstraintCoefficientsC)[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsD = (*mConstraintCoefficientsD)[tVectorIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyAuxiliaryVariablesY = (*mTrialAuxiliaryVariableY)[tVectorIndex];

            const OrdinalType tNumDual = tMyAuxiliaryVariablesY.size();
            for(OrdinalType tIndex = 0; tIndex < tNumDual; tIndex++)
            {
                ScalarType tDualMinusConstraintCoefficientC = tMyDual[tIndex] - tMyCoefficientsC[tIndex];
                tMyAuxiliaryVariablesY[tIndex] = tDualMinusConstraintCoefficientC / tMyCoefficientsD[tIndex];
                // Project auxiliary variables Y to feasible set (Y >= 0)
                tMyAuxiliaryVariablesY[tIndex] = std::max(tMyAuxiliaryVariablesY[tIndex], static_cast<ScalarType>(0));
            }
        }
        ScalarType tDualDotConstraintCoefficientA = locus::dot(aDual, *mConstraintCoefficientsA);
        mTrialAuxiliaryVariableZ = (tDualDotConstraintCoefficientA - mObjectiveCoefficientA)
                / (static_cast<ScalarType>(2) * mEpsilon);
        // Project auxiliary variables Z to feasible set (Z >= 0)
        mTrialAuxiliaryVariableZ = std::max(mTrialAuxiliaryVariableZ, static_cast<ScalarType>(0));
    }
    /*! Compute: \sum_{i=1}^{m}\left( c_iy_i + \frac{1}{2}d_iy_i^2 \right) -
     * \lambda^{T}y - (\lambda^{T}a)z + \lambda^{T}r, where m is the number of constraints
     **/
    ScalarType computeConstraintContribution(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<OrdinalType>(1));

        const OrdinalType tNumVectors = aDual.getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors);
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mDualWorkVector->fill(static_cast<ScalarType>(0));
            const locus::Vector<ScalarType, OrdinalType> & tMyDual = aDual[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsC = (*mConstraintCoefficientsC)[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsD = (*mConstraintCoefficientsD)[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tMyAuxiliaryVariablesY =
                    (*mTrialAuxiliaryVariableY)[tVectorIndex];

            const OrdinalType tNumDuals = tMyDual.size();
            for(OrdinalType tIndex = 0; tIndex < tNumDuals; tIndex++)
            {
                ScalarType tValueOne = tMyCoefficientsC[tIndex] * tMyAuxiliaryVariablesY[tIndex];
                ScalarType tValueTwo = tMyCoefficientsD[tIndex] * tMyAuxiliaryVariablesY[tIndex]
                        * tMyAuxiliaryVariablesY[tIndex];
                (*mDualWorkVector)[tIndex] = tValueOne + tValueTwo;
            }

            tStorage[tVectorIndex] = mDualReductionOperations->sum(mDualWorkVector.operator*());
        }

        const ScalarType tInitialValue = 0;
        ScalarType tConstraintSummationTerm = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);

        // Add additional contributions to inequality summation term
        ScalarType tDualDotConstraintCoeffR = locus::dot(aDual, *mConstraintCoefficientsR);
        ScalarType tDualDotConstraintCoeffA = locus::dot(aDual, *mConstraintCoefficientsA);
        ScalarType tDualDotTrialAuxiliaryVariableY = locus::dot(aDual, *mTrialAuxiliaryVariableY);
        ScalarType tOutput = tConstraintSummationTerm - tDualDotTrialAuxiliaryVariableY
                - (tDualDotConstraintCoeffA * mTrialAuxiliaryVariableZ) + tDualDotConstraintCoeffR;

        return (tOutput);
    }
    /*
     * Compute \lambda_j\times{p}_j and \lambda_j\times{q}_j, where
     * j=1,\dots,N_{c}. Here, N_{c} denotes the number of constraints.
     **/
    void computeDualTimesConstraintCoefficientTerms(const locus::MultiVector<ScalarType, OrdinalType> & aDual)
    {
        const OrdinalType tDualVectorIndex = 0;
        locus::fill(static_cast<ScalarType>(0), mDualTimesCoefficientsP.operator*());
        locus::fill(static_cast<ScalarType>(0), mDualTimesCoefficientsQ.operator*());
        const locus::Vector<ScalarType, OrdinalType> & tDual = aDual[tDualVectorIndex];

        const ScalarType tBeta = 1;
        const OrdinalType tNumConstraints = tDual.size();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            const locus::MultiVector<ScalarType, OrdinalType> & tMyConstraintCoefficientsQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();

            const OrdinalType tNumControlVectors = tMyConstraintCoefficientsP.getNumVectors();
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsP =
                        tMyConstraintCoefficientsP[tVectorIndex];
                locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsP =
                        (*mDualTimesCoefficientsP)[tVectorIndex];
                tMyDualTimesCoefficientsP.update(tDual[tConstraintIndex], tMyCoefficientsP, tBeta);

                const locus::Vector<ScalarType, OrdinalType> & tMyCoefficientsQ =
                        tMyConstraintCoefficientsQ[tVectorIndex];
                locus::Vector<ScalarType, OrdinalType> & tMyDualTimesCoefficientsQ =
                        (*mDualTimesCoefficientsQ)[tVectorIndex];
                tMyDualTimesCoefficientsQ.update(tDual[tConstraintIndex], tMyCoefficientsQ, tBeta);
            }
        }
    }

private:
    ScalarType mEpsilon;
    ScalarType mObjectiveCoefficientA;
    ScalarType mObjectiveCoefficientR;
    ScalarType mTrialAuxiliaryVariableZ;
    ScalarType mCurrentObjectiveFunctionValue;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkVector;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVectorOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVectorTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTermA;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTermB;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mLowerAsymptotes;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mUpperAsymptotes;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualTimesCoefficientsP;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualTimesCoefficientsQ;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mObjectiveCoefficientsP;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mObjectiveCoefficientsQ;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControlUpperBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsA;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsC;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsD;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintCoefficientsR;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialAuxiliaryVariableY;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentConstraintValues;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

    std::vector<std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>>> mConstraintCoefficientsP;
    std::vector<std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>>> mConstraintCoefficientsQ;

private:
    DualProblemStageMng(const locus::DualProblemStageMng<ScalarType, OrdinalType> & aRhs);
    locus::DualProblemStageMng<ScalarType, OrdinalType> & operator=(const locus::DualProblemStageMng<ScalarType, OrdinalType> & aRhs);
};

struct ccsa
{
    enum method_t
    {
        MMA = 1,
        GCMMA = 2
    };

    enum stop_t
    {
        STATIONARITY_TOLERANCE = 1,
        KKT_CONDITIONS_TOLERANCE = 2,
        CONTROL_STAGNATION = 3,
        OBJECTIVE_STAGNATION = 4,
        MAX_NUMBER_ITERATIONS = 5,
        OPTIMALITY_AND_FEASIBILITY_MET = 6,
        NOT_CONVERGED = 7,
    };
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxDualSolver
{
public:
    ConservativeConvexSeparableAppxDualSolver()
    {
    }
    virtual ~ConservativeConvexSeparableAppxDualSolver()
    {
    }

private:
    ConservativeConvexSeparableAppxDualSolver(const locus::ConservativeConvexSeparableAppxDualSolver<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableAppxDualSolver<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableAppxDualSolver<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableApproximation
{
public:
    virtual ~ConservativeConvexSeparableApproximation()
    {
    }

    virtual void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
                       locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class DualProblemSolver
{
public:
    virtual ~DualProblemSolver()
    {
    }

    virtual void solve(locus::MultiVector<ScalarType, OrdinalType> & aDual,
                       locus::MultiVector<ScalarType, OrdinalType> & aTrialControl) = 0;
    virtual void update(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void updateObjectiveCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void updateConstraintCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientDualSolver : public locus::DualProblemSolver<ScalarType, OrdinalType>
{
public:
    explicit NonlinearConjugateGradientDualSolver(const locus::DataFactory<ScalarType, OrdinalType> & aPrimalDataFactory) :
            mDualWork(aPrimalDataFactory.dual().create()),
            mDualInitialGuess(aPrimalDataFactory.dual().create()),
            mDualDataFactory(std::make_shared<locus::DataFactory<ScalarType, OrdinalType>>()),
            mDualStageMng(),
            mDualAlgorithm(),
            mDualDataMng()
    {
        this->initialize(aPrimalDataFactory);
    }
    virtual ~NonlinearConjugateGradientDualSolver()
    {
    }

    void solve(locus::MultiVector<ScalarType, OrdinalType> & aDual,
               locus::MultiVector<ScalarType, OrdinalType> & aTrialControl)
    {
        this->reset();
        mDualDataMng->setInitialGuess(mDualInitialGuess.operator*());
        mDualAlgorithm->solve();

        mDualStageMng->getTrialControl(aTrialControl);
        const locus::MultiVector<ScalarType, OrdinalType> & tDualSolution = mDualDataMng->getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tDualSolution, static_cast<ScalarType>(0), aDual);
        // Store dual solution and use it as the initial guess for the next iteration.
        locus::update(static_cast<ScalarType>(1), tDualSolution, static_cast<ScalarType>(0), mDualInitialGuess.operator*());
    }

    void update(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->update(aDataMng);
    }
    void updateObjectiveCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->updateObjectiveCoefficients(aDataMng);
    }
    void updateConstraintCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->updateConstraintCoefficients(aDataMng);
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->initializeAuxiliaryVariables(aDataMng);
    }

private:
    void reset()
    {
        const ScalarType tValue = 0;
        mDualDataMng->setCurrentObjectiveFunctionValue(tValue);
        mDualDataMng->setPreviousObjectiveFunctionValue(tValue);

        locus::fill(tValue, mDualWork.operator*());
        mDualDataMng->setTrialStep(mDualWork.operator*());
        mDualDataMng->setCurrentControl(mDualWork.operator*());
        mDualDataMng->setPreviousControl(mDualWork.operator*());
        mDualDataMng->setCurrentGradient(mDualWork.operator*());
        mDualDataMng->setPreviousGradient(mDualWork.operator*());
    }
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aPrimalDataFactory)
    {
        mDualDataFactory->allocateControl(mDualWork.operator*());
        mDualDataMng = std::make_shared<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>>(mDualDataFactory.operator*());

        ScalarType tValue = 0;
        mDualDataMng->setControlLowerBounds(tValue);
        tValue = std::numeric_limits<ScalarType>::max();
        mDualDataMng->setControlLowerBounds(tValue);

        mDualStageMng = std::make_shared<locus::DualProblemStageMng<ScalarType, OrdinalType>>(aPrimalDataFactory);
        mDualAlgorithm = std::make_shared<locus::NonlinearConjugateGradient<ScalarType, OrdinalType>>(mDualDataFactory, mDualDataMng, mDualStageMng);
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualInitialGuess;

    std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> mDualDataFactory;
    std::shared_ptr<locus::DualProblemStageMng<ScalarType, OrdinalType>> mDualStageMng;
    std::shared_ptr<locus::NonlinearConjugateGradient<ScalarType, OrdinalType>> mDualAlgorithm;
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> mDualDataMng;

private:
    NonlinearConjugateGradientDualSolver(const locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class MethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>
{
public:
    explicit MethodMovingAsymptotes(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mConstraintValues(aDataFactory.dual().create()),
            mDualSolver(std::make_shared<locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType>>(aDataFactory))
    {
    }
    virtual ~MethodMovingAsymptotes()
    {
    }

    void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        // NOTE: REMBER THAT THE GLOBALIZATION FACTORS FOR BOTH OBJECTIVE AND CONSTRAINTS ARE SET TO ZERO IF USING
        //       THE MMA APPROACH. THE MMA METHOD IS DETECTED AT THE OUTTER LOOP LEVEL (NOT THE SUBPROBLEM LEVEL)
        //       AND THE VALUES SET INSIDE THE INITIALZE FUNCTION IN THE ALGORITHM CLASS. DEFAULT VALUES ARE ONLY
        //       SET FOR THE GCMMA CASE.
        mDualSolver->update(aDataMng);
        mDualSolver->updateObjectiveCoefficients(aDataMng);
        mDualSolver->updateConstraintCoefficients(aDataMng);
        mDualSolver->solve(mTrialDual.operator*(), mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());
        locus::bounds::computeActiveAndInactiveSets(mTrialControl.operator*(),
                                                    tLowerBounds,
                                                    tUpperBounds,
                                                    mActiveSet.operator*(),
                                                    mInactiveSet.operator*());
        aDataMng.setActiveSet(mActiveSet.operator*());
        aDataMng.setInactiveSet(mInactiveSet.operator*());

        ScalarType tObjectiveFunctionValue = aPrimalProblemStageMng.evaluateObjective(mTrialControl.operator*());
        aDataMng.setCurrentObjectiveFunctionValue(tObjectiveFunctionValue);
        aPrimalProblemStageMng.evaluateConstraints(mTrialControl.operator*(), mConstraintValues.operator*());
        aDataMng.setCurrentConstraintValues(mConstraintValues.operator*());
        aDataMng.setCurrentControl(mTrialControl.operator*());
        aDataMng.setDual(mTrialDual.operator*());
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->initializeAuxiliaryVariables(aDataMng);
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintValues;
    std::shared_ptr<locus::DualProblemSolver<ScalarType, OrdinalType>> mDualSolver;

private:
    MethodMovingAsymptotes(const locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
    locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & operator=(const locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class GloballyConvergentMethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>
{
public:
    explicit GloballyConvergentMethodMovingAsymptotes(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(10),
            mNumIterationsDone(0),
            mStagnationTolerance(1e-6),
            mMinObjectiveGlobalizationFactor(1e-5),
            mCurrentTrialObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mPreviousTrialObjectiveFunctionValue(std::numeric_limits<ScalarType>::max()),
            mKarushKuhnTuckerConditionsTolerance(1e-6),
            mStoppingCriterion(locus::ccsa::stop_t::NOT_CONVERGED),
            mControlWorkOne(),
            mControlWorkTwo(),
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mDeltaControl(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mTrialConstraintValues(aDataFactory.dual().create()),
            mMinConstraintGlobalizationFactors(aDataFactory.dual().create()),
            mDualSolver(std::make_shared<locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType>>(aDataFactory)),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize();
    }
    virtual ~GloballyConvergentMethodMovingAsymptotes()
    {
    }

    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    void setNumIterationsDone(const OrdinalType & aInput)
    {
        mNumIterationsDone = aInput;
    }

    ScalarType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    void setStagnationTolerance(const ScalarType & aInput) const
    {
        mStagnationTolerance = aInput;
    }
    ScalarType getKarushKuhnTuckerConditionsTolerance() const
    {
        return (mKarushKuhnTuckerConditionsTolerance);
    }
    void setKarushKuhnTuckerConditionsTolerance(const ScalarType & aInput)
    {
        mKarushKuhnTuckerConditionsTolerance = aInput;
    }

    locus::ccsa::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    void setStoppingCriterion(const locus::ccsa::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->update(aDataMng);

        OrdinalType tIterations = 0;
        this->setNumIterationsDone(tIterations);
        while(1)
        {
            mDualSolver->updateObjectiveCoefficients(aDataMng);
            mDualSolver->updateConstraintCoefficients(aDataMng);
            mDualSolver->solve(mTrialDual.operator*(), mTrialControl.operator*());

            const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
            const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
            locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());
            locus::bounds::computeActiveAndInactiveSets(mTrialControl.operator*(),
                                                        tLowerBounds,
                                                        tUpperBounds,
                                                        mActiveSet.operator*(),
                                                        mInactiveSet.operator*());
            aDataMng.setActiveSet(mActiveSet.operator*());
            aDataMng.setInactiveSet(mInactiveSet.operator*());

            mCurrentTrialObjectiveFunctionValue = aPrimalProblemStageMng.evaluateObjective(mTrialControl.operator*());
            aPrimalProblemStageMng.evaluateConstraints(mTrialControl.operator*(), mTrialConstraintValues.operator*());

            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = aDataMng.getCurrentControl();
            locus::update(static_cast<ScalarType>(1), *mTrialControl, static_cast<ScalarType>(0), *mDeltaControl);
            locus::update(static_cast<ScalarType>(-1), tCurrentControl, static_cast<ScalarType>(1), *mDeltaControl);
            this->updateObjectiveGlobalizationFactor(aDataMng);
            this->updateConstraintGlobalizationFactors(aDataMng);

            tIterations++;
            this->setNumIterationsDone(tIterations);
            if(this->checkStoppingCriteria(aDataMng) == true)
            {
                break;
            }
        }

        aDataMng.setDual(mTrialDual.operator*());
        aDataMng.setCurrentControl(mTrialControl.operator*());
        aDataMng.setCurrentObjectiveFunctionValue(mCurrentTrialObjectiveFunctionValue);
        aDataMng.setCurrentConstraintValues(mTrialConstraintValues.operator*());
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->initializeAuxiliaryVariables(aDataMng);
    }

private:
    void initialize()
    {
        const OrdinalType tVectorIndex = 0;
        mControlWorkOne = mTrialControl->operator[](tVectorIndex).create();
        mControlWorkTwo = mTrialControl->operator[](tVectorIndex).create();
        locus::fill(static_cast<ScalarType>(1e-5), mMinConstraintGlobalizationFactors.operator*());
    }
    void updateObjectiveGlobalizationFactor(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        locus::fill(static_cast<ScalarType>(0), mControlWorkOne.operator*());
        locus::fill(static_cast<ScalarType>(0), mControlWorkTwo.operator*());

        const OrdinalType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ScalarType> tStorageOne(tNumVectors, 0);
        std::vector<ScalarType> tStorageTwo(tNumVectors, 0);

        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tWorkVectorOne = mControlWorkOne->operator[](tVectorIndex);
            locus::Vector<ScalarType, OrdinalType> & tWorkVectorTwo = mControlWorkOne->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
            const locus::Vector<ScalarType, OrdinalType> & tCurrentGradient = aDataMng.getCurrentObjectiveGradient(tVectorIndex);
            assert(tWorkVectorOne.size() == tWorkVectorTwo.size());

            const OrdinalType tNumControls = tWorkVectorOne.size();
            for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ScalarType tNumerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                ScalarType tDenominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                        - (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]);
                tWorkVectorOne[tControlIndex] = tNumerator / tDenominator;

                tNumerator = ((tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                        * tCurrentGradient[tControlIndex] * tDeltaControl[tControlIndex])
                        + (tCurrentSigma[tControlIndex] * std::abs(tCurrentGradient[tControlIndex])
                                * (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]));
                tWorkVectorTwo[tControlIndex] = tNumerator / tDenominator;
            }

            tStorageOne[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorOne);
            tStorageTwo[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorTwo);
        }

        const ScalarType tInitialValue = 0;
        ScalarType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        tFunctionEvaluationW = static_cast<ScalarType>(0.5) * tFunctionEvaluationW;
        ScalarType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
        const ScalarType tCurrentObjectiveValue = aDataMng.getCurrentObjectiveFunctionValue();
        tFunctionEvaluationV = tCurrentObjectiveValue + tFunctionEvaluationV;

        ScalarType tGlobalizationFactor = aDataMng.getDualObjectiveGlobalizationFactor();
        const ScalarType tCcsaFunctionValue = tFunctionEvaluationV + (tGlobalizationFactor * tFunctionEvaluationW);

        const ScalarType tActualOverPredictedReduction = (mCurrentTrialObjectiveFunctionValue - tCcsaFunctionValue)
                / tFunctionEvaluationW;
        if(tActualOverPredictedReduction > static_cast<ScalarType>(0))
        {
            ScalarType tValueOne = static_cast<ScalarType>(10) * tGlobalizationFactor;
            ScalarType tValueTwo = static_cast<ScalarType>(1.1)
                    * (tGlobalizationFactor + tActualOverPredictedReduction);
            tGlobalizationFactor = std::min(tValueOne, tValueTwo);
        }

        aDataMng.setDualObjectiveGlobalizationFactor(tGlobalizationFactor);
    }
    void updateConstraintGlobalizationFactors(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<OrdinalType>(1));
        const OrdinalType tNumDualVectors = 1;
        const locus::Vector<ScalarType, OrdinalType> & tConstraintValues =
                aDataMng.getCurrentConstraintValues(tNumDualVectors);
        locus::Vector<ScalarType, OrdinalType> & tGlobalizationFactors =
                aDataMng.getConstraintGlobalizationFactors(tNumDualVectors);

        const OrdinalType tNumConstraints = aDataMng.getNumConstraints();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            locus::fill(static_cast<ScalarType>(0), mControlWorkOne.operator*());
            locus::fill(static_cast<ScalarType>(0), mControlWorkTwo.operator*());

            const OrdinalType tNumVectors = mTrialControl->getNumVectors();
            std::vector<ScalarType> tStorageOne(tNumVectors, 0);
            std::vector<ScalarType> tStorageTwo(tNumVectors, 0);
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                locus::Vector<ScalarType, OrdinalType> & tWorkVectorOne = mControlWorkOne->operator[](tVectorIndex);
                locus::Vector<ScalarType, OrdinalType> & tWorkVectorTwo = mControlWorkOne->operator[](tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ScalarType, OrdinalType> & tConstraintGradient =
                        aDataMng.getCurrentConstraintGradients(tConstraintIndex, tVectorIndex);
                assert(tDeltaControl.size() == tWorkVectorOne.size());
                assert(tWorkVectorOne.size() == tWorkVectorTwo.size());
                assert(tConstraintGradient.size() == tDeltaControl.size());

                const OrdinalType tNumControls = tWorkVectorOne.size();
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ScalarType numerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                    ScalarType denominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                            - (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]);
                    tWorkVectorOne[tControlIndex] = numerator / denominator;

                    numerator = ((tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
                            * tConstraintGradient[tControlIndex] * tDeltaControl[tControlIndex])
                            + (tCurrentSigma[tControlIndex] * std::abs(tConstraintGradient[tControlIndex])
                                    * (tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex]));
                    tWorkVectorTwo[tControlIndex] = numerator / denominator;
                }

                tStorageOne[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorOne);
                tStorageTwo[tVectorIndex] = mControlReductionOperations->sum(tWorkVectorTwo);
            }

            const ScalarType tInitialValue = 0;
            ScalarType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
            tFunctionEvaluationW = static_cast<ScalarType>(0.5) * tFunctionEvaluationW;

            ScalarType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
            tFunctionEvaluationV = tFunctionEvaluationV + tConstraintValues[tConstraintIndex];

            ScalarType tCcsaFunctionValue = tFunctionEvaluationV + (tGlobalizationFactors[tConstraintIndex] * tFunctionEvaluationW);
            ScalarType tActualOverPredictedReduction = (tConstraintValues[tConstraintIndex] - tCcsaFunctionValue) / tFunctionEvaluationW;

            if(tActualOverPredictedReduction > static_cast<ScalarType>(0))
            {
                ScalarType tValueOne = static_cast<ScalarType>(10) * tGlobalizationFactors[tConstraintIndex];
                ScalarType tValueTwo = static_cast<ScalarType>(1.1)
                        * (tGlobalizationFactors[tConstraintIndex] + tActualOverPredictedReduction);
                tGlobalizationFactors[tConstraintIndex] = std::min(tValueOne, tValueTwo);
            }
        }
    }
    bool checkStoppingCriteria(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        aDataMng.computeKarushKuhnTuckerConditionsInexactness(mTrialControl.operator*(), mTrialDual.operator*());
        const ScalarType t_KKT_ConditionsInexactness = aDataMng.getKarushKuhnTuckerConditionsInexactness();

        const ScalarType tObjectiveStagnation =
                std::abs(mCurrentTrialObjectiveFunctionValue - mPreviousTrialObjectiveFunctionValue);

        const OrdinalType tNumIterationsDone = this->getNumIterationsDone();

        bool tStop = false;
        if(tNumIterationsDone >= this->getMaxNumIterations())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::MAX_NUMBER_ITERATIONS);
        }
        else if(t_KKT_ConditionsInexactness < this->getKarushKuhnTuckerConditionsTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::KKT_CONDITIONS_TOLERANCE);
        }
        else if(tObjectiveStagnation < this->getStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::OBJECTIVE_STAGNATION);
        }

        return (tStop);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mStagnationTolerance;
    ScalarType mMinObjectiveGlobalizationFactor;
    ScalarType mCurrentTrialObjectiveFunctionValue;
    ScalarType mPreviousTrialObjectiveFunctionValue;
    ScalarType mKarushKuhnTuckerConditionsTolerance;

    locus::ccsa::stop_t mStoppingCriterion;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkTwo;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDeltaControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialConstraintValues;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mMinConstraintGlobalizationFactors;

    std::shared_ptr<locus::DualProblemSolver<ScalarType, OrdinalType>> mDualSolver;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    GloballyConvergentMethodMovingAsymptotes(const locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
    locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & operator=(const locus::GloballyConvergentMethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableApproximationsAlgorithm
{
public:
    ConservativeConvexSeparableApproximationsAlgorithm(const std::shared_ptr<locus::PrimalProblemStageMng<ScalarType, OrdinalType>> & aPrimalStageMng,
                                                       const std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType>> & aDataMng,
                                                       const std::shared_ptr<locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>> & aSubProblem) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mOptimalityTolerance(1e-4),
            mStagnationTolerance(1e-3),
            mFeasibilityTolerance(1e-4),
            mStationarityTolerance(1e-4),
            mObjectiveStagnationTolerance(1e-8),
            mMovingAsymptoteExpansionFactor(1.2),
            mMovingAsymptoteContractionFactor(0.4),
            mKarushKuhnTuckerConditionsTolerance(1e-4),
            mMovingAsymptoteUpperBoundScaleFactor(10),
            mMovingAsymptoteLowerBoundScaleFactor(0.01),
            mStoppingCriterion(locus::ccsa::stop_t::NOT_CONVERGED),
            mDualWork(),
            mControlWork(),
            mPreviousSigma(),
            mAntepenultimateControl(),
            mWorkMultiVectorList(),
            mPrimalStageMng(aPrimalStageMng),
            mDataMng(aDataMng),
            mSubProblem(aSubProblem)
    {
        this->initialize(aDataMng.operator*());
    }
    ~ConservativeConvexSeparableApproximationsAlgorithm()
    {
    }

    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }

    ScalarType getOptimalityTolerance() const
    {
        return (mOptimalityTolerance);
    }
    void setOptimalityTolerance(const ScalarType & aInput)
    {
        mOptimalityTolerance = aInput;
    }
    ScalarType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    void setStagnationTolerance(const ScalarType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    ScalarType getFeasibilityTolerance() const
    {
        return (mFeasibilityTolerance);
    }
    void setFeasibilityTolerance(const ScalarType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }
    ScalarType getStationarityTolerance() const
    {
        return (mStationarityTolerance);
    }
    void setStationarityTolerance(const ScalarType & aInput)
    {
        mStationarityTolerance = aInput;
    }
    ScalarType getObjectiveStagnationTolerance() const
    {
        return (mObjectiveStagnationTolerance);
    }
    void setObjectiveStagnationTolerance(const ScalarType & aInput)
    {
        mObjectiveStagnationTolerance = aInput;
    }
    ScalarType getMovingAsymptoteExpansionFactor() const
    {
        return (mMovingAsymptoteExpansionFactor);
    }
    void setMovingAsymptoteExpansionFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteExpansionFactor = aInput;
    }
    ScalarType getMovingAsymptoteContractionFactor() const
    {
        return (mMovingAsymptoteContractionFactor);
    }
    void setMovingAsymptoteContractionFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteContractionFactor = aInput;
    }
    ScalarType getKarushKuhnTuckerConditionsTolerance() const
    {
        return (mKarushKuhnTuckerConditionsTolerance);
    }
    void setKarushKuhnTuckerConditionsTolerance(const ScalarType & aInput)
    {
        mKarushKuhnTuckerConditionsTolerance = aInput;
    }
    ScalarType getMovingAsymptoteUpperBoundScaleFactor() const
    {
        return (mMovingAsymptoteUpperBoundScaleFactor);
    }
    void setMovingAsymptoteUpperBoundScaleFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteUpperBoundScaleFactor = aInput;
    }
    ScalarType getMovingAsymptoteLowerBoundScaleFactor() const
    {
        return (mMovingAsymptoteLowerBoundScaleFactor);
    }
    void setMovingAsymptoteLowerBoundScaleFactor(const ScalarType & aInput)
    {
        mMovingAsymptoteLowerBoundScaleFactor = aInput;
    }

    locus::ccsa::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    void setStoppingCriterion(const locus::ccsa::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    void solve()
    {
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        const ScalarType tCurrentObjectiveFunctionValue = mPrimalStageMng->evaluateObjective(tControl);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mPrimalStageMng->evaluateConstraints(tControl, mDualWork.operator*());
        mDataMng->setDual(mDualWork.operator*());
        mSubProblem->initializeAuxiliaryVariables(mDataMng.operator*());

        mNumIterationsDone = 0;
        while(1)
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
            mPrimalStageMng->computeGradient(tCurrentControl, mControlWork.operator*());
            mDataMng->setCurrentObjectiveGradient(mControlWork.operator*());
            mPrimalStageMng->computeConstraintGradients(tCurrentControl, mWorkMultiVectorList.operator*());
            mDataMng->setCurrentConstraintGradients(mWorkMultiVectorList.operator*());

            if(this->checkStoppingCriteria() == true)
            {
                break;
            }

            this->updateSigmaParameters();

            const locus::MultiVector<ScalarType, OrdinalType> & tPreviousControl = mDataMng->getPreviousControl();
            locus::update(static_cast<ScalarType>(1),
                          tPreviousControl,
                          static_cast<ScalarType>(0),
                          mAntepenultimateControl.operator*());
            locus::update(static_cast<ScalarType>(1),
                          tCurrentControl,
                          static_cast<ScalarType>(0),
                          mControlWork.operator*());
            mDataMng->setPreviousControl(mControlWork.operator*());

            mSubProblem->solve(mPrimalStageMng.operator*(), mDataMng.operator*());

            mNumIterationsDone++;
        }
    }

private:
    void initialize(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualWork = aDataMng.getDual().create();
        mControlWork = aDataMng.getCurrentControl().create();
        mPreviousSigma = aDataMng.getCurrentControl().create();
        mAntepenultimateControl = aDataMng.getCurrentControl().create();
        mWorkMultiVectorList = aDataMng.getCurrentConstraintGradients().create();
    }
    bool checkStoppingCriteria()
    {
        bool tStop = false;

        mDataMng->computeStagnationMeasure();
        mDataMng->computeFeasibilityMeasure();
        mDataMng->computeStationarityMeasure();
        mDataMng->computeNormProjectedGradient();
        mDataMng->computeObjectiveStagnationMeasure();

        const locus::MultiVector<ScalarType, OrdinalType> & tDual = mDataMng->getDual();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        mDataMng->computeKarushKuhnTuckerConditionsInexactness(tControl, tDual);

        const ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
        const ScalarType tFeasibilityMeasure = mDataMng->getFeasibilityMeasure();
        const ScalarType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ScalarType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
        const ScalarType tObjectiveStagnationMeasure = mDataMng->getObjectiveStagnationMeasure();
        const ScalarType t_KKT_ConditionsInexactness = mDataMng->getKarushKuhnTuckerConditionsInexactness();

        if(tStagnationMeasure < this->getStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::CONTROL_STAGNATION);
        }
        else if(tStationarityMeasure < this->getStationarityTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::STATIONARITY_TOLERANCE);
        }
        else if( (tFeasibilityMeasure < this->getFeasibilityTolerance())
                && (tNormProjectedGradient < this->getOptimalityTolerance()) )
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::OPTIMALITY_AND_FEASIBILITY_MET);
        }
        else if(t_KKT_ConditionsInexactness < this->getKarushKuhnTuckerConditionsTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::KKT_CONDITIONS_TOLERANCE);
        }
        else if(mNumIterationsDone < this->getMaxNumIterations())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::MAX_NUMBER_ITERATIONS);
        }
        else if(tObjectiveStagnationMeasure < this->getObjectiveStagnationTolerance())
        {
            tStop = true;
            this->setStoppingCriterion(locus::ccsa::stop_t::OBJECTIVE_STAGNATION);
        }

        return (tStop);
    }
    void updateSigmaParameters()
    {
        assert(mControlWork.get() != nullptr);
        assert(mPreviousSigma.get() != nullptr);

        const locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = mDataMng->getCurrentSigma();
        locus::update(static_cast<ScalarType>(1), tCurrentSigma, static_cast<ScalarType>(0), *mPreviousSigma);

        const OrdinalType tNumIterationsDone = this->getNumIterationsDone();
        if(tNumIterationsDone < static_cast<OrdinalType>(2))
        {
            const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = mDataMng->getControlUpperBounds();
            locus::update(static_cast<ScalarType>(1), tUpperBounds, static_cast<ScalarType>(0), *mControlWork);
            const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = mDataMng->getControlLowerBounds();
            locus::update(static_cast<ScalarType>(-1), tLowerBounds, static_cast<ScalarType>(1), *mControlWork);
            locus::scale(static_cast<ScalarType>(0.5), mControlWork.operator*());
            mDataMng->setCurrentSigma(mControlWork.operator*());
        }
        else
        {
            const OrdinalType tNumVectors = mControlWork->getNumVectors();
            locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
            for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                const ScalarType tExpansionFactor = this->getMovingAsymptoteExpansionFactor();
                const ScalarType tContractionFactor = this->getMovingAsymptoteContractionFactor();

                const locus::MultiVector<ScalarType, OrdinalType> & tCurrentControl = mDataMng->getCurrentControl();
                const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = mDataMng->getControlUpperBounds();
                const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = mDataMng->getControlLowerBounds();
                const locus::MultiVector<ScalarType, OrdinalType> & tPreviousControl = mDataMng->getPreviousControl();
                const locus::MultiVector<ScalarType, OrdinalType> & tPreviousSigma = mPreviousSigma->operator[](tVectorIndex);
                const locus::MultiVector<ScalarType, OrdinalType> & tAntepenultimateControl = mAntepenultimateControl->operator[](tVectorIndex);

                const OrdinalType tNumberControls = mControlWork->operator[](tVectorIndex).size();
                locus::MultiVector<ScalarType, OrdinalType> & tCurrentSigma = mControlWork->operator[](tVectorIndex);
                for(OrdinalType tControlIndex = 0; tControlIndex < tNumberControls; tControlIndex++)
                {
                    ScalarType tValue = (tCurrentControl[tControlIndex] - tPreviousControl[tControlIndex])
                            * (tPreviousControl[tControlIndex] - tAntepenultimateControl[tControlIndex]);
                    if(tValue > static_cast<ScalarType>(0))
                    {
                        tCurrentSigma[tControlIndex] = tExpansionFactor * tPreviousSigma[tControlIndex];
                    }
                    else if(tValue < static_cast<ScalarType>(0))
                    {
                        tCurrentSigma[tControlIndex] = tContractionFactor * tPreviousSigma[tControlIndex];
                    }
                    else
                    {
                        tCurrentSigma[tControlIndex] = tPreviousSigma[tControlIndex];
                    }
                    // check that lower bound is satisfied
                    const ScalarType tLowerBoundScaleFactor = this->getMovingAsymptoteLowerBoundScaleFactor();
                    tValue = tLowerBoundScaleFactor * (tUpperBounds[tControlIndex] - tLowerBounds[tControlIndex]);
                    tCurrentSigma[tControlIndex] = std::max(tValue, tCurrentSigma[tControlIndex]);
                    // check that upper bound is satisfied
                    const ScalarType tUpperBoundScaleFactor = this->getMovingAsymptoteUpperBoundScaleFactor();
                    tValue = tUpperBoundScaleFactor * (tUpperBounds[tControlIndex] - tLowerBounds[tControlIndex]);
                    tCurrentSigma[tControlIndex] = std::min(tValue, tCurrentSigma[tControlIndex]);
                }
            }
            mDataMng->setCurrentSigma(mControlWork.operator*());
        }
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mOptimalityTolerance;
    ScalarType mStagnationTolerance;
    ScalarType mFeasibilityTolerance;
    ScalarType mStationarityTolerance;
    ScalarType mObjectiveStagnationTolerance;
    ScalarType mMovingAsymptoteExpansionFactor;
    ScalarType mMovingAsymptoteContractionFactor;
    ScalarType mKarushKuhnTuckerConditionsTolerance;
    ScalarType mMovingAsymptoteUpperBoundScaleFactor;
    ScalarType mMovingAsymptoteLowerBoundScaleFactor;

    locus::ccsa::stop_t mStoppingCriterion;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousSigma;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mAntepenultimateControl;
    std::shared_ptr<locus::MultiVectorList<ScalarType, OrdinalType>> mWorkMultiVectorList;

    std::shared_ptr<locus::PrimalProblemStageMng<ScalarType, OrdinalType>> mPrimalStageMng;
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType>> mDataMng;
    std::shared_ptr<locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>> mSubProblem;

private:
    ConservativeConvexSeparableApproximationsAlgorithm(const locus::ConservativeConvexSeparableApproximationsAlgorithm<ScalarType, OrdinalType> & aRhs);
    locus::ConservativeConvexSeparableApproximationsAlgorithm<ScalarType, OrdinalType> & operator=(const locus::ConservativeConvexSeparableApproximationsAlgorithm<ScalarType, OrdinalType> & aRhs);
};

}

/**********************************************************************************************************/
/*********************************************** UNIT TESTS ***********************************************/
/**********************************************************************************************************/



/**********************************************************************************************************/
/*********************************************** UNIT TESTS ***********************************************/
/**********************************************************************************************************/

namespace LocusTest
{

template<typename ScalarType, typename OrdinalType>
void printMultiVector(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
{
    std::cout << "\nPRINT MULTI-VECTOR\n" << std::flush;
    const OrdinalType tNumVectors = aInput.getNumVectors();
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        for(size_t tElementIndex = 0; tElementIndex < aInput[tVectorIndex].size(); tElementIndex++)
        {
            std::cout << "VectorIndex = " << tVectorIndex << ", Data(" << tVectorIndex << ", " << tElementIndex
                    << ") = " << aInput(tVectorIndex, tElementIndex) << "\n" << std::flush;
        }
    }
}

template<typename ScalarType, typename OrdinalType>
void checkVectorData(const locus::Vector<ScalarType, OrdinalType> & aInput,
                     const locus::Vector<ScalarType, OrdinalType> & aGold,
                     ScalarType aTolerance = 1e-6)
{
    assert(aInput.size() == aGold.size());

    OrdinalType tNumElements = aInput.size();
    for(OrdinalType tElemIndex = 0; tElemIndex < tNumElements; tElemIndex++)
    {
        EXPECT_NEAR(aInput[tElemIndex], aGold[tElemIndex], aTolerance);
    }
}

template<typename ScalarType, typename OrdinalType>
void checkMultiVectorData(const locus::MultiVector<ScalarType, OrdinalType> & aInput,
                          const locus::MultiVector<ScalarType, OrdinalType> & aGold,
                          ScalarType aTolerance = 1e-6)
{
    assert(aInput.getNumVectors() == aGold.getNumVectors());
    OrdinalType tNumVectors = aInput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        OrdinalType tNumElements = aInput[tVectorIndex].size();
        for(OrdinalType tElemIndex = 0; tElemIndex < tNumElements; tElemIndex++)
        {
            EXPECT_NEAR(aInput(tVectorIndex,tElemIndex), aGold(tVectorIndex,tElemIndex), aTolerance);
        }
    }
}

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

TEST(LocusTest, DualDataFactory)
{
    locus::DataFactory<double, size_t> tFactoryOne;

    // Test Factories for Dual Data
    size_t tNumVectors = 2;
    size_t tNumElements = 8;
    tFactoryOne.allocateDual(tNumElements, tNumVectors);

    size_t tGoldNumVectors = 2;
    EXPECT_EQ(tFactoryOne.dual().getNumVectors(), tGoldNumVectors);

    size_t tGoldNumElements = 8;
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryOne.dual(tIndex).size(), tGoldNumElements);
    }

    locus::StandardVectorReductionOperations<double,size_t> tInterface;
    tFactoryOne.allocateDualReductionOperations(tInterface);

    std::vector<double> tStandardVector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double,size_t> tlocusVector(tStandardVector);
    double tSum = tFactoryOne.getDualReductionOperations().sum(tlocusVector);
    double tGold = 55;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tSum, tGold, tTolerance);

    // Test Second Factory for Dual Data
    locus::DataFactory<double,size_t> tFactoryTwo;
    locus::StandardMultiVector<double,size_t> tMultiVector(tNumVectors, tlocusVector);
    tFactoryTwo.allocateDual(tMultiVector);

    tGoldNumElements = 10;
    EXPECT_EQ(tFactoryTwo.dual().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryTwo.dual(tIndex).size(), tGoldNumElements);
    }

    // Test Third Factory for Dual Data
    locus::DataFactory<double,size_t> tFactoryThree;
    tFactoryThree.allocateDual(tlocusVector, tNumVectors);

    EXPECT_EQ(tFactoryThree.dual().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryThree.dual(tIndex).size(), tGoldNumElements);
    }

    // Test Fourth Factory for Dual Data (Default NumVectors = 1)
    locus::DataFactory<double,size_t> tFactoryFour;
    tFactoryFour.allocateDual(tlocusVector);

    tNumVectors = 1;
    tGoldNumVectors = 1;
    EXPECT_EQ(tFactoryFour.dual().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryFour.dual(tIndex).size(), tGoldNumElements);
    }
}

TEST(LocusTest, StateDataFactory)
{
    locus::DataFactory<double,size_t> tFactoryOne;

    // Test Factories for State Data
    size_t tNumVectors = 2;
    size_t tNumElements = 8;
    tFactoryOne.allocateState(tNumElements, tNumVectors);

    size_t tGoldNumVectors = 2;
    EXPECT_EQ(tFactoryOne.state().getNumVectors(), tGoldNumVectors);

    size_t tGoldNumElements = 8;
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryOne.state(tIndex).size(), tGoldNumElements);
    }

    locus::StandardVectorReductionOperations<double,size_t> tInterface;
    tFactoryOne.allocateStateReductionOperations(tInterface);

    std::vector<double> tStandardVector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double,size_t> tlocusVector(tStandardVector);
    double tSum = tFactoryOne.getStateReductionOperations().sum(tlocusVector);
    double tGold = 55;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tSum, tGold, tTolerance);

    // Test Second Factory for State Data
    locus::DataFactory<double,size_t> tFactoryTwo;
    locus::StandardMultiVector<double,size_t> tMultiVector(tNumVectors, tlocusVector);
    tFactoryTwo.allocateState(tMultiVector);

    tGoldNumElements = 10;
    EXPECT_EQ(tFactoryTwo.state().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryTwo.state(tIndex).size(), tGoldNumElements);
    }

    // Test Third Factory for State Data
    locus::DataFactory<double,size_t> tFactoryThree;
    tFactoryThree.allocateState(tlocusVector, tNumVectors);

    EXPECT_EQ(tFactoryThree.state().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryThree.state(tIndex).size(), tGoldNumElements);
    }

    // Test Fourth Factory for State Data (Default NumVectors = 1)
    locus::DataFactory<double,size_t> tFactoryFour;
    tFactoryFour.allocateState(tlocusVector);

    tNumVectors = 1;
    tGoldNumVectors = 1;
    EXPECT_EQ(tFactoryFour.state().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryFour.state(tIndex).size(), tGoldNumElements);
    }
}

TEST(LocusTest, ControlDataFactory)
{
    locus::DataFactory<double,size_t> tFactoryOne;

    // ********* Test Factories for Control Data *********
    size_t tNumVectors = 2;
    size_t tNumElements = 8;
    tFactoryOne.allocateControl(tNumElements, tNumVectors);

    size_t tGoldNumVectors = 2;
    EXPECT_EQ(tFactoryOne.control().getNumVectors(), tGoldNumVectors);

    size_t tGoldNumElements = 8;
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryOne.control(tIndex).size(), tGoldNumElements);
    }

    locus::StandardVectorReductionOperations<double,size_t> tInterface;
    tFactoryOne.allocateControlReductionOperations(tInterface);

    std::vector<double> tStandardVector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double,size_t> tlocusVector(tStandardVector);
    double tSum = tFactoryOne.getControlReductionOperations().sum(tlocusVector);
    double tGold = 55;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tSum, tGold, tTolerance);

    // ********* Test Second Factory for Control Data *********
    locus::DataFactory<double,size_t> tFactoryTwo;
    locus::StandardMultiVector<double,size_t> tMultiVector(tNumVectors, tlocusVector);
    tFactoryTwo.allocateControl(tMultiVector);

    tGoldNumElements = 10;
    EXPECT_EQ(tFactoryTwo.control().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryTwo.control(tIndex).size(), tGoldNumElements);
    }

    // ********* Test Third Factory for Control Data *********
    locus::DataFactory<double,size_t> tFactoryThree;
    tFactoryThree.allocateControl(tlocusVector, tNumVectors);

    EXPECT_EQ(tFactoryThree.control().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryThree.control(tIndex).size(), tGoldNumElements);
    }

    // ********* Test Fourth Factory for Control Data (Default NumVectors = 1) *********
    locus::DataFactory<double,size_t> tFactoryFour;
    tFactoryFour.allocateControl(tlocusVector);

    tNumVectors = 1;
    tGoldNumVectors = 1;
    EXPECT_EQ(tFactoryFour.control().getNumVectors(), tGoldNumVectors);
    for(size_t tIndex = 0; tIndex < tNumVectors; tIndex++)
    {
        EXPECT_EQ(tFactoryFour.control(tIndex).size(), tGoldNumElements);
    }
}

TEST(LocusTest, OptimalityCriteriaObjectiveTest)
{
    locus::StandardVectorReductionOperations<double,size_t> tInterface;
    locus::OptimalityCriteriaTestObjectiveOne<double,size_t> tObjective(tInterface);

    size_t tNumVectors = 1;
    size_t tNumElements = 5;
    std::vector<double> tData(tNumElements, 0.);
    locus::StandardMultiVector<double,size_t> tState(tNumVectors, tData);
    locus::StandardMultiVector<double,size_t> tControl(tNumVectors, tData);

    // ********* Set Control Data For Test *********
    const size_t tVectorIndex = 0;
    tData = { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        tControl(tVectorIndex, tIndex) = tData[tIndex];
    }

    // ********* Test Objective Value *********
    double tObjectiveValue = tObjective.value(tState, tControl);

    double tTolerance = 1e-6;
    double tGoldValue = 1.3401885069;
    EXPECT_NEAR(tObjectiveValue, tGoldValue, tTolerance);

    // ********* Test Objective Gradient *********
    locus::StandardMultiVector<double,size_t> tGradient(tNumVectors, tData);
    tObjective.gradient(tState, tControl, tGradient);

    std::vector<double> tGoldGradient(tNumElements, 0.);
    std::fill(tGoldGradient.begin(), tGoldGradient.end(), 0.0624);
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        EXPECT_NEAR(tGradient(tVectorIndex, tIndex), tGoldGradient[tIndex], tTolerance);
    }
}

TEST(LocusTest, OptimalityCriteriaInequalityTestOne)
{
    locus::OptimalityCriteriaTestInequalityOne<double,size_t> tInequality;

    size_t tNumVectors = 1;
    size_t tNumElements = 5;
    std::vector<double> tData(tNumElements, 0.);
    locus::StandardMultiVector<double,size_t> tState(tNumVectors, tData);
    locus::StandardMultiVector<double,size_t> tControl(tNumVectors, tData);

    // ********* Set Control Data For Test *********
    const size_t tVectorIndex = 0;
    tData = { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        tControl(tVectorIndex, tIndex) = tData[tIndex];
    }

    double tInequalityValue = tInequality.value(tState, tControl);

    double tTolerance = 1e-6;
    double tGoldValue = -5.07057774290498e-6;
    EXPECT_NEAR(tInequalityValue, tGoldValue, tTolerance);

    // ********* Test Inequality Gradient *********
    locus::StandardMultiVector<double,size_t> tGradient(tNumVectors, tData);
    tInequality.gradient(tState, tControl, tGradient);

    std::vector<double> tGoldGradient =
            { -0.13778646890793422, -0.14864537557631985, -0.13565219858574704, -0.1351771199123859, -0.13908690613190111 };
    for(size_t tIndex = 0; tIndex < tData.size(); tIndex++)
    {
        EXPECT_NEAR(tGradient(tVectorIndex, tIndex), tGoldGradient[tIndex], tTolerance);
    }
}

TEST(LocusTest, OptimalityCriteriaDataMng)
{
    // ********* Test Factories for Dual Data *********
    size_t tNumVectors = 1;
    locus::DataFactory<double,size_t> tFactory;

    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 10;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 5;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double,size_t> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    locus::OptimalityCriteriaDataMng<double,size_t> tDataMng(tFactory);
    double tValue = 23;
    tDataMng.setCurrentObjectiveValue(tValue);

    double tGold = tDataMng.getCurrentObjectiveValue();
    double tTolerance = 1e-6;
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tValue = 24;
    tDataMng.setPreviousObjectiveValue(tValue);
    tGold = tDataMng.getPreviousObjectiveValue();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // ********* Test Dual Functions *********
    tValue = 0.23;
    size_t tIndex = 0;
    tDataMng.setCurrentDual(tIndex, tValue);
    tGold = 0.23;
    EXPECT_NEAR(tDataMng.getCurrentDual()[tIndex], tGold, tTolerance);

    tValue = 0.345;
    tDataMng.setCurrentConstraintValue(tIndex, tValue);
    tGold = 0.345;
    EXPECT_NEAR(tDataMng.getCurrentConstraintValues()[tIndex], tGold, tTolerance);

    // ********* Test Initial Guess Functions *********
    tValue = 0.18;
    locus::StandardMultiVector<double,size_t> tInitialGuess(tNumVectors, tNumControls, tValue);
    tDataMng.setInitialGuess(tInitialGuess);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tInitialGuess, tTolerance);

    tValue = 0.44;
    size_t tVectorIndex = 0;
    tInitialGuess[tVectorIndex].fill(tValue);
    tDataMng.setInitialGuess(tVectorIndex, tInitialGuess[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tInitialGuess.operator [](tVectorIndex));

    tValue = 0.07081982;
    tInitialGuess[tVectorIndex].fill(tValue);
    tDataMng.setInitialGuess(tVectorIndex, tValue);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tInitialGuess.operator [](tVectorIndex));

    tValue = 0.10111983;
    tInitialGuess[tVectorIndex].fill(tValue);
    tDataMng.setInitialGuess(tValue);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tInitialGuess.operator [](tVectorIndex));

    // ********* Test Control Functions *********
    tValue = 0.08;
    locus::StandardMultiVector<double,size_t> tCurrentControl(tNumVectors, tNumControls, tValue);
    tDataMng.setCurrentControl(tCurrentControl);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tCurrentControl, tTolerance);

    tValue = 0.11;
    tCurrentControl[tVectorIndex].fill(tValue);
    tDataMng.setCurrentControl(tVectorIndex, tCurrentControl[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tCurrentControl.operator [](tVectorIndex));

    tValue = 0.09;
    locus::StandardMultiVector<double,size_t> tPreviousControl(tNumVectors, tNumControls, tValue);
    tDataMng.setPreviousControl(tPreviousControl);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tPreviousControl, tTolerance);

    tValue = 0.21;
    tPreviousControl[tVectorIndex].fill(tValue);
    tDataMng.setPreviousControl(tVectorIndex, tPreviousControl[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tPreviousControl.operator [](tVectorIndex));

    // ********* Test Objective Gradient Functions *********
    tValue = 0.88;
    locus::StandardMultiVector<double,size_t> tObjectiveGradient(tNumVectors, tNumControls, tValue);
    tDataMng.setObjectiveGradient(tObjectiveGradient);
    LocusTest::checkMultiVectorData(tDataMng.getObjectiveGradient(), tObjectiveGradient, tTolerance);

    tValue = 0.91;
    tObjectiveGradient[tVectorIndex].fill(tValue);
    tDataMng.setObjectiveGradient(tVectorIndex, tObjectiveGradient[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getObjectiveGradient(tVectorIndex), tObjectiveGradient.operator [](tVectorIndex));

    // ********* Test Inequality Gradient Functions *********
    tValue = 0.68;
    locus::StandardMultiVector<double,size_t> tInequalityGradient(tNumVectors, tNumControls, tValue);
    tDataMng.setInequalityGradient(tInequalityGradient);
    LocusTest::checkMultiVectorData(tDataMng.getInequalityGradient(), tInequalityGradient, tTolerance);

    tValue = 0.61;
    tInequalityGradient[tVectorIndex].fill(tValue);
    tDataMng.setInequalityGradient(tVectorIndex, tInequalityGradient[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getInequalityGradient(tVectorIndex), tInequalityGradient.operator [](tVectorIndex));

    // ********* Test Control Lower Bounds Functions *********
    tValue = 1e-3;
    locus::StandardMultiVector<double,size_t> tLowerBounds(tNumVectors, tNumControls, tValue);
    tDataMng.setControlLowerBounds(tLowerBounds);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tLowerBounds, tTolerance);

    tValue = 1e-2;
    tLowerBounds[tVectorIndex].fill(tValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tLowerBounds[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tLowerBounds.operator [](tVectorIndex));

    tValue = -1;
    tDataMng.setControlLowerBounds(tVectorIndex, tValue);
    tLowerBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tLowerBounds.operator [](tVectorIndex));

    tValue = 0.5;
    tDataMng.setControlLowerBounds(tVectorIndex, tValue);
    tLowerBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tLowerBounds.operator [](tVectorIndex));

    // ********* Test Control Upper Bounds Functions *********
    tValue = 1;
    locus::StandardMultiVector<double,size_t> tUpperBounds(tNumVectors, tNumControls, tValue);
    tDataMng.setControlUpperBounds(tUpperBounds);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tUpperBounds, tTolerance);

    tValue = 0.99;
    tUpperBounds[tVectorIndex].fill(tValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tUpperBounds[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tUpperBounds.operator [](tVectorIndex));

    tValue = 10;
    tDataMng.setControlUpperBounds(tVectorIndex, tValue);
    tUpperBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tUpperBounds.operator [](tVectorIndex));

    tValue = 8;
    tDataMng.setControlUpperBounds(tVectorIndex, tValue);
    tUpperBounds[tVectorIndex].fill(tValue);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tUpperBounds.operator [](tVectorIndex));

    // ********* Test Compute Stagnation Measure Functions *********
    tCurrentControl[tVectorIndex].fill(1.5);
    tDataMng.setCurrentControl(tCurrentControl);
    tPreviousControl[tVectorIndex].fill(4.0);
    tDataMng.setPreviousControl(tPreviousControl);
    tDataMng.computeStagnationMeasure();

    tGold = 2.5;
    tValue = tDataMng.getStagnationMeasure();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // ********* Test Compute Max Inequality Value Functions *********
    tDataMng.computeMaxInequalityValue();
    tGold = 0.345;
    tValue = tDataMng.getMaxInequalityValue();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // ********* Test Compute Max Inequality Value Functions *********
    tDataMng.computeNormObjectiveGradient();
    tGold = 2.0348218595248;
    tValue = tDataMng.getNormObjectiveGradient();
    EXPECT_NEAR(tValue, tGold, tTolerance);
}

TEST(LocusTest, OptimalityCriteriaStageMngSimpleTest)
{
    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumVectors = 1;
    locus::DataFactory<double,size_t> tFactory;
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 1;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 5;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double,size_t> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Optimality Criteria Data Manager *********
    locus::OptimalityCriteriaDataMng<double,size_t> tDataMng(tFactory);

    // ********* Allocate Stage Manager *********
    locus::CriterionList<double, size_t> tInequalityList;
    locus::OptimalityCriteriaTestInequalityOne<double,size_t> tInequality;
    tInequalityList.add(tInequality);
    locus::OptimalityCriteriaTestObjectiveOne<double,size_t> tObjective(tReductionOperations);
    locus::OptimalityCriteriaStageMng<double,size_t> tStageMng(tFactory, tObjective, tInequalityList);

    // ********* Test Update Function *********
    std::vector<double> tData =
        { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    locus::StandardVector<double,size_t> tControl(tData);

    size_t tVectorIndex = 0;
    tDataMng.setCurrentControl(tVectorIndex, tControl);
    tStageMng.update(tDataMng);

    double tTolerance = 1e-6;
    double tGoldValue = 1.3401885069;
    double tObjectiveValue = tDataMng.getCurrentObjectiveValue();
    EXPECT_NEAR(tObjectiveValue, tGoldValue, tTolerance);

    std::fill(tData.begin(), tData.end(), 0.0624);
    locus::StandardVector<double,size_t> tGoldObjectiveGradient(tData);
    LocusTest::checkVectorData(tDataMng.getObjectiveGradient(tVectorIndex), tGoldObjectiveGradient);

    const size_t tConstraintIndex = 0;
    tGoldValue = -5.07057774290498e-6;
    const locus::MultiVector<double, size_t> & tCurrentControl = tDataMng.getCurrentControl();
    double tInequalityValue = tStageMng.evaluateInequality(tConstraintIndex, tCurrentControl);
    EXPECT_NEAR(tInequalityValue, tGoldValue, tTolerance);

    tStageMng.computeInequalityGradient(tDataMng);
    tData = { -0.13778646890793422, -0.14864537557631985, -0.13565219858574704, -0.1351771199123859, -0.13908690613190111 };
    locus::StandardVector<double,size_t> tGoldInequalityGradient(tData);
    LocusTest::checkVectorData(tDataMng.getInequalityGradient(tVectorIndex), tGoldInequalityGradient);
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

TEST(LocusTest, SynthesisOptimizationSubProblem)
{
    // ********* NOTE: Default OrdinalType = size_t *********
    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumVectors = 1;
    locus::DataFactory<double> tFactory;
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 1;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 2;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Optimality Criteria Data Manager *********
    locus::OptimalityCriteriaDataMng<double> tDataMng(tFactory);

    // ********* Allocate Synthesis Optimization Sub-Problem  *********
    locus::SynthesisOptimizationSubProblem<double> tSubProblem(tDataMng);

    double tGold = 1e-4;
    double tValue = tSubProblem.getBisectionTolerance();
    double tTolerance = 1e-6;
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setBisectionTolerance(1e-1);
    tGold = 0.1;
    tValue = tSubProblem.getBisectionTolerance();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 0.01;
    tValue = tSubProblem.getMoveLimit();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setMoveLimit(0.15);
    tGold = 0.15;
    tValue = tSubProblem.getMoveLimit();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 0.5;
    tValue = tSubProblem.getDampingPower();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setDampingPower(0.25);
    tGold = 0.25;
    tValue = tSubProblem.getDampingPower();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 0;
    tValue = tSubProblem.getDualLowerBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setDualLowerBound(0.35);
    tGold = 0.35;
    tValue = tSubProblem.getDualLowerBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    tGold = 1e4;
    tValue = tSubProblem.getDualUpperBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);
    tSubProblem.setDualUpperBound(0.635);
    tGold = 0.635;
    tValue = tSubProblem.getDualUpperBound();
    EXPECT_NEAR(tValue, tGold, tTolerance);

    // NOTE: I NEED TO UNIT TEST SUBPROBLEM WITH PHYSICS-BASED CRITERIA
}

TEST(LocusTest, OptimalityCriteria)
{
    // ********* NOTE: Default OrdinalType = size_t *********

    // ********* Allocate Core Optimization Data Templates *********
    size_t tNumVectors = 1;
    locus::DataFactory<double> tFactory;
    size_t tNumDual = 1;
    tFactory.allocateDual(tNumDual, tNumVectors);
    size_t tNumStates = 1;
    tFactory.allocateState(tNumStates, tNumVectors);
    size_t tNumControls = 2;
    tFactory.allocateControl(tNumControls, tNumVectors);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tFactory.allocateDualReductionOperations(tReductionOperations);
    tFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Optimality Criteria Data Manager *********
    std::shared_ptr<locus::OptimalityCriteriaDataMng<double>> tDataMng =
            std::make_shared<locus::OptimalityCriteriaDataMng<double>>(tFactory);

    // ********* Set Bounds and Initial Guess *********
    double tValue = 0.5;
    tDataMng->setControlLowerBounds(tValue);
    tValue = 10;
    tDataMng->setControlUpperBounds(tValue);
    tValue = 1;
    tDataMng->setInitialGuess(tValue);

    // ********* Allocate Stage Manager *********
    locus::CriterionList<double> tInequalityList;
    locus::OptimalityCriteriaTestInequalityTwo<double> tInequality;
    tInequalityList.add(tInequality);
    locus::OptimalityCriteriaTestObjectiveTwo<double> tObjective;
    std::shared_ptr<locus::OptimalityCriteriaStageMng<double>> tStageMng =
            std::make_shared<locus::OptimalityCriteriaStageMng<double>>(tFactory, tObjective, tInequalityList);

    // ********* Allocate Optimality Criteria Algorithm *********
    std::shared_ptr<locus::SingleConstraintSubProblemTypeLP<double>> tSubProlem =
            std::make_shared<locus::SingleConstraintSubProblemTypeLP<double>>(*tDataMng);
    locus::OptimalityCriteria<double> tOptimalityCriteria(tDataMng, tStageMng, tSubProlem);
    tOptimalityCriteria.solve();

    size_t tVectorIndex = 0;
    const locus::Vector<double> & tControl = tDataMng->getCurrentControl(tVectorIndex);
    double tTolerance = 1e-6;
    double tGoldControlOne = 0.5;
    EXPECT_NEAR(tControl[0], tGoldControlOne, tTolerance);
    double tGoldControlTwo = 1.375;
    EXPECT_NEAR(tControl[1], tGoldControlTwo, tTolerance);
    size_t tGoldNumIterations = 5;
    EXPECT_EQ(tOptimalityCriteria.getNumIterationsDone(), tGoldNumIterations);
}

/* ******************************************************************* */
/* ***************** AUGMENTED LAGRANGIAN UNIT TESTS ***************** */
/* ******************************************************************* */

TEST(LocusTest, Project)
{
    // ********* Allocate Input Data *********
    const size_t tNumVectors = 8;
    std::vector<double> tVectorGold = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tlocusVector(tVectorGold);
    // Default for second template typename is OrdinalType = size_t
    locus::StandardMultiVector<double> tData(tNumVectors, tlocusVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tData[tVectorIndex].update(1., tlocusVector, 0.);
    }

    // ********* Allocate Lower & Upper Bounds *********
    const double tLowerBoundValue = 2;
    const size_t tNumElementsPerVector = tVectorGold.size();
    locus::StandardMultiVector<double> tLowerBounds(tNumVectors, tNumElementsPerVector, tLowerBoundValue);
    const double tUpperBoundValue = 7;
    locus::StandardMultiVector<double> tUpperBounds(tNumVectors, tNumElementsPerVector, tUpperBoundValue);

    locus::bounds::project(tLowerBounds, tUpperBounds, tData);

    std::vector<double> tVectorBoundsGold = { 2, 2, 3, 4, 5, 6, 7, 7, 7, 7 };
    locus::StandardVector<double> tlocusBoundVector(tVectorBoundsGold);
    locus::StandardMultiVector<double> tGoldData(tNumVectors, tlocusBoundVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tGoldData[tVectorIndex].update(1., tlocusBoundVector, 0.);
    }
    LocusTest::checkMultiVectorData(tData, tGoldData);
}

TEST(LocusTest, CheckBounds)
{
    // ********* Allocate Lower & Upper Bounds *********
    const size_t tNumVectors = 1;
    const size_t tNumElements = 5;
    double tLowerBoundValue = 2;
    locus::StandardMultiVector<double> tLowerBounds(tNumVectors, tNumElements, tLowerBoundValue);
    double tUpperBoundValue = 7;
    locus::StandardMultiVector<double> tUpperBounds(tNumVectors, tNumElements, tUpperBoundValue);
    ASSERT_NO_THROW(locus::bounds::checkBounds(tLowerBounds, tUpperBounds));

    tUpperBoundValue = 2;
    locus::fill(tUpperBoundValue, tUpperBounds);
    ASSERT_THROW(locus::bounds::checkBounds(tLowerBounds, tUpperBounds), std::invalid_argument);
}

TEST(LocusTest, ComputeActiveAndInactiveSet)
{
    // ********* Allocate Input Data *********
    const size_t tNumVectors = 4;
    std::vector<double> tVectorGold = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    locus::StandardVector<double> tlocusVector(tVectorGold);
    // Default for second template typename is OrdinalType = size_t
    locus::StandardMultiVector<double> tData(tNumVectors, tlocusVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tData[tVectorIndex].update(1., tlocusVector, 0.);
    }

    // ********* Allocate Lower & Upper Bounds *********
    const double tLowerBoundValue = 2;
    const size_t tNumElementsPerVector = tVectorGold.size();
    locus::StandardMultiVector<double> tLowerBounds(tNumVectors, tNumElementsPerVector, tLowerBoundValue);
    const double tUpperBoundValue = 7;
    locus::StandardMultiVector<double> tUpperBounds(tNumVectors, tNumElementsPerVector, tUpperBoundValue);

    // ********* Allocate Active & Inactive Sets *********
    locus::StandardMultiVector<double> tActiveSet(tNumVectors, tNumElementsPerVector, tUpperBoundValue);
    locus::StandardMultiVector<double> tInactiveSet(tNumVectors, tNumElementsPerVector, tUpperBoundValue);

    // ********* Compute Active & Inactive Sets *********
    locus::bounds::project(tLowerBounds, tUpperBounds, tData);
    locus::bounds::computeActiveAndInactiveSets(tData, tLowerBounds, tUpperBounds, tActiveSet, tInactiveSet);

    std::vector<double> tActiveSetGold = { 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
    std::vector<double> tInactiveSetGold = { 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 };
    locus::StandardVector<double> tlocusActiveSetVectorGold(tActiveSetGold);
    locus::StandardVector<double> tlocusInactiveSetVectorGold(tInactiveSetGold);
    locus::StandardMultiVector<double> tActiveSetGoldData(tNumVectors, tlocusActiveSetVectorGold);
    locus::StandardMultiVector<double> tInactiveSetGoldData(tNumVectors, tlocusInactiveSetVectorGold);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        tActiveSetGoldData[tVectorIndex].update(1., tlocusActiveSetVectorGold, 0.);
        tInactiveSetGoldData[tVectorIndex].update(1., tlocusInactiveSetVectorGold, 0.);
    }
    LocusTest::checkMultiVectorData(tActiveSet, tActiveSetGoldData);
    LocusTest::checkMultiVectorData(tInactiveSet, tInactiveSetGoldData);
}

TEST(LocusTest, TrustRegionAlgorithmDataMng)
{
    // ********* Test Factories for Dual Data *********
    locus::DataFactory<double> tDataFactory;

    // ********* Allocate Core Optimization Data Templates *********
    const size_t tNumDuals = 10;
    const size_t tNumDualVectors = 2;
    tDataFactory.allocateDual(tNumDuals, tNumDualVectors);
    const size_t tNumStates = 20;
    const size_t tNumStateVectors = 6;
    tDataFactory.allocateState(tNumStates, tNumStateVectors);
    const size_t tNumControls = 5;
    const size_t tNumControlVectors = 3;
    tDataFactory.allocateControl(tNumControls, tNumControlVectors);

    // ********* Allocate Reduction Operations *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateDualReductionOperations(tReductionOperations);
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Trust Region Algorithm Data Manager *********
    locus::TrustRegionAlgorithmDataMng<double> tDataMng(tDataFactory);

    // ********* Test Trust Region Algorithm Data Manager *********
    // TEST NUMBER OF VECTORS FUNCTIONS
    EXPECT_EQ(tNumDualVectors, tDataMng.getNumDualVectors());
    EXPECT_EQ(tNumControlVectors, tDataMng.getNumControlVectors());

    // TEST CURRENT OBJECTIVE FUNCTION VALUE INTERFACES
    double tGoldValue = std::numeric_limits<double>::max();
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tGoldValue = 0.123;
    tDataMng.setCurrentObjectiveFunctionValue(0.123);
    EXPECT_NEAR(tGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    // TEST PREVIOUS OBJECTIVE FUNCTION VALUE INTERFACES
    tGoldValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tGoldValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tGoldValue = 0.321;
    tDataMng.setPreviousObjectiveFunctionValue(0.321);
    EXPECT_NEAR(tGoldValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // 1) TEST INITIAL GUESS INTERFACES
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    double tValue = 0.5;
    tDataMng.setInitialGuess(0.5);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::StandardVector<double> tlocusControlVector(tNumControls, tValue);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    // 2) TEST INITIAL GUESS INTERFACES
    tValue = 0.3;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(0.1);
        tlocusControlVector.fill(tValue);
        tDataMng.setInitialGuess(tVectorIndex, tValue);
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    // 3) TEST INITIAL GUESS INTERFACES
    tValue = 0;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(0.1);
        tlocusControlVector.fill(tValue);
        tDataMng.setInitialGuess(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    // 4) TEST INITIAL GUESS INTERFACES
    locus::StandardMultiVector<double> tlocusControlMultiVector(tNumControlVectors, tlocusControlVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setInitialGuess(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector, tTolerance);

    // TEST DUAL VECTOR INTERFACES
    locus::StandardVector<double> tlocusDualVector(tNumDuals);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusDualVector.fill(tValue);
        tDataMng.setDual(tVectorIndex, tlocusDualVector);
        LocusTest::checkVectorData(tDataMng.getDual(tVectorIndex), tlocusDualVector, tTolerance);
    }

    tValue = 20;
    locus::StandardMultiVector<double> tlocusDualMultiVector(tNumDualVectors, tlocusDualVector);
    for(size_t tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusDualMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setDual(tlocusDualMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getDual(), tlocusDualMultiVector, tTolerance);

    // TEST TRIAL STEP INTERFACES
    tValue = 3;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setTrialStep(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setTrialStep(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tlocusControlMultiVector, tTolerance);

    // TEST ACTIVE SET INTERFACES
    tValue = 33;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setActiveSet(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getActiveSet(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setActiveSet(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getActiveSet(), tlocusControlMultiVector, tTolerance);

    // TEST INACTIVE SET INTERFACES
    tValue = 23;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setInactiveSet(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getInactiveSet(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getInactiveSet(), tlocusControlMultiVector, tTolerance);

    // TEST CURRENT CONTROL INTERFACES
    tValue = 30;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setCurrentControl(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector, tTolerance);

    // TEST PREVIOUS CONTROL INTERFACES
    tValue = 80;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setPreviousControl(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setPreviousControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector, tTolerance);

    // TEST CURRENT GRADIENT INTERFACES
    tValue = 7882;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setCurrentGradient(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getCurrentGradient(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector, tTolerance);

    // TEST PREVIOUS GRADIENT INTERFACES
    tValue = 101183;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setPreviousGradient(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getPreviousGradient(tVectorIndex), tlocusControlVector, tTolerance);
    }

    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tlocusControlMultiVector[tVectorIndex].fill(tVectorIndex);
    }
    tDataMng.setPreviousGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector, tTolerance);

    // TEST CONTROL LOWER BOUND INTERFACES
    tValue = -std::numeric_limits<double>::max();
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tlocusControlMultiVector, tTolerance);
    tValue = 1e-3;
    tDataMng.setControlLowerBounds(tValue);
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tlocusControlMultiVector, tTolerance);

    tValue = 1e-4;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + (static_cast<double>(tVectorIndex) * tValue);
        tDataMng.setControlLowerBounds(tVectorIndex, tValue);
        tlocusControlVector.fill(tValue);
        LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setControlLowerBounds(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1e-2;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue * static_cast<double>(tVectorIndex + 1);
        tlocusControlMultiVector[tVectorIndex].fill(tValue);
    }
    tDataMng.setControlLowerBounds(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tlocusControlMultiVector, tTolerance);

    // TEST CONTROL UPPER BOUND INTERFACES
    tValue = std::numeric_limits<double>::max();
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tlocusControlMultiVector, tTolerance);
    tValue = 1e-3;
    tDataMng.setControlUpperBounds(tValue);
    locus::fill(tValue, tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tlocusControlMultiVector, tTolerance);

    tValue = 1e-4;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + (static_cast<double>(tVectorIndex) * tValue);
        tDataMng.setControlUpperBounds(tVectorIndex, tValue);
        tlocusControlVector.fill(tValue);
        LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tValue);
        tDataMng.setControlUpperBounds(tVectorIndex, tlocusControlVector);
        LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tlocusControlVector, tTolerance);
    }

    tValue = 1e-2;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        tValue = tValue * static_cast<double>(tVectorIndex + 1);
        tlocusControlMultiVector[tVectorIndex].fill(tValue);
    }
    tDataMng.setControlUpperBounds(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tlocusControlMultiVector, tTolerance);

    // TEST GRADIENT INEXACTNESS FLAG FUNCTIONS
    EXPECT_FALSE(tDataMng.isGradientInexactnessToleranceExceeded());
    tDataMng.setGradientInexactnessFlag(true);
    EXPECT_TRUE(tDataMng.isGradientInexactnessToleranceExceeded());
    tDataMng.setGradientInexactnessFlag(false);
    EXPECT_FALSE(tDataMng.isGradientInexactnessToleranceExceeded());

    // TEST OBJECTIVE INEXACTNESS FLAG FUNCTIONS
    EXPECT_FALSE(tDataMng.isObjectiveInexactnessToleranceExceeded());
    tDataMng.setObjectiveInexactnessFlag(true);
    EXPECT_TRUE(tDataMng.isObjectiveInexactnessToleranceExceeded());
    tDataMng.setObjectiveInexactnessFlag(false);
    EXPECT_FALSE(tDataMng.isObjectiveInexactnessToleranceExceeded());

    // TEST COMPUTE STAGNATION MEASURE FUNCTION
    tValue = 1e-2;
    for(size_t tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
    {
        double tCurrentValue = tValue + static_cast<double>(tVectorIndex);
        tlocusControlVector.fill(tCurrentValue);
        tDataMng.setCurrentControl(tVectorIndex, tlocusControlVector);
        double tPreviousValue = tValue * static_cast<double>(tVectorIndex * tVectorIndex);
        tlocusControlVector.fill(tPreviousValue);
        tDataMng.setPreviousControl(tVectorIndex, tlocusControlVector);
    }
    tValue = 1.97;
    tDataMng.computeStagnationMeasure();
    EXPECT_NEAR(tValue, tDataMng.getStagnationMeasure(), tTolerance);

    // TEST COMPUTE NORM OF PROJECTED VECTOR
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    locus::fill(2., tlocusControlMultiVector);
    tValue = 7.745966692414834;
    EXPECT_NEAR(tValue, tDataMng.computeProjectedVectorNorm(tlocusControlMultiVector), tTolerance);

    locus::fill(1., tlocusControlMultiVector);
    size_t tVectorIndex = 1;
    size_t tElementIndex = 2;
    tlocusControlMultiVector(tVectorIndex, tElementIndex) = 0;
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    locus::fill(2., tlocusControlMultiVector);
    tValue = 7.483314773547883;
    EXPECT_NEAR(tValue, tDataMng.computeProjectedVectorNorm(tlocusControlMultiVector), tTolerance);

    // TEST COMPUTE PROJECTED GRADIENT NORM
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    locus::fill(3., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    tDataMng.computeNormProjectedGradient();
    tValue = 11.61895003862225;
    EXPECT_NEAR(tValue, tDataMng.getNormProjectedGradient(), tTolerance);

    locus::fill(1., tlocusControlMultiVector);
    tVectorIndex = 0;
    tElementIndex = 0;
    tlocusControlMultiVector(tVectorIndex, tElementIndex) = 0;
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    tDataMng.computeNormProjectedGradient();
    tValue = 11.224972160321824;
    EXPECT_NEAR(tValue, tDataMng.getNormProjectedGradient(), tTolerance);

    // TEST COMPUTE STATIONARY MEASURE
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setInactiveSet(tlocusControlMultiVector);
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    locus::fill(-2., tlocusControlMultiVector);
    tDataMng.setControlLowerBounds(tlocusControlMultiVector);
    locus::fill(12., tlocusControlMultiVector);
    tDataMng.setControlUpperBounds(tlocusControlMultiVector);
    locus::fill(-1., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    tDataMng.computeStationarityMeasure();
    tValue = 3.872983346207417;
    EXPECT_NEAR(tValue, tDataMng.getStationarityMeasure(), tTolerance);

    // TEST RESET STAGE FUNCTION
    tValue = 1;
    tDataMng.setCurrentObjectiveFunctionValue(tValue);
    EXPECT_NEAR(tValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tValue = 2;
    tDataMng.setPreviousObjectiveFunctionValue(tValue);
    EXPECT_NEAR(tValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector);
    locus::fill(-2., tlocusControlMultiVector);
    tDataMng.setPreviousControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector);
    locus::fill(10., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector);
    locus::fill(-12., tlocusControlMultiVector);
    tDataMng.setPreviousGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector);

    tDataMng.resetCurrentStageDataToPreviousStageData();

    tValue = 2;
    EXPECT_NEAR(tValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    locus::fill(-2., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector);
    locus::fill(-12., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector);

    // TEST STORE CURRENT STAGE DATA FUNCTION
    tValue = 1;
    tDataMng.setCurrentObjectiveFunctionValue(tValue);
    EXPECT_NEAR(tValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tValue = 2;
    EXPECT_NEAR(tValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    locus::fill(1., tlocusControlMultiVector);
    tDataMng.setCurrentControl(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tlocusControlMultiVector);
    locus::fill(-2., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector);
    locus::fill(10., tlocusControlMultiVector);
    tDataMng.setCurrentGradient(tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tlocusControlMultiVector);
    locus::fill(-12., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector);

    tDataMng.storeCurrentStageData();

    tValue = 1;
    EXPECT_NEAR(tValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    locus::fill(1., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tlocusControlMultiVector);
    locus::fill(10., tlocusControlMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tlocusControlMultiVector);
}

TEST(LocusTest, RosenbrockCriterion)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    double tValue = 2;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Rosenbrock<double> tCriterion;
    // TEST OBJECTIVE FUNCTION EVALUATION
    double tObjectiveValue = tCriterion.value(tState, tControl);
    const double tGoldValue = 401;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tObjectiveValue, tTolerance);

    // TEST GRADIENT EVALUATION FUNCTION
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 0) = 1602;
    tGoldVector(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST HESSIAN TIMES VECTOR FUNCTION
    tValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tCriterion.hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 3202;
    tGoldVector(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, CircleCriterion)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Circle<double> tCriterion;

    // TEST OBJECTIVE FUNCTION EVALUATION
    double tObjectiveValue = tCriterion.value(tState, tControl);
    const double tGoldValue = 2;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tObjectiveValue, tTolerance);

    // TEST GRADIENT EVALUATION FUNCTION
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 1) = -4;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST HESSIAN TIMES VECTOR FUNCTION
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tCriterion.hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, RadiusCriterion)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 0.5;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Radius<double> tCriterion;

    // TEST OBJECTIVE FUNCTION EVALUATION
    double tObjectiveValue = tCriterion.value(tState, tControl);
    const double tGoldValue = -0.5;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldValue, tObjectiveValue, tTolerance);

    // TEST GRADIENT EVALUATION FUNCTION
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tCriterion.gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    locus::fill(1., tGoldVector);
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST HESSIAN TIMES VECTOR FUNCTION
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    const size_t tVectorIndex = 0;
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tCriterion.hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, AnalyticalGradient)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    const size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    locus::Circle<double> tCriterion;
    locus::AnalyticalGradient<double> tGradient(tCriterion);

    // TEST COMPUTE FUNCTION
    tGradient.compute(tState, tControl, tOutput);

    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls);
    tGold(tVectorIndex, 0) = 0.0;
    tGold(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGold);
}

TEST(LocusTest, AnalyticalHessian)
{
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    const size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);

    std::shared_ptr<locus::Circle<double>> tCriterion = std::make_shared<locus::Circle<double>>();
    locus::AnalyticalHessian<double> tHessian(tCriterion);

    // TEST APPLY VECTOR TO HESSIAN OPERATOR FUNCTION
    tHessian.apply(tState, tControl, tVector, tHessianTimesVector);

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, Preconditioner)
{
    locus::IdentityPreconditioner<double> tPreconditioner;

    const double tValue = 1;
    const size_t tNumVectors = 1;
    const size_t tNumControls = 2;
    const size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    // TEST APPLY PRECONDITIONER AND APPLY INVERSE PRECONDITIONER FUNCTIONS
    tPreconditioner.applyInvPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);
    locus::fill(0., tOutput);
    tPreconditioner.applyPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);

    // TEST CREATE FUNCTION
    std::shared_ptr<locus::Preconditioner<double>> tCopy = tPreconditioner.create();
    tCopy->applyInvPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);
    locus::fill(0., tOutput);
    tCopy->applyPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tVector);
}

TEST(LocusTest, CriterionList)
{
    locus::CriterionList<double> tList;
    size_t tGoldInteger = 0;
    EXPECT_EQ(tGoldInteger, tList.size());

    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    tList.add(tCircle);
    tGoldInteger = 1;
    EXPECT_EQ(tGoldInteger, tList.size());
    tList.add(tRadius);
    tGoldInteger = 2;
    EXPECT_EQ(tGoldInteger, tList.size());

    // ** TEST FIRST CRITERION OBJECTIVE **
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    const double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    size_t tCriterionIndex = 0;
    double tOutput = tList[tCriterionIndex].value(tState, tControl);

    double tGoldScalar = 2;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);

    // TEST FIRST CRITERION GRADIENT
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tList[tCriterionIndex].gradient(tState, tControl, tGradient);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 1) = -4;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST FIRST CRITERION HESSIAN TIMES VECTOR FUNCTION
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tList[tCriterionIndex].hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);

    // ** TEST SECOND CRITERION OBJECTIVE **
    tCriterionIndex = 1;
    locus::fill(0.5, tControl);
    tOutput = tList[tCriterionIndex].value(tState, tControl);
    tGoldScalar = -0.5;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);

    // TEST SECOND CRITERION GRADIENT
    locus::fill(0., tGradient);
    tList[tCriterionIndex].gradient(tState, tControl, tGradient);
    locus::fill(1., tGoldVector);
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // TEST SECOND HESSIAN TIMES VECTOR FUNCTION
    locus::fill(0.5, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(0., tHessianTimesVector);
    tList[tCriterionIndex].hessian(tState, tControl, tVector, tHessianTimesVector);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);

    // **** TEST CREATE FUNCTION ****
    std::shared_ptr<locus::CriterionList<double>> tCopy = tList.create();
    // FIRST OBJECTIVE
    tCriterionIndex = 0;
    locus::fill(1.0, tControl);
    tOutput = tCopy->operator [](tCriterionIndex).value(tState, tControl);
    tGoldScalar = 2;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);
    // SECOND OBJECTIVE
    tCriterionIndex = 1;
    locus::fill(0.5, tControl);
    tOutput = tCopy->operator [](tCriterionIndex).value(tState, tControl);
    tGoldScalar = -0.5;
    EXPECT_NEAR(tGoldScalar, tOutput, tTolerance);
}

TEST(LocusTest, GradientOperatorList)
{
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;

    locus::AnalyticalGradient<double> tCircleGradient(tCircle);
    locus::AnalyticalGradient<double> tRadiusGradient(tRadius);
    locus::GradientOperatorList<double> tList;

    // ********* TEST ADD FUNCTION *********
    tList.add(tCircleGradient);
    size_t tIntegerGold = 1;
    EXPECT_EQ(tIntegerGold, tList.size());

    tList.add(tRadiusGradient);
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tList.size());

    // ********* ALLOCATE DATA STRUCTURES FOR TEST *********
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    // ********* TEST OPERATOR[] - FIRST CRITERION *********
    size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    size_t tGradientOperatorIndex = 0;
    tList[tGradientOperatorIndex].compute(tState, tControl, tOutput);

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(tVectorIndex, 0) = 0.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST OPERATOR[] - SECOND CRITERION *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 1;
    tList[tGradientOperatorIndex].compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - FIRST CRITERION *********
    tValue = 1.0;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 0;
    tList.ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 0.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - SECOND CRITERION *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 1;
    tList.ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - FIRST CRITERION *********
    std::shared_ptr<locus::GradientOperatorList<double>> tListCopy = tList.create();

    tValue = 1.0;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 0;
    tListCopy->ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 0.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - SECOND CRITERION *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    locus::fill(0., tOutput);
    tGradientOperatorIndex = 1;
    tListCopy->ptr(tGradientOperatorIndex)->compute(tState, tControl, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);
}

TEST(LocusTest, LinearOperatorList)
{
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;

    locus::AnalyticalHessian<double> tCircleHessian(tCircle);
    locus::AnalyticalHessian<double> tRadiusHessian(tRadius);
    locus::LinearOperatorList<double> tList;

    // ********* TEST ADD FUNCTION *********
    tList.add(tCircleHessian);
    size_t tIntegerGold = 1;
    EXPECT_EQ(tIntegerGold, tList.size());

    tList.add(tRadiusHessian);
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tList.size());

    // ********* ALLOCATE DATA STRUCTURES FOR TEST *********
    const size_t tNumStates = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tState(tNumVectors, tNumStates);
    double tValue = 1;
    const size_t tNumControls = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);

    // ********* TEST OPERATOR[] - FIRST CRITERION *********
    size_t tVectorIndex = 0;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tVector(tVectorIndex, 1) = -2.;
    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    size_t tLinearOperatorIndex = 0;
    tList[tLinearOperatorIndex].apply(tState, tControl, tVector, tOutput);

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST OPERATOR[] - SECOND CRITERION *********
    locus::fill(0., tOutput);
    tValue = 0.5;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 1;
    tList[tLinearOperatorIndex].apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - FIRST CRITERION *********
    locus::fill(0., tOutput);
    tValue = 1.0;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 0;
    tList.ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST PTR - SECOND CRITERION *********
    locus::fill(0., tOutput);
    tValue = 0.5;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 1;
    tList.ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - FIRST CRITERION *********
    std::shared_ptr<locus::LinearOperatorList<double>> tListCopy = tList.create();

    locus::fill(0., tOutput);
    tValue = 1.0;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 0;
    tListCopy->ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 2.;
    tGoldVector(tVectorIndex, 1) = -8.;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);

    // ********* TEST CREATE - SECOND CRITERION *********
    locus::fill(0., tOutput);
    tValue = 0.5;
    locus::fill(tValue, tVector);
    tVector(tVectorIndex, 1) = -2.;
    locus::fill(tValue, tControl);
    tLinearOperatorIndex = 1;
    tListCopy->ptr(tLinearOperatorIndex)->apply(tState, tControl, tVector, tOutput);
    tGoldVector(tVectorIndex, 0) = 1.0;
    tGoldVector(tVectorIndex, 1) = -4.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldVector);
}

TEST(LocusTest, AugmentedLagrangianStageMng)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tList;
    tList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    locus::AugmentedLagrangianStageMng<double> tStageMng(tDataFactory, tCircle, tList);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER FUNCTIONALITIES *********
    size_t tIntegerGold = 0;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveFunctionEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveGradientEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveHessianEvaluations());
    for(size_t tIndex = 0; tIndex < tList.size(); tIndex++)
    {
        EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintEvaluations(tIndex));
        EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintHessianEvaluations(tIndex));
        EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintGradientEvaluations(tIndex));
    }

    double tScalarGold = 1;
    double tTolerance = 1e-6;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);
    tScalarGold = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarGold, tStageMng.getNormObjectiveFunctionGradient(), tTolerance);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER OBJECTIVE EVALUATION *********
    double tValue = 1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tValue);
    tValue = tStageMng.evaluateObjective(tControl);
    tScalarGold = 2.5;
    EXPECT_NEAR(tScalarGold, tValue, tTolerance);
    tIntegerGold = 1;
    const size_t tConstraintIndex = 0;;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveFunctionEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintEvaluations(tConstraintIndex));

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - UPDATE LAGRANGE MULTIPLIERS *********
    tScalarGold = 1.;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);
    EXPECT_FALSE(tStageMng.updateLagrangeMultipliers());
    locus::StandardMultiVector<double> tLagrangeMultipliers(tNumVectors, tNumDuals);
    tStageMng.getLagrangeMultipliers(tLagrangeMultipliers);
    locus::StandardMultiVector<double> tLagrangeMultipliersGold(tNumVectors, tNumDuals);
    LocusTest::checkMultiVectorData(tLagrangeMultipliers, tLagrangeMultipliersGold);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - UPDATE LAGRANGE MULTIPLIERS & EVALUATE CONSTRAINT *********
    tValue = 0.5;
    locus::fill(tValue, tControl);
    tStageMng.evaluateConstraint(tControl);
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintEvaluations(tConstraintIndex));

    tStageMng.updateCurrentConstraintValues();
    locus::StandardMultiVector<double> tCurrentConstraintValues(tNumVectors, tNumDuals);
    tStageMng.getCurrentConstraintValues(tCurrentConstraintValues);
    tValue = -0.5;
    locus::StandardMultiVector<double> tCurrentConstraintValuesGold(tNumVectors, tNumDuals, tValue);
    LocusTest::checkMultiVectorData(tCurrentConstraintValues, tCurrentConstraintValuesGold);

    tScalarGold = 0.2;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);
    EXPECT_FALSE(tStageMng.updateLagrangeMultipliers());
    tStageMng.getLagrangeMultipliers(tLagrangeMultipliers);
    tScalarGold = 0.04;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentLagrangeMultipliersPenalty(), tTolerance);

    tValue = -2.5;
    locus::fill(tValue, tLagrangeMultipliersGold);
    LocusTest::checkMultiVectorData(tLagrangeMultipliers, tLagrangeMultipliersGold);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - COMPUTE FEASIBILITY MEASURE *********
    tStageMng.computeCurrentFeasibilityMeasure();
    tScalarGold = 0.5;
    EXPECT_NEAR(tScalarGold, tStageMng.getCurrentFeasibilityMeasure(), tTolerance);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - COMPUTE GRADIENT *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng.setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng.setConstraintGradients(tGradientList);

    locus::StandardMultiVector<double> tOutput(tNumVectors, tNumControls);
    tStageMng.computeGradient(tControl, tOutput);
    tScalarGold = 6.0827625302982193;
    EXPECT_NEAR(tScalarGold, tStageMng.getNormObjectiveFunctionGradient(), tTolerance);
    tIntegerGold = 1;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveGradientEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintGradientEvaluations(tConstraintIndex));
    locus::StandardMultiVector<double> tGoldMultiVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldMultiVector(tVectorIndex, 0) = -16;
    tGoldMultiVector(tVectorIndex, 1) = -21;
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - APPLY VECTOR TO HESSIAN *********
    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng.setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng.setConstraintHessians(tHessianList);

    tValue = 1.0;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tValue);
    tStageMng.applyVectorToHessian(tControl, tVector, tOutput);
    EXPECT_EQ(tIntegerGold, tStageMng.getNumObjectiveHessianEvaluations());
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintHessianEvaluations(tConstraintIndex));
    tIntegerGold = 2;
    EXPECT_EQ(tIntegerGold, tStageMng.getNumConstraintGradientEvaluations(tConstraintIndex));
    tGoldMultiVector(tVectorIndex, 0) = 22.0;
    tGoldMultiVector(tVectorIndex, 1) = 24.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);

    // ********* TEST AUGMENTED LAGRANGIAN STAGE MANAGER - APPLY PRECONDITIONER *********
    tStageMng.applyVectorToPreconditioner(tControl, tVector, tOutput);
    tGoldMultiVector(tVectorIndex, 0) = 1.0;
    tGoldMultiVector(tVectorIndex, 1) = 1.0;
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);
    tStageMng.applyVectorToInvPreconditioner(tControl, tVector, tOutput);
    LocusTest::checkMultiVectorData(tOutput, tGoldMultiVector);
}

TEST(LocusTest, SteihaugTointSolver)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE SOLVER *********
    locus::ProjectedSteihaugTointPcg<double> tSolver(tDataFactory);

    // ********* TEST MAX NUM ITERATIONS FUNCTIONS *********
    size_t tIntegerGold = 200;
    EXPECT_EQ(tIntegerGold, tSolver.getMaxNumIterations());
    tIntegerGold = 300;
    tSolver.setMaxNumIterations(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tSolver.getMaxNumIterations());

    // ********* TEST NUM ITERATIONS DONE FUNCTIONS *********
    tIntegerGold = 0;
    EXPECT_EQ(tIntegerGold, tSolver.getNumIterationsDone());
    tIntegerGold = 2;
    tSolver.setNumIterationsDone(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tSolver.getNumIterationsDone());

    // ********* TEST SOLVER TOLERANCE FUNCTIONS *********
    double tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tSolver.getSolverTolerance());
    tScalarGold = 0.2;
    tSolver.setSolverTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getSolverTolerance());

    // ********* TEST SET TRUST REGION RADIUS FUNCTIONS *********
    tScalarGold = 0;
    EXPECT_EQ(tScalarGold, tSolver.getTrustRegionRadius());
    tScalarGold = 2;
    tSolver.setTrustRegionRadius(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getTrustRegionRadius());

    // ********* TEST RESIDUAL NORM FUNCTIONS *********
    tScalarGold = 0;
    EXPECT_EQ(tScalarGold, tSolver.getNormResidual());
    tScalarGold = 1e-2;
    tSolver.setNormResidual(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getNormResidual());

    // ********* TEST RELATIVE TOLERANCE FUNCTIONS *********
    tScalarGold = 1e-1;
    EXPECT_EQ(tScalarGold, tSolver.getRelativeTolerance());
    tScalarGold = 1e-3;
    tSolver.setRelativeTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getRelativeTolerance());

    // ********* TEST RELATIVE TOLERANCE EXPONENTIAL FUNCTIONS *********
    tScalarGold = 0.5;
    EXPECT_EQ(tScalarGold, tSolver.getRelativeToleranceExponential());
    tScalarGold = 0.75;
    tSolver.setRelativeToleranceExponential(tScalarGold);
    EXPECT_EQ(tScalarGold, tSolver.getRelativeToleranceExponential());

    // ********* TEST RELATIVE TOLERANCE EXPONENTIAL FUNCTIONS *********
    locus::solver::stop_t tStopGold = locus::solver::stop_t::MAX_ITERATIONS;
    EXPECT_EQ(tStopGold, tSolver.getStoppingCriterion());
    tStopGold = locus::solver::stop_t::TRUST_REGION_RADIUS;
    tSolver.setStoppingCriterion(tStopGold);
    EXPECT_EQ(tStopGold, tSolver.getStoppingCriterion());

    // ********* TEST INVALID CURVATURE FUNCTION *********
    double tScalarValue = -1;
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::NEGATIVE_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = 0;
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::ZERO_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::INF_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(tSolver.invalidCurvatureDetected(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::NaN_CURVATURE, tSolver.getStoppingCriterion());
    tScalarValue = 1;
    EXPECT_FALSE(tSolver.invalidCurvatureDetected(tScalarValue));

    // ********* TEST TOLERANCE SATISFIED FUNCTION *********
    tScalarValue = 5e-9;
    EXPECT_TRUE(tSolver.toleranceSatisfied(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::TOLERANCE, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(tSolver.toleranceSatisfied(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::NaN_NORM_RESIDUAL, tSolver.getStoppingCriterion());
    tScalarValue = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(tSolver.toleranceSatisfied(tScalarValue));
    EXPECT_EQ(locus::solver::stop_t::INF_NORM_RESIDUAL, tSolver.getStoppingCriterion());
    tScalarValue = 1;
    EXPECT_FALSE(tSolver.toleranceSatisfied(tScalarValue));

    // ********* TEST COMPUTE STEIHAUG TOINT STEP FUNCTION *********
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tNewtonStep(tNumVectors, tNumControls);
    tNewtonStep(0,0) = 0.345854922279793;
    tNewtonStep(0,1) = 1.498704663212435;
    locus::StandardMultiVector<double> tConjugateDirection(tNumVectors, tNumControls);
    tConjugateDirection(0,0) = 1.5;
    tConjugateDirection(0,1) = 6.5;
    locus::StandardMultiVector<double> tPrecTimesNewtonStep(tNumVectors, tNumControls);
    tPrecTimesNewtonStep(0,0) = 0.345854922279793;
    tPrecTimesNewtonStep(0,1) = 1.498704663212435;
    locus::StandardMultiVector<double> tPrecTimesConjugateDirection(tNumVectors, tNumControls);
    tPrecTimesConjugateDirection(0,0) = 1.5;
    tPrecTimesConjugateDirection(0,1) = 6.5;

    tScalarValue = 0.833854004007896;
    tSolver.setTrustRegionRadius(tScalarValue);
    tScalarValue = tSolver.computeSteihaugTointStep(tNewtonStep, tConjugateDirection, tPrecTimesNewtonStep, tPrecTimesConjugateDirection);

    double tTolerance = 1e-6;
    tScalarGold = -0.105569948186529;
    EXPECT_NEAR(tScalarGold, tScalarValue, tTolerance);
}

TEST(LocusTest, ProjectedSteihaugTointPcg)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tList;
    tList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    locus::AugmentedLagrangianStageMng<double> tStageMng(tDataFactory, tCircle, tList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng.setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng.setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng.setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng.setConstraintHessians(tHessianList);

    // ********* Allocate Trust Region Algorithm Data Manager *********
    locus::TrustRegionAlgorithmDataMng<double> tDataMng(tDataFactory);

    // ********* SET INITIAL DATA MANAGER VALUES *********
    double tScalarValue = 0.5;
    tDataMng.setInitialGuess(tScalarValue);
    tScalarValue = tStageMng.evaluateObjective(tDataMng.getCurrentControl());
    tStageMng.updateCurrentConstraintValues();
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    const double tTolerance = 1e-6;
    double tScalarGoldValue = 4.875;
    EXPECT_NEAR(tScalarGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls);
    tStageMng.computeGradient(tDataMng.getCurrentControl(), tVector);
    tDataMng.setCurrentGradient(tVector);
    tDataMng.computeNormProjectedGradient();
    tScalarGoldValue = 6.670832032063167;
    EXPECT_NEAR(tScalarGoldValue, tDataMng.getNormProjectedGradient(), tTolerance);
    tVector(0, 0) = -1.5;
    tVector(0, 1) = -6.5;
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tVector);

    // ********* ALLOCATE SOLVER DATA STRUCTURE *********
    locus::ProjectedSteihaugTointPcg<double> tSolver(tDataFactory);

    // ********* CONVERGENCE: SOLVER TOLERANCE MET *********
    tScalarValue = tDataMng.getNormProjectedGradient();
    tSolver.setTrustRegionRadius(tScalarValue);
    tSolver.solve(tStageMng, tDataMng);
    size_t tIntegerGoldValue = 2;
    EXPECT_EQ(tIntegerGoldValue, tSolver.getNumIterationsDone());
    EXPECT_EQ(locus::solver::stop_t::TOLERANCE, tSolver.getStoppingCriterion());
    EXPECT_TRUE(tSolver.getNormResidual() < tSolver.getSolverTolerance());
    tVector(0, 0) = -0.071428571428571;
    tVector(0, 1) = 1.642857142857143;
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tVector);

    // ********* CONVERGENCE: MAX NUMBER OF ITERATIONS *********
    tSolver.setMaxNumIterations(2);
    tSolver.setSolverTolerance(1e-15);
    tSolver.solve(tStageMng, tDataMng);
    EXPECT_EQ(tIntegerGoldValue, tSolver.getNumIterationsDone());
    EXPECT_EQ(locus::solver::stop_t::MAX_ITERATIONS, tSolver.getStoppingCriterion());
    EXPECT_FALSE(tSolver.getNormResidual() < tSolver.getSolverTolerance());
    tVector(0, 0) = -0.071428571428571;
    tVector(0, 1) = 1.642857142857143;
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tVector);

    // ********* CONVERGENCE: TRUST REGION RADIUS VIOLATED *********
    tSolver.setTrustRegionRadius(0.833854004007896);
    tSolver.solve(tStageMng, tDataMng);
    tIntegerGoldValue = 1;
    EXPECT_EQ(tIntegerGoldValue, tSolver.getNumIterationsDone());
    EXPECT_EQ(locus::solver::stop_t::TRUST_REGION_RADIUS, tSolver.getStoppingCriterion());
    EXPECT_FALSE(tSolver.getNormResidual() < tSolver.getSolverTolerance());
    tVector(0, 0) = 0.1875;
    tVector(0, 1) = 0.8125;
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tVector);
}

TEST(LocusTest, TrustRegionStepMng)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    locus::KelleySachsStepMng<double> tStepMng(tDataFactory);

    // ********* TEST ACTUAL REDUCTION FUNCTIONS *********
    double tTolerance = 1e-6;
    double tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualReduction(), tTolerance);
    tScalarGoldValue = 0.45;
    tStepMng.setActualReduction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualReduction(), tTolerance);

    // ********* TEST TRUST REGION RADIUS FUNCTIONS *********
    tScalarGoldValue = 1e4;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionRadius(), tTolerance);
    tScalarGoldValue = 1e2;
    tStepMng.setTrustRegionRadius(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionRadius(), tTolerance);

    // ********* TEST TRUST REGION CONTRACTION FUNCTIONS *********
    tScalarGoldValue = 0.5;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionContraction(), tTolerance);
    tScalarGoldValue = 0.25;
    tStepMng.setTrustRegionContraction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionContraction(), tTolerance);

    // ********* TEST TRUST REGION EXPANSION FUNCTIONS *********
    tScalarGoldValue = 2;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionExpansion(), tTolerance);
    tScalarGoldValue = 8;
    tStepMng.setTrustRegionExpansion(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionExpansion(), tTolerance);

    // ********* TEST MIN TRUST REGION RADIUS FUNCTIONS *********
    tScalarGoldValue = 1e-4;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinTrustRegionRadius(), tTolerance);
    tScalarGoldValue = 1e-2;
    tStepMng.setMinTrustRegionRadius(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinTrustRegionRadius(), tTolerance);

    // ********* TEST MAX TRUST REGION RADIUS FUNCTIONS *********
    tScalarGoldValue = 1e4;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMaxTrustRegionRadius(), tTolerance);
    tScalarGoldValue = 1e1;
    tStepMng.setMaxTrustRegionRadius(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMaxTrustRegionRadius(), tTolerance);

    // ********* TEST GRADIENT INEXACTNESS FUNCTIONS *********
    tScalarGoldValue = 1;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessToleranceConstant(), tTolerance);
    tScalarGoldValue = 2;
    tStepMng.setGradientInexactnessToleranceConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessToleranceConstant(), tTolerance);
    // TEST INEXACTNESS TOLERANCE: SELECT CURRENT TRUST REGION RADIUS
    tScalarGoldValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessTolerance(), tTolerance);
    tScalarGoldValue = 1e3;
    tStepMng.updateGradientInexactnessTolerance(tScalarGoldValue);
    tScalarGoldValue = 200;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessTolerance(), tTolerance);
    // TEST INEXACTNESS TOLERANCE: SELECT USER INPUT
    tScalarGoldValue = 1e1;
    tStepMng.updateGradientInexactnessTolerance(tScalarGoldValue);
    tScalarGoldValue = 20;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getGradientInexactnessTolerance(), tTolerance);

    // ********* TEST OBJECTIVE INEXACTNESS FUNCTIONS *********
    tScalarGoldValue = 1;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessToleranceConstant(), tTolerance);
    tScalarGoldValue = 3;
    tStepMng.setObjectiveInexactnessToleranceConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessToleranceConstant(), tTolerance);
    // TEST INEXACTNESS TOLERANCE
    tScalarGoldValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessTolerance(), tTolerance);
    tScalarGoldValue = 100;
    tStepMng.updateObjectiveInexactnessTolerance(tScalarGoldValue);
    tScalarGoldValue = 30;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getObjectiveInexactnessTolerance(), tTolerance);

    // ********* TEST ACTUAL OVER PREDICTED REDUCTION BOUND FUNCTIONS *********
    tScalarGoldValue = 0.25;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionMidBound(), tTolerance);
    tScalarGoldValue = 0.4;
    tStepMng.setActualOverPredictedReductionMidBound(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionMidBound(), tTolerance);

    tScalarGoldValue = 0.1;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionLowerBound(), tTolerance);
    tScalarGoldValue = 0.05;
    tStepMng.setActualOverPredictedReductionLowerBound(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionLowerBound(), tTolerance);

    tScalarGoldValue = 0.75;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionUpperBound(), tTolerance);
    tScalarGoldValue = 0.8;
    tStepMng.setActualOverPredictedReductionUpperBound(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReductionUpperBound(), tTolerance);

    // ********* TEST PREDICTED REDUCTION FUNCTIONS *********
    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getPredictedReduction(), tTolerance);
    tScalarGoldValue = 0.12;
    tStepMng.setPredictedReduction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getPredictedReduction(), tTolerance);

    // ********* TEST MIN COSINE ANGLE TOLERANCE FUNCTIONS *********
    tScalarGoldValue = 1e-2;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinCosineAngleTolerance(), tTolerance);
    tScalarGoldValue = 0.1;
    tStepMng.setMinCosineAngleTolerance(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMinCosineAngleTolerance(), tTolerance);

    // ********* TEST ACTUAL OVER PREDICTED REDUCTION FUNCTIONS *********
    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReduction(), tTolerance);
    tScalarGoldValue = 0.23;
    tStepMng.setActualOverPredictedReduction(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReduction(), tTolerance);

    // ********* TEST NUMBER OF TRUST REGION SUBPROBLEM ITERATIONS FUNCTIONS *********
    size_t tIntegerGoldValue = 0;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getNumTrustRegionSubProblemItrDone());
    tStepMng.updateNumTrustRegionSubProblemItrDone();
    tIntegerGoldValue = 1;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getNumTrustRegionSubProblemItrDone());

    tIntegerGoldValue = 30;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getMaxNumTrustRegionSubProblemItr());
    tIntegerGoldValue = 50;
    tStepMng.setMaxNumTrustRegionSubProblemItr(tIntegerGoldValue);
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getMaxNumTrustRegionSubProblemItr());

    EXPECT_TRUE(tStepMng.isInitialTrustRegionRadiusSetToNormProjectedGradient());
    tStepMng.setInitialTrustRegionRadiusSetToNormProjectedGradient(false);
    EXPECT_FALSE(tStepMng.isInitialTrustRegionRadiusSetToNormProjectedGradient());
}

TEST(LocusTest, KelleySachsStepMng)
{
    // ********* ALLOCATE DATA FACTORY *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tList;
    tList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    locus::AugmentedLagrangianStageMng<double> tStageMng(tDataFactory, tCircle, tList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng.setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng.setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng.setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng.setConstraintHessians(tHessianList);

    // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
    locus::TrustRegionAlgorithmDataMng<double> tDataMng(tDataFactory);

    // ********* SET INITIAL DATA MANAGER VALUES *********
    double tScalarValue = 0.5;
    tDataMng.setInitialGuess(tScalarValue);
    tScalarValue = tStageMng.evaluateObjective(tDataMng.getCurrentControl());
    tStageMng.updateCurrentConstraintValues();
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    const double tTolerance = 1e-6;
    double tScalarGoldValue = 4.875;
    EXPECT_NEAR(tScalarGoldValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls);
    tStageMng.computeGradient(tDataMng.getCurrentControl(), tVector);
    tDataMng.setCurrentGradient(tVector);
    tDataMng.computeNormProjectedGradient();
    double tNormProjectedGradientGold = 6.670832032063167;
    EXPECT_NEAR(tNormProjectedGradientGold, tDataMng.getNormProjectedGradient(), tTolerance);
    tDataMng.computeStationarityMeasure();
    double tStationarityMeasureGold = 6.670832032063167;
    EXPECT_NEAR(tStationarityMeasureGold, tDataMng.getStationarityMeasure(), tTolerance);
    tVector(0, 0) = -1.5;
    tVector(0, 1) = -6.5;
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tVector);

    // ********* ALLOCATE SOLVER *********
    locus::ProjectedSteihaugTointPcg<double> tSolver(tDataFactory);

    // ********* ALLOCATE STEP MANAGER *********
    locus::KelleySachsStepMng<double> tStepMng(tDataFactory);
    tStepMng.setTrustRegionRadius(tNormProjectedGradientGold);

    // ********* TEST CONSTANT FUNCTIONS *********
    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEtaConstant(), tTolerance);
    tScalarGoldValue = 0.12;
    tStepMng.setEtaConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEtaConstant(), tTolerance);

    tScalarGoldValue = 0;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEpsilonConstant(), tTolerance);
    tScalarGoldValue = 0.11;
    tStepMng.setEpsilonConstant(tScalarGoldValue);
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getEpsilonConstant(), tTolerance);

    // ********* TEST SUBPROBLEM SOLVE *********
    tScalarValue = 0.01;
    tStepMng.setEtaConstant(tScalarValue);
    EXPECT_TRUE(tStepMng.solveSubProblem(tDataMng, tStageMng, tSolver));

    // VERIFY CURRENT SUBPROBLEM SOLVE RESULTS
    size_t tIntegerGoldValue = 4;
    EXPECT_EQ(tIntegerGoldValue, tStepMng.getNumTrustRegionSubProblemItrDone());
    tScalarGoldValue = 0.768899024566474;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualOverPredictedReduction(), tTolerance);
    tScalarGoldValue = 1.757354736328125;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getMidPointObjectiveFunctionValue(), tTolerance);
    tScalarGoldValue = 3.335416016031584;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getTrustRegionRadius(), tTolerance);
    tScalarGoldValue = -3.117645263671875;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getActualReduction(), tTolerance);
    tScalarGoldValue = -4.0546875;
    EXPECT_NEAR(tScalarGoldValue, tStepMng.getPredictedReduction(), tTolerance);
    tScalarGoldValue = 0.066708320320632;
    EXPECT_NEAR(tScalarGoldValue, tSolver.getSolverTolerance(), tTolerance);
    const locus::MultiVector<double> & tMidControl = tStepMng.getMidPointControls();
    locus::StandardMultiVector<double> tVectorGold(tNumVectors, tNumControls);
    tVectorGold(0, 0) = 0.6875;
    tVectorGold(0, 1) = 1.3125;
    LocusTest::checkMultiVectorData(tMidControl, tVectorGold);
    const locus::MultiVector<double> & tTrialStep = tDataMng.getTrialStep();
    tVectorGold(0, 0) = 0.1875;
    tVectorGold(0, 1) = 0.8125;
    LocusTest::checkMultiVectorData(tTrialStep, tVectorGold);
}

TEST(LocusTest, KelleySachsAlgorithm)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory->allocateDual(tNumDuals);
    tDataFactory->allocateControl(tNumControls);

    // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<double>> tDataMng =
            std::make_shared<locus::TrustRegionAlgorithmDataMng<double>>(*tDataFactory);
    double tScalarValue = 0.5;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -100;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 100;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tConstraintList;
    tConstraintList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    std::shared_ptr<locus::AugmentedLagrangianStageMng<double>> tStageMng =
            std::make_shared<locus::AugmentedLagrangianStageMng<double>>(*tDataFactory, tCircle, tConstraintList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng->setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng->setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng->setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng->setConstraintHessians(tHessianList);

    // ********* ALLOCATE KELLEY-SACHS ALGORITHM *********
    locus::KelleySachsAugmentedLagrangian<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);

    // TEST MAXIMUM NUMBER OF UPDATES
    size_t tIntegerGold = 10;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumUpdates());
    tIntegerGold = 5;
    tAlgorithm.setMaxNumUpdates(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumUpdates());

    // TEST NUMBER ITERATIONS DONE
    tIntegerGold = 0;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getNumIterationsDone());
    tIntegerGold = 3;
    tAlgorithm.setNumIterationsDone(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tAlgorithm.getNumIterationsDone());

    // TEST NUMBER ITERATIONS DONE
    tIntegerGold = 100;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumIterations());
    tIntegerGold = 30;
    tAlgorithm.setMaxNumIterations(tIntegerGold);
    EXPECT_EQ(tIntegerGold, tAlgorithm.getMaxNumIterations());

    // TEST STOPPING CRITERIA
    locus::algorithm::stop_t tGold = locus::algorithm::stop_t::NOT_CONVERGED;
    EXPECT_EQ(tGold, tAlgorithm.getStoppingCriterion());
    tGold = locus::algorithm::stop_t::NaN_NORM_TRIAL_STEP;
    tAlgorithm.setStoppingCriterion(tGold);
    EXPECT_EQ(tGold, tAlgorithm.getStoppingCriterion());

    // TEST GRADIENT TOLERANCE
    double tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getGradientTolerance());
    tScalarGold = 1e-4;
    tAlgorithm.setGradientTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getGradientTolerance());

    // TEST TRIAL STEP TOLERANCE
    tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getTrialStepTolerance());
    tScalarGold = 1e-3;
    tAlgorithm.setTrialStepTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getTrialStepTolerance());

    // TEST OBJECTIVE TOLERANCE
    tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getObjectiveTolerance());
    tScalarGold = 1e-5;
    tAlgorithm.setObjectiveTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getObjectiveTolerance());

    // TEST CONTROL STAGNATION TOLERANCE
    tScalarGold = 1e-8;
    EXPECT_EQ(tScalarGold, tAlgorithm.getStagnationTolerance());
    tScalarGold = 1e-2;
    tAlgorithm.setStagnationTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getStagnationTolerance());

    // TEST ACTUAL REDUCTION TOLERANCE
    tScalarGold = 1e-10;
    EXPECT_EQ(tScalarGold, tAlgorithm.getActualReductionTolerance());
    tScalarGold = 1e-9;
    tAlgorithm.setActualReductionTolerance(tScalarGold);
    EXPECT_EQ(tScalarGold, tAlgorithm.getActualReductionTolerance());

    // TEST UPDATE CONTROL
    const size_t tNumVectors = 1;
    locus::KelleySachsStepMng<double> tStepMng(*tDataFactory);
    locus::StandardMultiVector<double> tMidControls(tNumVectors, tNumControls);
    tMidControls(0,0) = 0.6875;
    tMidControls(0,1) = 1.3125;
    tStepMng.setMidPointControls(tMidControls);
    locus::StandardMultiVector<double> tMidGradient(tNumVectors, tNumControls);
    tMidGradient(0,0) = 1.0185546875;
    tMidGradient(0,1) = 0.3876953125;
    tScalarValue = 1.757354736328125;
    tStepMng.setMidPointObjectiveFunctionValue(tScalarValue);
    tScalarValue = -3.117645263671875;
    tStepMng.setActualReduction(tScalarValue);
    EXPECT_TRUE(tAlgorithm.updateControl(tMidGradient, tStepMng, *tDataMng, *tStageMng));

    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(0,0) = -0.3310546875;
    tGoldVector(0,1) = 0.9248046875;
    const locus::MultiVector<double> & tCurrentControl = tDataMng->getCurrentControl();
    LocusTest::checkMultiVectorData(tCurrentControl, tGoldVector);
    const double tTolerance = 1e-6;
    tScalarGold = 2.327059142438884;
    EXPECT_NEAR(tScalarGold, tStepMng.getActualReduction(), tTolerance);
    tScalarGold = 4.084413878767009;
    EXPECT_NEAR(tScalarGold, tDataMng->getCurrentObjectiveFunctionValue(), tTolerance);
}

TEST(LocusTest, KelleySachsAugmentedLagrangian)
{
    // ********* ALLOCATE DATA FACTORY *********
    std::shared_ptr<locus::DataFactory<double>> tDataFactory =
            std::make_shared<locus::DataFactory<double>>();
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory->allocateDual(tNumDuals);
    tDataFactory->allocateControl(tNumControls);

    // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<double>> tDataMng =
            std::make_shared<locus::TrustRegionAlgorithmDataMng<double>>(*tDataFactory);
    double tScalarValue = 0.5;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -100;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 100;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
    locus::Circle<double> tCircle;
    locus::Radius<double> tRadius;
    locus::CriterionList<double> tConstraintList;
    tConstraintList.add(tRadius);

    // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
    std::shared_ptr<locus::AugmentedLagrangianStageMng<double>> tStageMng =
            std::make_shared<locus::AugmentedLagrangianStageMng<double>>(*tDataFactory, tCircle, tConstraintList);

    // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
    locus::AnalyticalGradient<double> tObjectiveGradient(tCircle);
    tStageMng->setObjectiveGradient(tObjectiveGradient);
    locus::AnalyticalGradient<double> tConstraintGradient(tRadius);
    locus::GradientOperatorList<double> tGradientList;
    tGradientList.add(tConstraintGradient);
    tStageMng->setConstraintGradients(tGradientList);

    locus::AnalyticalHessian<double> tObjectiveHessian(tCircle);
    tStageMng->setObjectiveHessian(tObjectiveHessian);
    locus::AnalyticalHessian<double> tConstraintHessian(tRadius);
    locus::LinearOperatorList<double> tHessianList;
    tHessianList.add(tConstraintHessian);
    tStageMng->setConstraintHessians(tHessianList);

    // ********* ALLOCATE KELLEY-SACHS ALGORITHM *********
    locus::KelleySachsAugmentedLagrangian<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    // TEST NUMBER OF ITERATIONS AND STOPPING CRITERION
    size_t tIntegerGold = 25;
    EXPECT_EQ(tIntegerGold, tAlgorithm.getNumIterationsDone());
    locus::algorithm::stop_t tGold = locus::algorithm::stop_t::CONTROL_STAGNATION;
    EXPECT_EQ(tGold, tAlgorithm.getStoppingCriterion());

    // TEST OBJECTIVE FUNCTION VALUE
    const double tTolerance = 1e-6;
    double tScalarGold = 2.678009477208421;
    EXPECT_NEAR(tScalarGold, tDataMng->getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarGold = 2.678009477208421;

    // TEST CURRENT CONSTRAINT VALUE
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tConstraintValues(tNumVectors, tNumDuals);
    tStageMng->getCurrentConstraintValues(tConstraintValues);
    locus::StandardMultiVector<double> tGoldConstraintValues(tNumVectors, tNumDuals);
    tGoldConstraintValues(0,0) = 1.876192258460918e-4;
    LocusTest::checkMultiVectorData(tConstraintValues, tGoldConstraintValues);

    // TEST LAGRANGE MULTIPLIERS
    locus::StandardMultiVector<double> tLagrangeMulipliers(tNumVectors, tNumDuals);
    tStageMng->getLagrangeMultipliers(tLagrangeMulipliers);
    locus::StandardMultiVector<double> tGoldtLagrangeMulipliers(tNumVectors, tNumDuals);
    tGoldtLagrangeMulipliers(0,0) = 2.209155776190176;
    LocusTest::checkMultiVectorData(tLagrangeMulipliers, tGoldtLagrangeMulipliers);

    // TEST CONTROL SOLUTION
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    tGoldVector(0,0) = 0.311608429003505;
    tGoldVector(0,1) = 0.950309321326385;
    const locus::MultiVector<double> & tCurrentControl = tDataMng->getCurrentControl();
    LocusTest::checkMultiVectorData(tCurrentControl, tGoldVector);

    // TEST CURRENT AUGMENTED LAGRANGIAN GRADIENT
    tGoldVector(0,0) = 0.073079644963231;
    tGoldVector(0,1) = 0.222870312033612;
    const locus::MultiVector<double> & tCurrentGradient = tDataMng->getCurrentGradient();
    LocusTest::checkMultiVectorData(tCurrentGradient, tGoldVector);
}

/* ******************************************************************* */
/* ************* NONLINEAR CONJUGATE GRADIENT UNIT TESTS ************* */
/* ******************************************************************* */

TEST(LocusTest, NonlinearConjugateGradientDataMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);
    size_t tOrdinalValue = 1;
    EXPECT_EQ(tDataMng.getNumControlVectors(), tOrdinalValue);

    // ********* TEST OBJECTIVE FUNCTION VALUE *********
    const double tTolerance = 1e-6;
    double tScalarValue = std::numeric_limits<double>::max();
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);
    tScalarValue = 45;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getCurrentObjectiveFunctionValue(), tTolerance);
    tScalarValue = 123;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    EXPECT_NEAR(tScalarValue, tDataMng.getPreviousObjectiveFunctionValue(), tTolerance);

    // ********* TEST INITIAL GUESS FUNCTIONS *********
    EXPECT_FALSE(tDataMng.isInitialGuessSet());
    tScalarValue = 2;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tMultiVector(tNumVectors, tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tScalarValue);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.5;
    const size_t tVectorIndex = 0;
    tDataMng.setInitialGuess(tVectorIndex, tScalarValue);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setInitialGuess(tMultiVector);
    EXPECT_TRUE(tDataMng.isInitialGuessSet());
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 0.5;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    tDataMng.setInitialGuess(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tVector);

    // ********* TEST TRIAL STEP FUNCTIONS *********
    tScalarValue = 0.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setTrialStep(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getTrialStep(), tMultiVector);

    tScalarValue = 0.25;
    tVector.fill(tScalarValue);
    tDataMng.setTrialStep(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);

    // ********* TEST CURRENT CONTROL FUNCTIONS *********
    tScalarValue = 1.2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentControl(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentControl(), tMultiVector);

    tScalarValue = 1.25;
    tVector.fill(tScalarValue);
    tDataMng.setCurrentControl(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentControl(tVectorIndex), tVector);

    // ********* TEST PREVIOUS CONTROL FUNCTIONS *********
    tScalarValue = 1.21;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setPreviousControl(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tMultiVector);

    tScalarValue = 1.11;
    tVector.fill(tScalarValue);
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getPreviousControl(tVectorIndex), tVector);

    // ********* TEST CURRENT GRADIENT FUNCTIONS *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentGradient(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentGradient(), tMultiVector);

    tScalarValue = 3;
    tVector.fill(tScalarValue);
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getCurrentGradient(tVectorIndex), tVector);

    // ********* TEST PREVIOUS GRADIENT FUNCTIONS *********
    tScalarValue = 2.1;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setPreviousGradient(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tMultiVector);

    tScalarValue = 3.1;
    tVector.fill(tScalarValue);
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    LocusTest::checkVectorData(tDataMng.getPreviousGradient(tVectorIndex), tVector);

    // ********* TEST DEFAULT UPPER AND LOWER BOUNDS *********
    tScalarValue = -std::numeric_limits<double>::max();
    tVector.fill(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);
    LocusTest::checkVectorData(tDataMng.getControlLowerBounds(tVectorIndex), tVector);

    tScalarValue = std::numeric_limits<double>::max();
    tVector.fill(tScalarValue);
    locus::fill(tScalarValue,tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);
    LocusTest::checkVectorData(tDataMng.getControlUpperBounds(tVectorIndex), tVector);

    // ********* TEST LOWER BOUND FUNCTIONS *********
    tScalarValue = -10;
    tDataMng.setControlLowerBounds(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -9;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setControlLowerBounds(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -8;
    tVector.fill(tScalarValue);
    tDataMng.setControlLowerBounds(tVectorIndex, tVector);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    tScalarValue = -7;
    tDataMng.setControlLowerBounds(tVectorIndex, tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlLowerBounds(), tMultiVector);

    // ********* TEST UPPER BOUND FUNCTIONS *********
    tScalarValue = 10;
    tDataMng.setControlUpperBounds(tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 9;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setControlUpperBounds(tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 8;
    tVector.fill(tScalarValue);
    tDataMng.setControlUpperBounds(tVectorIndex, tVector);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    tScalarValue = 7;
    tDataMng.setControlUpperBounds(tVectorIndex, tScalarValue);
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getControlUpperBounds(), tMultiVector);

    // ********* TEST COMPUTE CONTROL STAGNATION MEASURE *********
    tScalarValue = 3;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentControl(tMultiVector);
    tVector[0] = 2;
    tVector[1] = 2.5;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tDataMng.computeStagnationMeasure();
    tScalarValue = 1.;
    EXPECT_NEAR(tDataMng.getStagnationMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE OBJECTIVE STAGNATION MEASURE *********
    tScalarValue = 1.25;
    tDataMng.setCurrentObjectiveFunctionValue(tScalarValue);
    tScalarValue = 0.75;
    tDataMng.setPreviousObjectiveFunctionValue(tScalarValue);
    tDataMng.computeObjectiveStagnationMeasure();
    tScalarValue = 0.5;
    EXPECT_NEAR(tDataMng.getObjectiveStagnationMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE NORM GRADIENT *********
    tScalarValue = 1;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setCurrentGradient(tMultiVector);
    tDataMng.computeNormGradient();
    tScalarValue = std::sqrt(2.);
    EXPECT_NEAR(tDataMng.getNormGradient(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE STATIONARITY MEASURE *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tMultiVector);
    tDataMng.setTrialStep(tMultiVector);
    tDataMng.computeStationarityMeasure();
    tScalarValue = std::sqrt(8.);
    EXPECT_NEAR(tDataMng.getStationarityMeasure(), tScalarValue, tTolerance);

    // ********* TEST COMPUTE STATIONARITY MEASURE *********
    tDataMng.storePreviousState();
    tScalarValue = 1.25;
    EXPECT_NEAR(tDataMng.getPreviousObjectiveFunctionValue(), tScalarValue, tTolerance);
    tScalarValue = 1;
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousGradient(), tMultiVector);
    tScalarValue = 3;
    locus::fill(tScalarValue, tMultiVector);
    LocusTest::checkMultiVectorData(tDataMng.getPreviousControl(), tMultiVector);
}

TEST(LocusTest, NonlinearConjugateGradientStandardStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Test Evaluate Objective Function *********
    double tScalarValue = 2;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tScalarValue = 401;
    double tTolerance = 1e-6;
    size_t tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveFunctionEvaluations(), tOrdinalValue);
    EXPECT_NEAR(tStageMng.evaluateObjective(tControl), tScalarValue, tTolerance);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveFunctionEvaluations(), tOrdinalValue);

    // ********* Test Compute Gradient *********
    tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveGradientEvaluations(), tOrdinalValue);
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tStageMng.computeGradient(tControl, tGradient);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveGradientEvaluations(), tOrdinalValue);
    locus::StandardMultiVector<double> tGoldVector(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGoldVector(tVectorIndex, 0) = 1602;
    tGoldVector(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGoldVector);

    // ********* Test Apply Vector to Hessian *********
    tOrdinalValue = 0;
    EXPECT_EQ(tStageMng.getNumObjectiveHessianEvaluations(), tOrdinalValue);
    tScalarValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tScalarValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tStageMng.applyVectorToHessian(tControl, tVector, tHessianTimesVector);
    tOrdinalValue = 1;
    EXPECT_EQ(tStageMng.getNumObjectiveHessianEvaluations(), tOrdinalValue);
    tGoldVector(tVectorIndex, 0) = 3202;
    tGoldVector(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGoldVector);
}

TEST(LocusTest, PolakRibiere)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Polak-Ribiere Direction *********
    locus::PolakRibiere<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -103.4;
    tVector[1] = -250.8;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, FletcherReeves)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Fletcher-Reeves Direction *********
    locus::FletcherReeves<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -110;
    tVector[1] = -264;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, HestenesStiefel)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -3;
    tVector[1] = -2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hestenes-Stiefel Direction *********
    locus::HestenesStiefel<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 1.3333333333333333;
    tVector[1] = -1.333333333333333;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, ConjugateDescent)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Conjugate Descent Direction *********
    locus::ConjugateDescent<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -110;
    tVector[1] = -264;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiLiao)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -1;
    tVector[1] = 2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tVector[0] = 2;
    tVector[1] = 3;
    tDataMng.setCurrentControl(tVectorIndex, tVector);

    // ********* Allocate Dai-Liao Direction *********
    locus::DaiLiao<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 0.05;
    tVector[1] = -3.9;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, PerryShanno)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousControl(tVectorIndex, tVector);
    tVector[0] = 2;
    tVector[1] = 3;
    tDataMng.setCurrentControl(tVectorIndex, tVector);

    // ********* Allocate Perry-Shanno Direction *********
    locus::PerryShanno<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -0.419267707083;
    tVector[1] = -0.722989195678;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, LiuStorey)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Liu-Storey Direction *********
    locus::LiuStorey<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -103.4;
    tVector[1] = -250.8;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, HagerZhang)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::HagerZhang<double> tDirection(tDataFactory);
    // TEST 1: SCALE FACTOR SELECTED
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -14.367346938775;
    tVector[1] = -72.734693877551;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
    // TEST 2: SCALE FACTOR NOT SELECTED, LOWER BOUND USED INSTEAD
    tVector.fill(1e-1);
    tDataMng.setTrialStep(tVectorIndex, tVector);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 11;
    tVector[1] = -22;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiYuan)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -1;
    tVector[1] = 2;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::DaiYuan<double> tDirection(tDataFactory);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = -1.5;
    tVector[1] = -7;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, DaiYuanHybrid)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    locus::NonlinearConjugateGradientStandardStageMng<double> tStageMng(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    locus::NonlinearConjugateGradientDataMng<double> tDataMng(tDataFactory);

    const size_t tVectorIndex = 0;
    locus::StandardVector<double> tVector(tNumControls);
    tVector[0] = -11;
    tVector[1] = 22;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 1;
    tVector[1] = 2;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tVector[0] = -1;
    tVector[1] = -2;
    tDataMng.setTrialStep(tVectorIndex, tVector);

    // ********* Allocate Hager-Zhang Direction *********
    locus::DaiYuanHybrid<double> tDirection(tDataFactory);
    // TEST 1: SCALED STEP
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 0.19642857142857;
    tVector[1] = -43.607142857142;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
    // TEST 2: UNSCALED STEP
    tVector[0] = -12;
    tVector[1] = -23;
    tDataMng.setCurrentGradient(tVectorIndex, tVector);
    tVector[0] = 11;
    tVector[1] = 22;
    tDataMng.setPreviousGradient(tVectorIndex, tVector);
    tDirection.computeScaledDescentDirection(tDataMng, tStageMng);
    tVector[0] = 12.067522825323;
    tVector[1] = 8.009932778168;
    LocusTest::checkVectorData(tDataMng.getTrialStep(tVectorIndex), tVector);
}

TEST(LocusTest, NonlinearConjugateGradientStateMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(tDataFactory);

    // ********* Allocate Nonlinear Conjugate Gradient State Manager *********
    locus::NonlinearConjugateGradientStateMng<double> tStateMng(tDataMng, tStageMng);

    // ********* Test Set Trial Step Function *********
    double tScalarValue = 0.1;
    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tMultiVector(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setTrialStep(tMultiVector);
    LocusTest::checkMultiVectorData(tStateMng.getTrialStep(), tMultiVector);

    // ********* Test Set Current Control Function *********
    tScalarValue = 2;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setCurrentControl(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getCurrentControl(), tControl);

    // ********* Test Set Current Control Function *********
    tScalarValue = 3;
    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls, tScalarValue);
    tStateMng.setCurrentGradient(tGradient);
    LocusTest::checkMultiVectorData(tStateMng.getCurrentGradient(), tGradient);

    // ********* Test Set Control Lower Bounds Function *********
    tScalarValue = std::numeric_limits<double>::min();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlLowerBounds(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getControlLowerBounds(), tControl);

    // ********* Test Set Control Upper Bounds Function *********
    tScalarValue = std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlUpperBounds(tControl);
    LocusTest::checkMultiVectorData(tStateMng.getControlUpperBounds(), tControl);

    // ********* Test Evaluate Objective Function *********
    tScalarValue = 2;
    locus::fill(tScalarValue, tControl);
    tScalarValue = 401;
    const double tTolerance = 1e-6;
    EXPECT_NEAR(tStateMng.evaluateObjective(tControl), tScalarValue, tTolerance);

    // ********* Test Set Current Objective Function *********
    tStateMng.setCurrentObjectiveValue(tScalarValue);
    EXPECT_NEAR(tStateMng.getCurrentObjectiveValue(), tScalarValue, tTolerance);

    // ********* Test Compute Gradient Function *********
    tScalarValue = 0;
    locus::fill(tScalarValue, tGradient);
    tStateMng.computeGradient(tControl, tGradient);
    locus::StandardMultiVector<double> tGold(tNumVectors, tNumControls);
    const size_t tVectorIndex = 0;
    tGold(tVectorIndex, 0) = 1602;
    tGold(tVectorIndex, 1) = -400;
    LocusTest::checkMultiVectorData(tGradient, tGold);

    // ********* Test Apply Vector to Hessian Function *********
    tScalarValue = 1;
    locus::StandardMultiVector<double> tVector(tNumVectors, tNumControls, tScalarValue);
    locus::StandardMultiVector<double> tHessianTimesVector(tNumVectors, tNumControls);
    tStateMng.applyVectorToHessian(tControl, tVector, tHessianTimesVector);
    tGold(tVectorIndex, 0) = 3202;
    tGold(tVectorIndex, 1) = -600;
    LocusTest::checkMultiVectorData(tHessianTimesVector, tGold);
}

TEST(LocusTest, QuadraticLineSearch)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumControls = 2;
    tDataFactory.allocateControl(tNumControls);

    // ********* Allocate Reduction Operations Interface *********
    locus::StandardVectorReductionOperations<double> tReductionOperations;
    tDataFactory.allocateControlReductionOperations(tReductionOperations);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(tDataFactory);

    // ********* Allocate Nonlinear Conjugate Gradient State Manager *********
    locus::NonlinearConjugateGradientStateMng<double> tStateMng(tDataMng, tStageMng);

    const size_t tNumVectors = 1;
    locus::StandardMultiVector<double> tControl(tNumVectors, tNumControls);
    double tScalarValue = std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlUpperBounds(tControl);
    tScalarValue = -std::numeric_limits<double>::max();
    locus::fill(tScalarValue, tControl);
    tStateMng.setControlLowerBounds(tControl);

    const size_t tVectorIndex = 0;
    tControl(tVectorIndex, 0) = 1.997506234413967;
    tControl(tVectorIndex, 1) = 3.990024937655861;
    tStateMng.setCurrentControl(tControl);
    tScalarValue = tStateMng.evaluateObjective(tControl);
    const double tTolerance = 1e-6;
    double tGoldScalarValue = 0.99501869156216238;
    EXPECT_NEAR(tScalarValue, tGoldScalarValue, tTolerance);
    tStateMng.setCurrentObjectiveValue(tScalarValue);

    locus::StandardMultiVector<double> tGradient(tNumVectors, tNumControls);
    tStateMng.computeGradient(tControl, tGradient);
    tStateMng.setCurrentGradient(tGradient);

    locus::StandardMultiVector<double> tTrialStep(tNumVectors, tNumControls);
    tTrialStep(tVectorIndex, 0) = -1.997506234413967;
    tTrialStep(tVectorIndex, 1) = -3.990024937655861;
    tStateMng.setTrialStep(tTrialStep);

    // ********* Allocate Quadratic Line Search *********
    locus::QuadraticLineSearch<double> tLineSearch(tDataFactory);
    tLineSearch.step(tStateMng);

    size_t tOrdinalValue = 7;
    EXPECT_EQ(tOrdinalValue, tLineSearch.getNumIterationsDone());
    tGoldScalarValue = 0.00243606117022465;
    EXPECT_NEAR(tLineSearch.getStepValue(), tGoldScalarValue, tTolerance);
    tGoldScalarValue = 0.99472430176791571;
    EXPECT_NEAR(tStateMng.getCurrentObjectiveValue(), tGoldScalarValue, tTolerance);
    tControl(tVectorIndex, 0) = 1.9926401870390293;
    tControl(tVectorIndex, 1) = 3.9803049928370093;
    LocusTest::checkMultiVectorData(tStateMng.getCurrentControl(), tControl);
}

TEST(LocusTest, NonlinearConjugateGradient_PolakRibiere_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    size_t tOrdinalValue = 37;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_PolakRibiere_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.solve();

    size_t tOrdinalValue = 68;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_FletcherReeves_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setFletcherReevesMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 89;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_FletcherReeves_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setFletcherReevesMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 75;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HestenesStiefel_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHestenesStiefelMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 23;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HestenesStiefel_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHestenesStiefelMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 35;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HagerZhang_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHagerZhangMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 56;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_HagerZhang_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setHagerZhangMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 53;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuanHybrid_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanHybridMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 24;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuanHybrid_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanHybridMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 47;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_GRADIENT, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuan_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 28;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_DaiYuan_Bounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 1.2; // NOTE: DIFFERENT INITIAL GUESS, DIVERGES IF INITIAL GUESS = 2
    tDataMng->setInitialGuess(tScalarValue);
    tScalarValue = -5;
    tDataMng->setControlLowerBounds(tScalarValue);
    tScalarValue = 5;
    tDataMng->setControlUpperBounds(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setDaiYuanMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 37;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::OBJECTIVE_STAGNATION, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

TEST(LocusTest, NonlinearConjugateGradient_PerryShanno_NoBounds)
{
    // ********* Allocate Data Factory *********
    const size_t tNumControls = 2;
    std::shared_ptr<locus::DataFactory<double>> tDataFactory = std::make_shared<locus::DataFactory<double>>();
    tDataFactory->allocateControl(tNumControls);

    // ********* Allocate Nonlinear Conjugate Gradient Stage Manager *********
    locus::Rosenbrock<double> tObjective;
    std::shared_ptr<locus::NonlinearConjugateGradientStandardStageMng<double>> tStageMng =
            std::make_shared<locus::NonlinearConjugateGradientStandardStageMng<double>>(*tDataFactory, tObjective);

    // ********* Allocate Nonlinear Conjugate Gradient Data Manager *********
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<double>> tDataMng =
            std::make_shared<locus::NonlinearConjugateGradientDataMng<double>>(*tDataFactory);
    double tScalarValue = 2;
    tDataMng->setInitialGuess(tScalarValue);

    // ********* Allocate Nonlinear Conjugate Gradient Algorithm *********
    locus::NonlinearConjugateGradient<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
    tAlgorithm.setPerryShannoMethod(tDataFactory.operator*());
    tAlgorithm.solve();

    size_t tOrdinalValue = 33;
    EXPECT_EQ(tOrdinalValue, tAlgorithm.getNumIterationsDone());
    EXPECT_EQ(locus::algorithm::stop_t::NORM_STEP, tAlgorithm.getStoppingCriteria());
    tScalarValue = 1;
    locus::StandardVector<double> tVector(tNumControls, tScalarValue);
    const size_t tVectorIndex = 0;
    const double tTolerance = 1e-4;
    LocusTest::checkVectorData(tDataMng->getCurrentControl(tVectorIndex), tVector, tTolerance);
}

/* ******************************************************************* */
/* ************** METHOD OF MOVING ASYMPTOTES UNIT TESTS ************* */
/* ******************************************************************* */

TEST(LocusTest, DualProblemStageMng)
{
    // ********* Allocate Data Factory *********
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    std::shared_ptr<locus::MethodMovingAsymptotes<double>> tSubProblem =
            std::make_shared<locus::MethodMovingAsymptotes<double>>(tDataFactory);
    std::shared_ptr<locus::PrimalProblemStageMng<double>> tPrimalProblem =
            std::make_shared<locus::PrimalProblemStageMng<double>>(tDataFactory);
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<double>> tDataMng =
            std::make_shared<locus::ConservativeConvexSeparableAppxDataMng<double>>(tDataFactory);

    locus::ConservativeConvexSeparableApproximationsAlgorithm<double> tAlgorithm(tPrimalProblem, tDataMng, tSubProblem);
}

}
