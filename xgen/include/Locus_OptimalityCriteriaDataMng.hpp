/*
 * Locus_OptimalityCriteriaDataMng.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIADATAMNG_HPP_
#define LOCUS_OPTIMALITYCRITERIADATAMNG_HPP_

#include <limits>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_ReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaDataMng
{
public:
    explicit OptimalityCriteriaDataMng(const locus::DataFactory<ScalarType, OrdinalType> & aFactory) :
            mStagnationMeasure(std::numeric_limits<ScalarType>::max()),
            mMaxInequalityValue(std::numeric_limits<ScalarType>::max()),
            mNormObjectiveGradient(std::numeric_limits<ScalarType>::max()),
            mCurrentObjectiveValue(std::numeric_limits<ScalarType>::max()),
            mPreviousObjectiveValue(std::numeric_limits<ScalarType>::max()),
            mCurrentDual(),
            mDualWorkVector(),
            mControlWorkVector(),
            mCurrentInequalityValues(),
            mCurrentControl(aFactory.control().create()),
            mPreviousControl(aFactory.control().create()),
            mObjectiveGradient(aFactory.control().create()),
            mInequalityGradient(aFactory.control().create()),
            mControlLowerBounds(aFactory.control().create()),
            mControlUpperBounds(aFactory.control().create()),
            mDualReductionOperations(),
            mControlReductionOperations()
    {
        this->initialize(aFactory);
    }
    ~OptimalityCriteriaDataMng()
    {
    }

    OrdinalType getNumConstraints() const
    {
        OrdinalType tNumVectors = mCurrentDual->size();
        return (tNumVectors);
    }
    OrdinalType getNumControlVectors() const
    {
        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        return (tNumVectors);
    }

    ScalarType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }
    ScalarType getMaxInequalityValue() const
    {
        return (mMaxInequalityValue);
    }
    ScalarType getNormObjectiveGradient() const
    {
        return (mNormObjectiveGradient);
    }

    void computeStagnationMeasure()
    {
        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ScalarType> storage(tNumVectors, std::numeric_limits<ScalarType>::min());
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            locus::Vector<ScalarType, OrdinalType> & tCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkVector->update(1., tCurrentControl, 0.);
            locus::Vector<ScalarType, OrdinalType> & tPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkVector->update(-1., tPreviousControl, 1.);
            mControlWorkVector->modulus();
            storage[tIndex] = mControlReductionOperations->max(*mControlWorkVector);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    void computeMaxInequalityValue()
    {
        mDualWorkVector->update(1., *mCurrentInequalityValues, 0.);
        mDualWorkVector->modulus();
        mMaxInequalityValue = mDualReductionOperations->max(*mDualWorkVector);
    }
    void computeNormObjectiveGradient()
    {
        ScalarType tCummulativeDotProduct = 0.;
        OrdinalType tNumVectors = mObjectiveGradient->getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyGradient = (*mObjectiveGradient)[tIndex];
            tCummulativeDotProduct += tMyGradient.dot(tMyGradient);
        }
        mNormObjectiveGradient = std::sqrt(tCummulativeDotProduct);
    }

    ScalarType getCurrentObjectiveValue() const
    {
        return (mCurrentObjectiveValue);
    }
    void setCurrentObjectiveValue(const ScalarType & aInput)
    {
        mCurrentObjectiveValue = aInput;
    }
    ScalarType getPreviousObjectiveValue() const
    {
        return (mPreviousObjectiveValue);
    }
    void setPreviousObjectiveValue(const ScalarType & aInput)
    {
        mPreviousObjectiveValue = aInput;
    }

    const locus::Vector<ScalarType, OrdinalType> & getCurrentDual() const
    {
        assert(mCurrentDual.get() != nullptr);
        assert(mCurrentDual->size() > static_cast<OrdinalType>(0));
        return (mCurrentDual.operator *());
    }
    void setCurrentDual(const OrdinalType & aIndex, const ScalarType & aValue)
    {
        assert(mCurrentDual.get() != nullptr);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mCurrentDual->size());
        mCurrentDual->operator [](aIndex) = aValue;
    }
    const locus::Vector<ScalarType, OrdinalType> & getCurrentConstraintValues() const
    {
        assert(mCurrentInequalityValues.get() != nullptr);
        assert(mCurrentInequalityValues->size() > static_cast<OrdinalType>(0));
        return (mCurrentInequalityValues.operator *());
    }
    const ScalarType & getCurrentConstraintValues(const OrdinalType & aIndex) const
    {
        assert(mCurrentInequalityValues.get() != nullptr);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mCurrentInequalityValues->size());
        return(mCurrentInequalityValues->operator [](aIndex));
    }
    void setCurrentConstraintValue(const OrdinalType & aIndex, const ScalarType & aValue)
    {
        assert(mCurrentInequalityValues.get() != nullptr);
        assert(aIndex >= static_cast<OrdinalType>(0));
        assert(aIndex < mCurrentInequalityValues->size());
        mCurrentInequalityValues->operator [](aIndex) = aValue;
    }

    void setInitialGuess(const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = mCurrentControl->getNumVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const ScalarType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
    }
    void setInitialGuess(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aInitialGuess)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInitialGuess, 0.);
    }
    void setInitialGuess(const locus::MultiVector<ScalarType, OrdinalType> & aInitialGuess)
    {
        assert(aInitialGuess.getNumVectors() == mCurrentControl->getNumVectors());

        const OrdinalType tNumVectors = aInitialGuess.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputInitialGuess = aInitialGuess[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyControl = mCurrentControl->operator [](tIndex);
            assert(tInputInitialGuess.size() == tMyControl.size());
            tMyControl.update(1., tInputInitialGuess, 0.);
        }
    }

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
    void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() == mCurrentControl->getNumVectors());

        const OrdinalType tNumVectors = aControl.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputControl = aControl[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyControl = mCurrentControl->operator [](tIndex);
            assert(tInputControl.size() == tMyControl.size());
            tMyControl.update(1., tInputControl, 0.);
        }
    }
    void setCurrentControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aControl)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aControl, 0.);
    }

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
    void setPreviousControl(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() == mPreviousControl->getNumVectors());

        const OrdinalType tNumVectors = aControl.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputControl = aControl[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyControl = mPreviousControl->operator [](tIndex);
            assert(tInputControl.size() == tMyControl.size());
            tMyControl.update(1., tInputControl, 0.);
        }
    }
    void setPreviousControl(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aControl)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        mPreviousControl->operator [](aVectorIndex).update(1., aControl, 0.);
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getObjectiveGradient() const
    {
        assert(mObjectiveGradient.get() != nullptr);

        return (mObjectiveGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getObjectiveGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mObjectiveGradient->getNumVectors());

        return (mObjectiveGradient->operator [](aVectorIndex));
    }
    void setObjectiveGradient(const locus::MultiVector<ScalarType, OrdinalType> & aGradient)
    {
        assert(aGradient.getNumVectors() == mObjectiveGradient->getNumVectors());

        const OrdinalType tNumVectors = aGradient.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputGradient = aGradient[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyObjectiveGradient = mObjectiveGradient->operator [](tIndex);
            assert(tInputGradient.size() == tMyObjectiveGradient.size());
            tMyObjectiveGradient.update(1., tInputGradient, 0.);
        }
    }
    void setObjectiveGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aGradient)
    {
        assert(mObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mObjectiveGradient->getNumVectors());

        mObjectiveGradient->operator [](aVectorIndex).update(1., aGradient, 0.);
    }

    const locus::MultiVector<ScalarType, OrdinalType> & getInequalityGradient() const
    {
        assert(mInequalityGradient.get() != nullptr);

        return (mInequalityGradient.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & getInequalityGradient(const OrdinalType & aVectorIndex) const
    {
        assert(mInequalityGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInequalityGradient->getNumVectors());

        return (mInequalityGradient->operator [](aVectorIndex));
    }
    void setInequalityGradient(const locus::MultiVector<ScalarType, OrdinalType> & aGradient)
    {
        assert(aGradient.getNumVectors() == mInequalityGradient->getNumVectors());

        const OrdinalType tNumVectors = aGradient.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputGradient = aGradient[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyInequalityGradient = mInequalityGradient->operator [](tIndex);
            assert(tInputGradient.size() == tMyInequalityGradient.size());
            tMyInequalityGradient.update(1., tInputGradient, 0.);
        }
    }
    void setInequalityGradient(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aGradient)
    {
        assert(mInequalityGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mInequalityGradient->getNumVectors());

        mInequalityGradient->operator [](aVectorIndex).update(1., aGradient, 0.);
    }

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
    void setControlLowerBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aLowerBound)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).update(1., aLowerBound, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aLowerBound)
    {
        assert(aLowerBound.getNumVectors() == mControlLowerBounds->getNumVectors());

        const OrdinalType tNumVectors = aLowerBound.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputLowerBound = aLowerBound[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyLowerBound = mControlLowerBounds->operator [](tIndex);
            assert(tInputLowerBound.size() == tMyLowerBound.size());
            tMyLowerBound.update(1., tInputLowerBound, 0.);
        }
    }

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
    void setControlUpperBounds(const OrdinalType & aVectorIndex, const locus::Vector<ScalarType, OrdinalType> & aUpperBound)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aUpperBound, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aUpperBound)
    {
        assert(aUpperBound.getNumVectors() == mControlUpperBounds->getNumVectors());

        const OrdinalType tNumVectors = aUpperBound.getNumVectors();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tInputUpperBound = aUpperBound[tIndex];
            locus::Vector<ScalarType, OrdinalType> & tMyUpperBound = mControlUpperBounds->operator [](tIndex);
            assert(tInputUpperBound.size() == tMyUpperBound.size());
            tMyUpperBound.update(1., tInputUpperBound, 0.);
        }
    }

private:
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aFactory)
    {
        assert(aFactory.dual().getNumVectors() > static_cast<OrdinalType>(0));
        assert(aFactory.control().getNumVectors() > static_cast<OrdinalType>(0));

        const OrdinalType tVECTOR_INDEX = 0;
        mCurrentDual = aFactory.dual(tVECTOR_INDEX).create();
        mDualWorkVector = aFactory.dual(tVECTOR_INDEX).create();
        mControlWorkVector = aFactory.control(tVECTOR_INDEX).create();
        mCurrentInequalityValues = aFactory.dual(tVECTOR_INDEX).create();

        mDualReductionOperations = aFactory.getDualReductionOperations().create();
        mControlReductionOperations = aFactory.getControlReductionOperations().create();
    }

private:
    ScalarType mStagnationMeasure;
    ScalarType mMaxInequalityValue;
    ScalarType mNormObjectiveGradient;
    ScalarType mCurrentObjectiveValue;
    ScalarType mPreviousObjectiveValue;

    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mCurrentDual;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mDualWorkVector;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mControlWorkVector;
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> mCurrentInequalityValues;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInequalityGradient;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlUpperBounds;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    OptimalityCriteriaDataMng(const locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType>&);
    locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIADATAMNG_HPP_ */
