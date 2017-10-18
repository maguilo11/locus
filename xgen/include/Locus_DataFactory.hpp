/*
 * Locus_DataFactory.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DATAFACTORY_HPP_
#define LOCUS_DATAFACTORY_HPP_

#include <cassert>

#include "Locus_StandardMultiVector.hpp"
#include "Locus_StandardVectorReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class DataFactory
{
public:
    DataFactory() :
            mDual(),
            mState(),
            mControl(),
            mDualReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ScalarType, OrdinalType>>()),
            mStateReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ScalarType, OrdinalType>>()),
            mControlReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ScalarType, OrdinalType>>())
    {
        const OrdinalType tNumStates = 1;
        this->allocateState(tNumStates);
    }
    ~DataFactory()
    {
    }

    void allocateDual(const OrdinalType & aNumElements, OrdinalType aNumVectors = 1)
    {
        locus::StandardVector<ScalarType, OrdinalType> tVector(aNumElements);
        mDual = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(aNumVectors, tVector);
    }
    void allocateDual(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mDual = aInput.create();
    }
    void allocateDual(const locus::Vector<ScalarType, OrdinalType> & aInput, OrdinalType aNumVectors = 1)
    {
        mDual = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(aNumVectors, aInput);
    }
    void allocateDualReductionOperations(const locus::ReductionOperations<ScalarType, OrdinalType> & aInput)
    {
        mDualReductionOperations = aInput.create();
    }

    void allocateState(const OrdinalType & aNumElements, OrdinalType aNumVectors = 1)
    {
        locus::StandardVector<ScalarType, OrdinalType> tVector(aNumElements);
        mState = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(aNumVectors, tVector);
    }
    void allocateState(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mState = aInput.create();
    }
    void allocateState(const locus::Vector<ScalarType, OrdinalType> & aInput, OrdinalType aNumVectors = 1)
    {
        mState = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(aNumVectors, aInput);
    }
    void allocateStateReductionOperations(const locus::ReductionOperations<ScalarType, OrdinalType> & aInput)
    {
        mStateReductionOperations = aInput.create();
    }

    void allocateControl(const OrdinalType & aNumElements, OrdinalType aNumVectors = 1)
    {
        locus::StandardVector<ScalarType, OrdinalType> tVector(aNumElements);
        mControl = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(aNumVectors, tVector);
    }
    void allocateControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
    {
        mControl = aInput.create();
    }
    void allocateControl(const locus::Vector<ScalarType, OrdinalType> & aInput, OrdinalType aNumVectors = 1)
    {
        mControl = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(aNumVectors, aInput);
    }
    void allocateControlReductionOperations(const locus::ReductionOperations<ScalarType, OrdinalType> & aInput)
    {
        mControlReductionOperations = aInput.create();
    }

    const locus::MultiVector<ScalarType, OrdinalType> & dual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator*());
    }
    const locus::Vector<ScalarType, OrdinalType> & dual(const OrdinalType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mDual->getNumVectors());

        return (mDual.operator *().operator [](aVectorIndex));
    }
    const locus::ReductionOperations<ScalarType, OrdinalType> & getDualReductionOperations() const
    {
        assert(mDualReductionOperations.get() != nullptr);
        return (mDualReductionOperations.operator *());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & state() const
    {
        assert(mState.get() != nullptr);
        return (mState.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & state(const OrdinalType & aVectorIndex) const
    {
        assert(mState.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mState->getNumVectors());

        return (mState.operator *().operator [](aVectorIndex));
    }
    const locus::ReductionOperations<ScalarType, OrdinalType> & getStateReductionOperations() const
    {
        assert(mStateReductionOperations.get() != nullptr);
        return (mStateReductionOperations.operator *());
    }

    const locus::MultiVector<ScalarType, OrdinalType> & control() const
    {
        assert(mControl.get() != nullptr);
        return (mControl.operator *());
    }
    const locus::Vector<ScalarType, OrdinalType> & control(const OrdinalType & aVectorIndex) const
    {
        assert(mControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<OrdinalType>(0));
        assert(aVectorIndex < mControl->getNumVectors());

        return (mControl.operator *().operator [](aVectorIndex));
    }
    const locus::ReductionOperations<ScalarType, OrdinalType> & getControlReductionOperations() const
    {
        assert(mControlReductionOperations.get() != nullptr);
        return (mControlReductionOperations.operator *());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mState;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControl;

    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mStateReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mControlReductionOperations;

private:
    DataFactory(const locus::DataFactory<ScalarType, OrdinalType>&);
    locus::DataFactory<ScalarType, OrdinalType> & operator=(const locus::DataFactory<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_DATAFACTORY_HPP_ */
