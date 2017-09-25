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

namespace locus
{

/**********************************************************************************************************/
/**************************************** LINEAR ALGEBRA OPERATIONS ***************************************/
/**********************************************************************************************************/

template<typename ElementType, typename IndexType = size_t>
class Vector
{
public:
    virtual ~Vector()
    {
    }

    //! Scales a Vector by a ElementType constant.
    virtual void scale(const ElementType & aInput) = 0;
    //! Entry-Wise product of two vectors.
    virtual void entryWiseProduct(const locus::Vector<ElementType, IndexType> & aInput) = 0;
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    virtual void update(const ElementType & aAlpha,
                        const locus::Vector<ElementType, IndexType> & aInputVector,
                        const ElementType & aBeta) = 0;
    //! Computes the absolute value of each element in the container.
    virtual void modulus() = 0;
    //! Returns the inner product of two vectors.
    virtual ElementType dot(const locus::Vector<ElementType, IndexType> & aInputVector) const = 0;
    //! Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    virtual void fill(const ElementType & aValue) = 0;
    //! Returns the number of local elements in the Vector.
    virtual IndexType size() const = 0;
    //! Creates an object of type locus::Vector
    virtual std::shared_ptr<locus::Vector<ElementType, IndexType>> create() const = 0;
    //! Operator overloads the square bracket operator
    virtual ElementType & operator [](const IndexType & aIndex) = 0;
    //! Operator overloads the square bracket operator
    virtual const ElementType & operator [](const IndexType & aIndex) const = 0;
};

template<typename ElementType, typename IndexType = size_t>
class StandardVector : public locus::Vector<ElementType, IndexType>
{
public:
    explicit StandardVector(const std::vector<ElementType> & aInput) :
            mData(aInput)
    {
    }
    StandardVector(const IndexType & aNumElements, ElementType aValue = 0) :
            mData(std::vector<ElementType>(aNumElements, aValue))
    {
    }
    virtual ~StandardVector()
    {
    }

    //! Scales a Vector by a ElementType constant.
    void scale(const ElementType & aInput)
    {
        IndexType tLength = this->size();
        for(IndexType tIndex = 0; tIndex < tLength; tIndex++)
        {
            mData[tIndex] = aInput * mData[tIndex];
        }
    }
    //! Element-wise multiplication of two vectors.
    void entryWiseProduct(const locus::Vector<ElementType, IndexType> & aInput)
    {
        IndexType tMyDataSize = mData.size();
        assert(aInput.size() == tMyDataSize);

        for(IndexType tIndex = 0; tIndex < tMyDataSize; tIndex++)
        {
            mData[tIndex] = aInput[tIndex] * mData[tIndex];
        }
    }
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const ElementType & aAlpha,
                const locus::Vector<ElementType, IndexType> & aInputVector,
                const ElementType & aBeta)
    {
        IndexType tMyDataSize = mData.size();
        assert(aInputVector.size() == tMyDataSize);
        for(IndexType tIndex = 0; tIndex < tMyDataSize; tIndex++)
        {
            mData[tIndex] = aBeta * mData[tIndex] + aAlpha * aInputVector[tIndex];
        }
    }
    //! Computes the absolute value of each element in the container.
    void modulus()
    {
        IndexType tLength = this->size();
        for(IndexType tIndex = 0; tIndex < tLength; tIndex++)
        {
            mData[tIndex] = std::abs(mData[tIndex]);
        }
    }
    //! Returns the inner product of two vectors.
    ElementType dot(const locus::Vector<ElementType, IndexType> & aInputVector) const
    {
        assert(aInputVector.size() == static_cast<IndexType>(mData.size()));

        const locus::StandardVector<ElementType, IndexType>& tInputVector =
                dynamic_cast<const locus::StandardVector<ElementType, IndexType>&>(aInputVector);

        ElementType tBaseValue = 0;
        ElementType tOutput = std::inner_product(mData.begin(), mData.end(), tInputVector.mData.begin(), tBaseValue);
        return (tOutput);
    }
    //! Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ElementType & aValue)
    {
        std::fill(mData.begin(), mData.end(), aValue);
    }
    //! Returns the number of local elements in the Vector.
    IndexType size() const
    {
        IndexType tOutput = mData.size();
        return (tOutput);
    }
    //! Creates object of type locus::Vector
    std::shared_ptr<locus::Vector<ElementType, IndexType>> create() const
    {
        const ElementType tBaseValue = 0;
        const IndexType tNumElements = this->size();
        std::shared_ptr<locus::Vector<ElementType, IndexType>> tOutput =
                std::make_shared<locus::StandardVector<ElementType, IndexType>>(tNumElements, tBaseValue);
        return (tOutput);
    }
    //! Operator overloads the square bracket operator
    ElementType & operator [](const IndexType & aIndex)
    {
        assert(aIndex < this->size());
        assert(aIndex >= static_cast<IndexType>(0));

        return (mData[aIndex]);
    }
    //! Operator overloads the square bracket operator
    const ElementType & operator [](const IndexType & aIndex) const
    {
        assert(aIndex < this->size());
        assert(aIndex >= static_cast<IndexType>(0));

        return (mData[aIndex]);
    }

private:
    std::vector<ElementType> mData;

private:
    StandardVector(const locus::StandardVector<ElementType, IndexType> &);
    locus::StandardVector<ElementType, IndexType> & operator=(const locus::StandardVector<ElementType, IndexType> &);
};

template<typename ElementType, typename IndexType = size_t>
class ReductionOperations
{
public:
    virtual ~ReductionOperations()
    {
    }

    //! Returns the maximum element in range
    virtual ElementType max(const locus::Vector<ElementType, IndexType> & aInput) const = 0;
    //! Returns the minimum element in range
    virtual ElementType min(const locus::Vector<ElementType, IndexType> & aInput) const = 0;
    //! Returns the sum of all the elements in container.
    virtual ElementType sum(const locus::Vector<ElementType, IndexType> & aInput) const = 0;
    //! Creates object of type locus::ReductionOperations
    virtual std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> create() const = 0;
};

template<typename ElementType, typename IndexType = size_t>
class StandardVectorReductionOperations : public locus::ReductionOperations<ElementType, IndexType>
{
public:
    StandardVectorReductionOperations()
    {
    }
    virtual ~StandardVectorReductionOperations()
    {
    }

    //! Returns the maximum element in range
    ElementType max(const locus::Vector<ElementType, IndexType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ElementType tValue = 0;
        const IndexType tSize = aInput.size();
        std::vector<ElementType> tCopy(tSize, tValue);
        for(IndexType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ElementType aMaxValue = *std::max_element(tCopy.begin(), tCopy.end());
        return (aMaxValue);
    }
    //! Returns the minimum element in range
    ElementType min(const locus::Vector<ElementType, IndexType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ElementType tValue = 0;
        const IndexType tSize = aInput.size();
        std::vector<ElementType> tCopy(tSize, tValue);
        for(IndexType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ElementType aMinValue = *std::min_element(tCopy.begin(), tCopy.end());
        return (aMinValue);
    }
    //! Returns the sum of all the elements in container.
    ElementType sum(const locus::Vector<ElementType, IndexType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ElementType tValue = 0;
        const IndexType tSize = aInput.size();
        std::vector<ElementType> tCopy(tSize, tValue);
        for(IndexType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ElementType tBaseValue = 0;
        ElementType tSum = std::accumulate(tCopy.begin(), tCopy.end(), tBaseValue);
        return (tSum);
    }
    //! Creates an instance of type locus::ReductionOperations
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> tCopy =
                std::make_shared<StandardVectorReductionOperations<ElementType, IndexType>>();
        return (tCopy);
    }

private:
    StandardVectorReductionOperations(const locus::StandardVectorReductionOperations<ElementType, IndexType> &);
    locus::StandardVectorReductionOperations<ElementType, IndexType> & operator=(const locus::StandardVectorReductionOperations<ElementType, IndexType> &);
};

template<typename ElementType, typename IndexType = size_t>
class DistributedReductionOperations : public locus::ReductionOperations<ElementType, IndexType>
{
public:
    DistributedReductionOperations()
    {
    }
    virtual ~DistributedReductionOperations()
    {
    }

    //! Returns the maximum element in range
    ElementType max(const locus::Vector<ElementType, IndexType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ElementType tValue = 0;
        const IndexType tSize = aInput.size();
        std::vector<ElementType> tCopy(tSize, tValue);
        for(IndexType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }
        ElementType aLocalMaxValue = *std::max_element(tCopy.begin(), tCopy.end());

        ElementType aGlobalMaxValue = aLocalMaxValue;
        MPI_Allreduce(&aGlobalMaxValue, &aLocalMaxValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        return (aLocalMaxValue);
    }
    //! Returns the minimum element in range
    ElementType min(const locus::Vector<ElementType, IndexType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ElementType tValue = 0;
        const IndexType tSize = aInput.size();
        std::vector<ElementType> tCopy(tSize, tValue);
        for(IndexType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }
        ElementType aLocalMinValue = *std::min_element(tCopy.begin(), tCopy.end());

        ElementType aGlobalMinValue = aLocalMinValue;
        MPI_Allreduce(&aGlobalMinValue, &aLocalMinValue, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        return (aGlobalMinValue);
    }
    //! Returns the sum of all the elements in container.
    ElementType sum(const locus::Vector<ElementType, IndexType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ElementType tValue = 0;
        const IndexType tSize = aInput.size();
        std::vector<ElementType> tCopy(tSize, tValue);
        for(IndexType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ElementType tBaseValue = 0;
        ElementType tLocalSum = std::accumulate(tCopy.begin(), tCopy.end(), tBaseValue);

        ElementType tGlobalSum = tLocalSum;
        MPI_Allreduce(&tGlobalSum, &tLocalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return (tLocalSum);
    }
    //! Creates an instance of type locus::ReductionOperations
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> tCopy =
                std::make_shared<DistributedReductionOperations<ElementType, IndexType>>();
        return (tCopy);
    }
    //! Return number of ranks (i.e. processes)
    IndexType getNumRanks() const
    {
        int tNumRanks = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &tNumRanks);
        assert(tNumRanks > static_cast<int>(0));
        return (tNumRanks);
    }

private:
    DistributedReductionOperations(const locus::DistributedReductionOperations<ElementType, IndexType> &);
    locus::DistributedReductionOperations<ElementType, IndexType> & operator=(const locus::DistributedReductionOperations<ElementType, IndexType> &);
};

template<typename ElementType, typename IndexType = size_t>
class MultiVector
{
public:
    virtual ~MultiVector()
    {
    }

    //! Returns number of vectors
    virtual IndexType getNumVectors() const = 0;
    //! Creates a copy of type MultiVector
    virtual std::shared_ptr<locus::MultiVector<ElementType, IndexType>> create() const = 0;
    //! Operator overloads the square bracket operator
    virtual locus::Vector<ElementType, IndexType> & operator [](const IndexType & aVectorIndex) = 0;
    //! Operator overloads the square bracket operator
    virtual const locus::Vector<ElementType, IndexType> & operator [](const IndexType & aVectorIndex) const = 0;
    //! Operator overloads the square bracket operator
    virtual ElementType & operator ()(const IndexType & aVectorIndex, const IndexType & aElementIndex) = 0;
    //! Operator overloads the square bracket operator
    virtual const ElementType & operator ()(const IndexType & aVectorIndex, const IndexType & aElementIndex) const = 0;
};

template<typename ElementType, typename IndexType = size_t>
class StandardMultiVector : public locus::MultiVector<ElementType, IndexType>
{
public:
    StandardMultiVector(const IndexType & aNumVectors, const std::vector<ElementType> & aStandardVectorTemplate) :
        mData(std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>>(aNumVectors))
    {
        this->initialize(aStandardVectorTemplate);
    }
    StandardMultiVector(const IndexType & aNumVectors, const locus::Vector<ElementType, IndexType> & aVectorTemplate) :
        mData(std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>>(aNumVectors))
    {
        this->initialize(aVectorTemplate);
    }
    explicit StandardMultiVector(const std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>>& aMultiVectorTemplate) :
        mData(std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>>(aMultiVectorTemplate.size()))
    {
        this->initialize(aMultiVectorTemplate);
    }
    StandardMultiVector(const IndexType & aNumVectors, const IndexType & aNumElementsPerVector, ElementType aValue = 0) :
        mData(std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>>(aNumVectors))
    {
        this->initialize(aNumElementsPerVector, aValue);
    }
    virtual ~StandardMultiVector()
    {
    }

    //! Creates a copy of type MultiVector
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> create() const
    {
        const IndexType tVectorIndex = 0;
        const IndexType tNumVectors = this->getNumVectors();
        std::shared_ptr<locus::MultiVector<ElementType, IndexType>> tOutput;
        const locus::Vector<ElementType, IndexType> & tVectorTemplate = *mData[tVectorIndex];
        tOutput = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(tNumVectors, tVectorTemplate);
        return (tOutput);
    }
    //! Number of vectors
    IndexType getNumVectors() const
    {
        IndexType tNumVectors = mData.size();
        return (tNumVectors);
    }
    //! Operator overloads the square bracket operator
    virtual locus::Vector<ElementType, IndexType> & operator [](const IndexType & aVectorIndex)
    {
        assert(mData.empty() == false);
        assert(aVectorIndex < this->getNumVectors());

        return (mData[aVectorIndex].operator *());
    }
    //! Operator overloads the square bracket operator
    virtual const locus::Vector<ElementType, IndexType> & operator [](const IndexType & aVectorIndex) const
    {
        assert(mData.empty() == false);
        assert(mData[aVectorIndex].get() != nullptr);
        assert(aVectorIndex < this->getNumVectors());

        return (mData[aVectorIndex].operator *());
    }
    //! Operator overloads the square bracket operator
    virtual ElementType & operator ()(const IndexType & aVectorIndex, const IndexType & aElementIndex)
    {
        assert(aVectorIndex < this->getNumVectors());
        assert(aElementIndex < mData[aVectorIndex]->size());

        return (mData[aVectorIndex].operator *().operator [](aElementIndex));
    }
    //! Operator overloads the square bracket operator
    virtual const ElementType & operator ()(const IndexType & aVectorIndex, const IndexType & aElementIndex) const
    {
        assert(aVectorIndex < this->getNumVectors());
        assert(aElementIndex < mData[aVectorIndex]->size());

        return (mData[aVectorIndex].operator *().operator [](aElementIndex));
    }

private:
    void initialize(const IndexType & aNumElementsPerVector, const ElementType & aValue)
    {
        locus::StandardVector<ElementType,IndexType> tVector(aNumElementsPerVector);

        IndexType tNumVectors = mData.size();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            mData[tIndex] = tVector.create();
            mData[tIndex]->fill(aValue);
        }
    }
    void initialize(const std::vector<ElementType> & aVectorTemplate)
    {
        locus::StandardVector<ElementType,IndexType> tVector(aVectorTemplate);

        IndexType tNumVectors = mData.size();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            mData[tIndex] = tVector.create();
        }
    }
    void initialize(const locus::Vector<ElementType, IndexType> & aVectorTemplate)
    {
        IndexType tNumVectors = mData.size();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            mData[tIndex] = aVectorTemplate.create();
        }
    }
    void initialize(const std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>> & aMultiVectorTemplate)
    {
        assert(mData.size() > 0);
        assert(aMultiVectorTemplate.size() > 0);
        IndexType tNumVectors = aMultiVectorTemplate.size();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            assert(aMultiVectorTemplate[tIndex]->size() > 0);
            mData[tIndex] = aMultiVectorTemplate[tIndex]->create();
            mData[tIndex]->update(static_cast<ElementType>(1.), *aMultiVectorTemplate[tIndex], static_cast<ElementType>(0.));
        }
    }

private:
    std::vector<std::shared_ptr<locus::Vector<ElementType, IndexType>>> mData;

private:
    StandardMultiVector(const locus::StandardMultiVector<ElementType, IndexType>&);
    locus::StandardMultiVector<ElementType, IndexType> & operator=(const locus::StandardMultiVector<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType>
ElementType dot(const locus::MultiVector<ElementType, IndexType> & aVectorOne,
                const locus::MultiVector<ElementType, IndexType> & aVectorTwo)
{
    assert(aVectorOne.getNumVectors() > static_cast<IndexType>(0));
    assert(aVectorOne.getNumVectors() == aVectorTwo.getNumVectors());

    ElementType tCummulativeSum = 0;
    IndexType tNumVectors = aVectorOne.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aVectorOne[tVectorIndex].size() > static_cast<IndexType>(0));
        assert(aVectorOne[tVectorIndex].size() == aVectorTwo[tVectorIndex].size());
        const locus::Vector<ElementType, IndexType> & tVectorOne = aVectorOne[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tVectorTwo = aVectorTwo[tVectorIndex];
        tCummulativeSum += tVectorOne.dot(tVectorTwo);
    }
    return(tCummulativeSum);
}

template<typename ElementType, typename IndexType>
void entryWiseProduct(const locus::MultiVector<ElementType, IndexType> & aInput,
                      locus::MultiVector<ElementType, IndexType> & aOutput)
{
    assert(aInput.getNumVectors() > static_cast<IndexType>(0));
    assert(aInput.getNumVectors() == aOutput.getNumVectors());

    IndexType tNumVectors = aInput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aInput[tVectorIndex].size() > static_cast<IndexType>(0));
        assert(aInput[tVectorIndex].size() == aOutput[tVectorIndex].size());
        locus::Vector<ElementType, IndexType> & tOutput = aOutput[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tInput = aInput[tVectorIndex];
        tOutput.entryWiseProduct(tInput);
    }
}

template<typename ElementType, typename IndexType>
void fill(const ElementType & aScalar, locus::MultiVector<ElementType, IndexType> & aOutput)
{
    assert(aOutput.getNumVectors() > static_cast<IndexType>(0));

    IndexType tNumVectors = aOutput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aOutput[tVectorIndex].size() > static_cast<IndexType>(0));
        locus::Vector<ElementType, IndexType> & tVector = aOutput[tVectorIndex];
        tVector.fill(aScalar);
    }
}

template<typename ElementType, typename IndexType>
void scale(const ElementType & aScalar, locus::MultiVector<ElementType, IndexType> & aOutput)
{
    assert(aOutput.getNumVectors() > static_cast<IndexType>(0));

    IndexType tNumVectors = aOutput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aOutput[tVectorIndex].size() > static_cast<IndexType>(0));
        locus::Vector<ElementType, IndexType> & tVector = aOutput[tVectorIndex];
        tVector.scale(aScalar);
    }
}

template<typename ElementType, typename IndexType>
ElementType norm(const locus::MultiVector<ElementType, IndexType> & aInput)
{
    ElementType tDotProduct = locus::dot(aInput, aInput);
    ElementType tNorm = std::sqrt(tDotProduct);
    return(tNorm);
}

//! Update vector values with scaled values of A, this = beta*this + alpha*A.
template<typename ElementType, typename IndexType>
void update(const ElementType & aAlpha,
            const locus::MultiVector<ElementType, IndexType> & aInput,
            const ElementType & aBeta,
            locus::MultiVector<ElementType, IndexType> & aOutput)
{
    assert(aInput.getNumVectors() > static_cast<IndexType>(0));
    assert(aInput.getNumVectors() == aOutput.getNumVectors());

    IndexType tNumVectors = aInput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aInput[tVectorIndex].size() > static_cast<IndexType>(0));
        assert(aInput[tVectorIndex].size() == aOutput[tVectorIndex].size());
        locus::Vector<ElementType, IndexType> & tOutputVector = aOutput[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tInputVector = aInput[tVectorIndex];
        tOutputVector.update(aAlpha, tInputVector, aBeta);
    }
}

template<typename ElementType, typename IndexType>
void gemv(const ElementType & aAlpha,
          const locus::MultiVector<ElementType, IndexType> & aMatrix,
          const locus::Vector<ElementType, IndexType> & aVector,
          const ElementType & aBeta,
          locus::Vector<ElementType, IndexType> & aOutput,
          bool aTranspose = false)
{
    const IndexType tNumElements = aMatrix[0].size();
    const IndexType tNumVectors = aMatrix.getNumVectors();

    if(aTranspose == false)
    {
        assert(tNumVectors == aOutput.size());
        assert(tNumElements == aVector.size());

        aOutput.scale(aBeta);
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            ElementType tOutputRowValue = aMatrix[tVectorIndex].dot(aVector);
            tOutputRowValue = aAlpha * tOutputRowValue;
            aOutput[tVectorIndex] = aOutput[tVectorIndex] + tOutputRowValue;
        }
    }
    else
    {
        assert(tNumVectors == aVector.size());
        assert(tNumElements == aOutput.size());

        aOutput.scale(aBeta);
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            ElementType tScaleFactor = aAlpha * aVector[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tVector = aMatrix[tVectorIndex];
            aOutput.update(tScaleFactor, tVector, static_cast<ElementType>(1));
        }
    }
}

/**********************************************************************************************************/
/************************************** OPTIMALITY CRITERIA ALGORITHM *************************************/
/**********************************************************************************************************/

template<typename ElementType, typename IndexType = size_t>
class DataFactory
{
public:
    DataFactory() :
            mDual(),
            mState(),
            mControl(),
            mDualReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ElementType, IndexType>>()),
            mStateReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ElementType, IndexType>>()),
            mControlReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ElementType, IndexType>>())
    {
        const IndexType tNumStates = 1;
        this->allocateState(tNumStates);
    }
    ~DataFactory()
    {
    }

    void allocateDual(const IndexType & aNumElements, IndexType aNumVectors = 1)
    {
        locus::StandardVector<ElementType, IndexType> tVector(aNumElements);
        mDual = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(aNumVectors, tVector);
    }
    void allocateDual(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        mDual = aInput.create();
    }
    void allocateDual(const locus::Vector<ElementType, IndexType> & aInput, IndexType aNumVectors = 1)
    {
        mDual = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(aNumVectors, aInput);
    }
    void allocateDualReductionOperations(const locus::ReductionOperations<ElementType, IndexType> & aInput)
    {
        mDualReductionOperations = aInput.create();
    }

    void allocateState(const IndexType & aNumElements, IndexType aNumVectors = 1)
    {
        locus::StandardVector<ElementType, IndexType> tVector(aNumElements);
        mState = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(aNumVectors, tVector);
    }
    void allocateState(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        mState = aInput.create();
    }
    void allocateState(const locus::Vector<ElementType, IndexType> & aInput, IndexType aNumVectors = 1)
    {
        mState = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(aNumVectors, aInput);
    }
    void allocateStateReductionOperations(const locus::ReductionOperations<ElementType, IndexType> & aInput)
    {
        mStateReductionOperations = aInput.create();
    }

    void allocateControl(const IndexType & aNumElements, IndexType aNumVectors = 1)
    {
        locus::StandardVector<ElementType, IndexType> tVector(aNumElements);
        mControl = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(aNumVectors, tVector);
    }
    void allocateControl(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        mControl = aInput.create();
    }
    void allocateControl(const locus::Vector<ElementType, IndexType> & aInput, IndexType aNumVectors = 1)
    {
        mControl = std::make_shared<locus::StandardMultiVector<ElementType, IndexType>>(aNumVectors, aInput);
    }
    void allocateControlReductionOperations(const locus::ReductionOperations<ElementType, IndexType> & aInput)
    {
        mControlReductionOperations = aInput.create();
    }

    const locus::MultiVector<ElementType, IndexType> & dual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator *());
    }
    const locus::Vector<ElementType, IndexType> & dual(const IndexType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mDual->getNumVectors());

        return (mDual.operator *().operator [](aVectorIndex));
    }
    const locus::ReductionOperations<ElementType, IndexType> & getDualReductionOperations() const
    {
        assert(mDualReductionOperations.get() != nullptr);
        return (mDualReductionOperations.operator *());
    }

    const locus::MultiVector<ElementType, IndexType> & state() const
    {
        assert(mState.get() != nullptr);
        return (mState.operator *());
    }
    const locus::Vector<ElementType, IndexType> & state(const IndexType & aVectorIndex) const
    {
        assert(mState.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mState->getNumVectors());

        return (mState.operator *().operator [](aVectorIndex));
    }
    const locus::ReductionOperations<ElementType, IndexType> & getStateReductionOperations() const
    {
        assert(mStateReductionOperations.get() != nullptr);
        return (mStateReductionOperations.operator *());
    }

    const locus::MultiVector<ElementType, IndexType> & control() const
    {
        assert(mControl.get() != nullptr);
        return (mControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & control(const IndexType & aVectorIndex) const
    {
        assert(mControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControl->getNumVectors());

        return (mControl.operator *().operator [](aVectorIndex));
    }
    const locus::ReductionOperations<ElementType, IndexType> & getControlReductionOperations() const
    {
        assert(mControlReductionOperations.get() != nullptr);
        return (mControlReductionOperations.operator *());
    }

private:
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDual;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mState;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControl;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mStateReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mControlReductionOperations;

private:
    DataFactory(const locus::DataFactory<ElementType, IndexType>&);
    locus::DataFactory<ElementType, IndexType> & operator=(const locus::DataFactory<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaDataMng
{
public:
    explicit OptimalityCriteriaDataMng(const locus::DataFactory<ElementType, IndexType> & aFactory) :
            mStagnationMeasure(std::numeric_limits<ElementType>::max()),
            mMaxInequalityValue(std::numeric_limits<ElementType>::max()),
            mNormObjectiveGradient(std::numeric_limits<ElementType>::max()),
            mCurrentObjectiveValue(std::numeric_limits<ElementType>::max()),
            mPreviousObjectiveValue(std::numeric_limits<ElementType>::max()),
            mCurrentDual(),
            mDualWorkVector(),
            mControlWorkVector(),
            mCurrentInequalityValues(),
            mCurrentState(aFactory.state().create()),
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

    IndexType getNumConstraints() const
    {
        IndexType tNumVectors = mCurrentDual->size();
        return (tNumVectors);
    }
    IndexType getNumControlVectors() const
    {
        IndexType tNumVectors = mCurrentControl->getNumVectors();
        return (tNumVectors);
    }

    ElementType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }
    ElementType getMaxInequalityValue() const
    {
        return (mMaxInequalityValue);
    }
    ElementType getNormObjectiveGradient() const
    {
        return (mNormObjectiveGradient);
    }

    void computeStagnationMeasure()
    {
        IndexType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ElementType> storage(tNumVectors, std::numeric_limits<ElementType>::min());
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            locus::Vector<ElementType, IndexType> & tCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkVector->update(1., tCurrentControl, 0.);
            locus::Vector<ElementType, IndexType> & tPreviousControl = mPreviousControl->operator[](tIndex);
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
        ElementType tCummulativeDotProduct = 0.;
        IndexType tNumVectors = mObjectiveGradient->getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyGradient = (*mObjectiveGradient)[tIndex];
            tCummulativeDotProduct += tMyGradient.dot(tMyGradient);
        }
        mNormObjectiveGradient = std::sqrt(tCummulativeDotProduct);
    }

    ElementType getCurrentObjectiveValue() const
    {
        return (mCurrentObjectiveValue);
    }
    void setCurrentObjectiveValue(const ElementType & aInput)
    {
        mCurrentObjectiveValue = aInput;
    }
    ElementType getPreviousObjectiveValue() const
    {
        return (mPreviousObjectiveValue);
    }
    void setPreviousObjectiveValue(const ElementType & aInput)
    {
        mPreviousObjectiveValue = aInput;
    }

    const locus::Vector<ElementType, IndexType> & getCurrentDual() const
    {
        assert(mCurrentDual.get() != nullptr);
        assert(mCurrentDual->size() > static_cast<IndexType>(0));
        return (mCurrentDual.operator *());
    }
    void setCurrentDual(const IndexType & aIndex, const ElementType & aValue)
    {
        assert(mCurrentDual.get() != nullptr);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mCurrentDual->size());
        mCurrentDual->operator [](aIndex) = aValue;
    }
    const locus::Vector<ElementType, IndexType> & getCurrentInequalityValues() const
    {
        assert(mCurrentInequalityValues.get() != nullptr);
        assert(mCurrentInequalityValues->size() > static_cast<IndexType>(0));
        return (mCurrentInequalityValues.operator *());
    }
    const ElementType & getCurrentInequalityValues(const IndexType & aIndex) const
    {
        assert(mCurrentInequalityValues.get() != nullptr);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mCurrentInequalityValues->size());
        return(mCurrentInequalityValues->operator [](aIndex));
    }
    void setCurrentInequalityValue(const IndexType & aIndex, const ElementType & aValue)
    {
        assert(mCurrentInequalityValues.get() != nullptr);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mCurrentInequalityValues->size());
        mCurrentInequalityValues->operator [](aIndex) = aValue;
    }

    void setInitialGuess(const ElementType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mCurrentControl->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setInitialGuess(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
    }
    void setInitialGuess(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInitialGuess)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInitialGuess, 0.);
    }
    void setInitialGuess(const locus::MultiVector<ElementType, IndexType> & aInitialGuess)
    {
        assert(aInitialGuess.getNumVectors() == mCurrentControl->getNumVectors());

        const IndexType tNumVectors = aInitialGuess.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputInitialGuess = aInitialGuess[tIndex];
            locus::Vector<ElementType, IndexType> & tMyControl = mCurrentControl->operator [](tIndex);
            assert(tInputInitialGuess.size() == tMyControl.size());
            tMyControl.update(1., tInputInitialGuess, 0.);
        }
    }

    const locus::MultiVector<ElementType, IndexType> & getCurrentState() const
    {
        assert(mCurrentState.get() != nullptr);

        return (mCurrentState.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentState(const IndexType & aVectorIndex) const
    {
        assert(mCurrentState.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentState->getNumVectors());

        return (mCurrentState->operator [](aVectorIndex));
    }
    void setCurrentState(const locus::MultiVector<ElementType, IndexType> & aState)
    {
        assert(mCurrentState.get() != nullptr);
        assert(aState.getNumVectors() > static_cast<IndexType>(0));
        assert(aState.getNumVectors() == mCurrentState->getNumVectors());

        const IndexType tNumVectors = aState.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputState = aState[tIndex];
            locus::Vector<ElementType, IndexType> & tMyState = mCurrentState->operator [](tIndex);
            assert(tInputState.size() == tMyState.size());
            tMyState.update(1., tInputState, 0.);
        }
    }
    void setCurrentState(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aState)
    {
        assert(mCurrentState.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentState->getNumVectors());

        mCurrentState->operator [](aVectorIndex).update(1., aState, 0.);
    }

    const locus::MultiVector<ElementType, IndexType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);

        return (mCurrentControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentControl(const IndexType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        assert(aControl.getNumVectors() == mCurrentControl->getNumVectors());

        const IndexType tNumVectors = aControl.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputControl = aControl[tIndex];
            locus::Vector<ElementType, IndexType> & tMyControl = mCurrentControl->operator [](tIndex);
            assert(tInputControl.size() == tMyControl.size());
            tMyControl.update(1., tInputControl, 0.);
        }
    }
    void setCurrentControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aControl)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aControl, 0.);
    }

    const locus::MultiVector<ElementType, IndexType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);
        return (mPreviousControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getPreviousControl(const IndexType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        assert(aControl.getNumVectors() == mPreviousControl->getNumVectors());

        const IndexType tNumVectors = aControl.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputControl = aControl[tIndex];
            locus::Vector<ElementType, IndexType> & tMyControl = mPreviousControl->operator [](tIndex);
            assert(tInputControl.size() == tMyControl.size());
            tMyControl.update(1., tInputControl, 0.);
        }
    }
    void setPreviousControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aControl)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        mPreviousControl->operator [](aVectorIndex).update(1., aControl, 0.);
    }

    const locus::MultiVector<ElementType, IndexType> & getObjectiveGradient() const
    {
        assert(mObjectiveGradient.get() != nullptr);

        return (mObjectiveGradient.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getObjectiveGradient(const IndexType & aVectorIndex) const
    {
        assert(mObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mObjectiveGradient->getNumVectors());

        return (mObjectiveGradient->operator [](aVectorIndex));
    }
    void setObjectiveGradient(const locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        assert(aGradient.getNumVectors() == mObjectiveGradient->getNumVectors());

        const IndexType tNumVectors = aGradient.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputGradient = aGradient[tIndex];
            locus::Vector<ElementType, IndexType> & tMyObjectiveGradient = mObjectiveGradient->operator [](tIndex);
            assert(tInputGradient.size() == tMyObjectiveGradient.size());
            tMyObjectiveGradient.update(1., tInputGradient, 0.);
        }
    }
    void setObjectiveGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aGradient)
    {
        assert(mObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mObjectiveGradient->getNumVectors());

        mObjectiveGradient->operator [](aVectorIndex).update(1., aGradient, 0.);
    }

    const locus::MultiVector<ElementType, IndexType> & getInequalityGradient() const
    {
        assert(mInequalityGradient.get() != nullptr);

        return (mInequalityGradient.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getInequalityGradient(const IndexType & aVectorIndex) const
    {
        assert(mInequalityGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mInequalityGradient->getNumVectors());

        return (mInequalityGradient->operator [](aVectorIndex));
    }
    void setInequalityGradient(const locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        assert(aGradient.getNumVectors() == mInequalityGradient->getNumVectors());

        const IndexType tNumVectors = aGradient.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputGradient = aGradient[tIndex];
            locus::Vector<ElementType, IndexType> & tMyInequalityGradient = mInequalityGradient->operator [](tIndex);
            assert(tInputGradient.size() == tMyInequalityGradient.size());
            tMyInequalityGradient.update(1., tInputGradient, 0.);
        }
    }
    void setInequalityGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aGradient)
    {
        assert(mInequalityGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mInequalityGradient->getNumVectors());

        mInequalityGradient->operator [](aVectorIndex).update(1., aGradient, 0.);
    }

    const locus::MultiVector<ElementType, IndexType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);

        return (mControlLowerBounds.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getControlLowerBounds(const IndexType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        return (mControlLowerBounds->operator [](aVectorIndex));
    }
    void setControlLowerBounds(const ElementType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mControlLowerBounds->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aLowerBound)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).update(1., aLowerBound, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ElementType, IndexType> & aLowerBound)
    {
        assert(aLowerBound.getNumVectors() == mControlLowerBounds->getNumVectors());

        const IndexType tNumVectors = aLowerBound.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputLowerBound = aLowerBound[tIndex];
            locus::Vector<ElementType, IndexType> & tMyLowerBound = mControlLowerBounds->operator [](tIndex);
            assert(tInputLowerBound.size() == tMyLowerBound.size());
            tMyLowerBound.update(1., tInputLowerBound, 0.);
        }
    }

    const locus::MultiVector<ElementType, IndexType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);

        return (mControlUpperBounds.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getControlUpperBounds(const IndexType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        return (mControlUpperBounds->operator [](aVectorIndex));
    }
    void setControlUpperBounds(const ElementType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mControlUpperBounds->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aUpperBound)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aUpperBound, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ElementType, IndexType> & aUpperBound)
    {
        assert(aUpperBound.getNumVectors() == mControlUpperBounds->getNumVectors());

        const IndexType tNumVectors = aUpperBound.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInputUpperBound = aUpperBound[tIndex];
            locus::Vector<ElementType, IndexType> & tMyUpperBound = mControlUpperBounds->operator [](tIndex);
            assert(tInputUpperBound.size() == tMyUpperBound.size());
            tMyUpperBound.update(1., tInputUpperBound, 0.);
        }
    }

private:
    void initialize(const locus::DataFactory<ElementType, IndexType> & aFactory)
    {
        assert(aFactory.dual().getNumVectors() > static_cast<IndexType>(0));
        assert(aFactory.control().getNumVectors() > static_cast<IndexType>(0));

        const IndexType tVECTOR_INDEX = 0;
        mCurrentDual = aFactory.dual(tVECTOR_INDEX).create();
        mDualWorkVector = aFactory.dual(tVECTOR_INDEX).create();
        mControlWorkVector = aFactory.control(tVECTOR_INDEX).create();
        mCurrentInequalityValues = aFactory.dual(tVECTOR_INDEX).create();

        mDualReductionOperations = aFactory.getDualReductionOperations().create();
        mControlReductionOperations = aFactory.getControlReductionOperations().create();
    }

private:
    ElementType mStagnationMeasure;
    ElementType mMaxInequalityValue;
    ElementType mNormObjectiveGradient;
    ElementType mCurrentObjectiveValue;
    ElementType mPreviousObjectiveValue;

    std::shared_ptr<locus::Vector<ElementType, IndexType>> mCurrentDual;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mDualWorkVector;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkVector;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mCurrentInequalityValues;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentState;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInequalityGradient;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlUpperBounds;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mControlReductionOperations;

private:
    OptimalityCriteriaDataMng(const locus::OptimalityCriteriaDataMng<ElementType, IndexType>&);
    locus::OptimalityCriteriaDataMng<ElementType, IndexType> & operator=(const locus::OptimalityCriteriaDataMng<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class Simulation
{
public:
    virtual ~Simulation()
    {
    }

    /*!
     * Solve partial differential equation simulation
     *  Parameters:
     *    \param In
     *          aControl: control variables
     *    \param Out
     *          aState: state variables
     */
    virtual void solve(const locus::Vector<ElementType, IndexType> & aControl,
                       locus::Vector<ElementType, IndexType> & aState) = 0;
};

template<typename ElementType, typename IndexType = size_t>
class Criterion
{
public:
    virtual ~Criterion()
    {
    }

    /*!
     * Evaluates criterion of type f(\mathbf{u}(\mathbf{z}),\mathbf{z})\colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}
     * \rightarrow\mathbb{R}, where u denotes the state and z denotes the control variables. This criterion
     * is typically associated with nonlinear programming optimization problems. For instance, PDE constrasize_t
     * optimization problems.
     *  Parameters:
     *    \param In
     *          aState: state variables
     *    \param In
     *          aControl: control variables
     *
     *  \return Objective function value
     **/
    virtual ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                              const locus::MultiVector<ElementType, IndexType> & aControl) = 0;
    /*!
     * Computes the gradient of a criterion of type f(\mathbf{u}(\mathbf{z}),\mathbf{z})\colon\mathbb{R}^{n_u}
     * \times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, where u denotes the state and z denotes the control variables.
     * This criterion is typically associated with nonlinear programming optimization problems. For instance, PDE
     * constraint optimization problems.
     *  Parameters:
     *    \param In
     *          aState: state variables
     *    \param In
     *          aControl: control variables
     *    \param Out
     *          aOutput: gradient
     **/
    virtual void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                          const locus::MultiVector<ElementType, IndexType> & aControl,
                          locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    /*!
     * Computes the application of a vector to the Hessian of a criterion of type f(\mathbf{u}(\mathbf{z}),\mathbf{z})
     * \colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, where u denotes the state and z denotes the
     * control variables. This criterion is typically associated with nonlinear programming optimization problems.
     * For instance, PDE constraint optimization problems.
     *  Parameters:
     *    \param In
     *          aState:   state variables
     *    \param In
     *          aControl: control variables
     *    \param In
     *          aVector:  direction vector
     *    \param Out
     *          aOutput:  Hessian times direction vector
     **/
    virtual void hessian(const locus::MultiVector<ElementType, IndexType> & aState,
                         const locus::MultiVector<ElementType, IndexType> & aControl,
                         const locus::MultiVector<ElementType, IndexType> & aVector,
                         locus::MultiVector<ElementType, IndexType> & aOutput)
    {
    }
    //! Creates an object of type locus::Criterion
    virtual std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const = 0;
};

template<typename ElementType, typename IndexType = size_t>
class CriterionList
{
public:
    CriterionList() :
            mList()
    {
    }
    ~CriterionList()
    {
    }

    IndexType size() const
    {
        return (mList.size());
    }
    void add(const locus::Criterion<ElementType, IndexType> & aCriterion)
    {
        mList.push_back(aCriterion.create());
    }
    void add(const std::shared_ptr<locus::Criterion<ElementType, IndexType>> & aCriterion)
    {
        mList.push_back(aCriterion);
    }
    locus::Criterion<ElementType, IndexType> & operator [](const IndexType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::Criterion<ElementType, IndexType> & operator [](const IndexType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::CriterionList<ElementType, IndexType>> create() const
    {
        assert(this->size() > static_cast<IndexType>(0));
        std::shared_ptr<locus::CriterionList<ElementType, IndexType>> tOutput =
                std::make_shared<locus::CriterionList<ElementType, IndexType>>();
        const IndexType tNumCriterion = this->size();
        for(IndexType tIndex = 0; tIndex < tNumCriterion; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);
            const std::shared_ptr<locus::Criterion<ElementType, IndexType>> & tCriterion = mList[tIndex];
            tOutput->add(tCriterion);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::Criterion<ElementType, IndexType>> & ptr(const IndexType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::Criterion<ElementType, IndexType>>> mList;

private:
    CriterionList(const locus::CriterionList<ElementType, IndexType>&);
    locus::CriterionList<ElementType, IndexType> & operator=(const locus::CriterionList<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaObjectiveTestTwo : public locus::Criterion<ElementType, IndexType>
{
public:
    OptimalityCriteriaObjectiveTestTwo()
    {
    }
    virtual ~OptimalityCriteriaObjectiveTestTwo()
    {
    }

    ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                      const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        const IndexType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<IndexType>(2));
        ElementType tOutput = aControl(tVectorIndex, 0) + (static_cast<ElementType>(2) * aControl(tVectorIndex, 1));
        return (tOutput);
    }

    void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                  const locus::MultiVector<ElementType, IndexType> & aControl,
                  locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        const IndexType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<IndexType>(2));
        aGradient(tVectorIndex, 0) = static_cast<ElementType>(1);
        aGradient(tVectorIndex, 1) = static_cast<ElementType>(2);
    }
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::Criterion<ElementType, IndexType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaObjectiveTestTwo<ElementType, IndexType>>();
        return (tOutput);
    }

private:
    OptimalityCriteriaObjectiveTestTwo(const locus::OptimalityCriteriaObjectiveTestTwo<ElementType, IndexType>&);
    locus::OptimalityCriteriaObjectiveTestTwo<ElementType, IndexType> & operator=(const locus::OptimalityCriteriaObjectiveTestTwo<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaObjectiveTestOne : public locus::Criterion<ElementType, IndexType>
{
public:
    OptimalityCriteriaObjectiveTestOne() :
            mConstant(0.0624),
            mReductionOperations(std::make_shared<locus::StandardVectorReductionOperations<ElementType,IndexType>>())
    {

    }
    explicit OptimalityCriteriaObjectiveTestOne(const locus::ReductionOperations<ElementType, IndexType> & aInterface) :
            mConstant(0.0624),
            mReductionOperations(aInterface.create())
    {
    }
    virtual ~OptimalityCriteriaObjectiveTestOne()
    {
    }

    ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                      const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        const IndexType tVectorIndex = 0;
        ElementType tSum = mReductionOperations->sum(aControl[tVectorIndex]);
        ElementType tOutput = mConstant * tSum;
        return (tOutput);
    }

    void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                  const locus::MultiVector<ElementType, IndexType> & aControl,
                  locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        const IndexType tVectorIndex = 0;
        aGradient[tVectorIndex].fill(mConstant);
    }
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::Criterion<ElementType, IndexType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaObjectiveTestOne<ElementType, IndexType>>(*mReductionOperations);
        return (tOutput);
    }

private:
    ElementType mConstant;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mReductionOperations;

private:
    OptimalityCriteriaObjectiveTestOne(const locus::OptimalityCriteriaObjectiveTestOne<ElementType, IndexType>&);
    locus::OptimalityCriteriaObjectiveTestOne<ElementType, IndexType> & operator=(const locus::OptimalityCriteriaObjectiveTestOne<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaInequalityTestOne : public locus::Criterion<ElementType, IndexType>
{
public:
    explicit OptimalityCriteriaInequalityTestOne() :
            mBound(1.)
    {
    }
    virtual ~OptimalityCriteriaInequalityTestOne()
    {
    }

    ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                      const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        const IndexType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<IndexType>(5));

        const ElementType tPower = 3.;
        ElementType tFirstTerm = static_cast<ElementType>(61.) / std::pow(aControl(tVectorIndex,0), tPower);
        ElementType tSecondTerm = static_cast<ElementType>(37.) / std::pow(aControl(tVectorIndex,1), tPower);
        ElementType tThirdTerm = static_cast<ElementType>(19.) / std::pow(aControl(tVectorIndex,2), tPower);
        ElementType tFourthTerm = static_cast<ElementType>(7.) / std::pow(aControl(tVectorIndex,3), tPower);
        ElementType tFifthTerm = static_cast<ElementType>(1.) / std::pow(aControl(tVectorIndex,4), tPower);

        ElementType tValue = tFirstTerm + tSecondTerm + tThirdTerm + tFourthTerm + tFifthTerm;
        ElementType tOutput = tValue - mBound;

        return (tOutput);
    }

    void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                  const locus::MultiVector<ElementType, IndexType> & aControl,
                  locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        const IndexType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<IndexType>(5));

        const ElementType tPower = 4;
        const ElementType tScaleFactor = -3.;
        aGradient(tVectorIndex,0) = tScaleFactor * (static_cast<ElementType>(61.) / std::pow(aControl(tVectorIndex,0), tPower));
        aGradient(tVectorIndex,1) = tScaleFactor * (static_cast<ElementType>(37.) / std::pow(aControl(tVectorIndex,1), tPower));
        aGradient(tVectorIndex,2) = tScaleFactor * (static_cast<ElementType>(19.) / std::pow(aControl(tVectorIndex,2), tPower));
        aGradient(tVectorIndex,3) = tScaleFactor * (static_cast<ElementType>(7.) / std::pow(aControl(tVectorIndex,3), tPower));
        aGradient(tVectorIndex,4) = tScaleFactor * (static_cast<ElementType>(1.) / std::pow(aControl(tVectorIndex,4), tPower));
    }
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::Criterion<ElementType, IndexType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaInequalityTestOne<ElementType, IndexType>>();
        return (tOutput);
    }

private:
    ElementType mBound;

private:
    OptimalityCriteriaInequalityTestOne(const locus::OptimalityCriteriaInequalityTestOne<ElementType, IndexType>&);
    locus::OptimalityCriteriaInequalityTestOne<ElementType, IndexType> & operator=(const locus::OptimalityCriteriaInequalityTestOne<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaInequalityTestTwo : public locus::Criterion<ElementType, IndexType>
{
public:
    explicit OptimalityCriteriaInequalityTestTwo()
    {
    }
    virtual ~OptimalityCriteriaInequalityTestTwo()
    {
    }

    ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                      const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        const IndexType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<IndexType>(2));

        ElementType tDenominator = aControl(tVectorIndex, 1) + (static_cast<ElementType>(0.25) * aControl(tVectorIndex, 0));
        ElementType tOutput = static_cast<ElementType>(1) - (static_cast<ElementType>(1.5) / tDenominator);

        return (tOutput);
    }

    void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                  const locus::MultiVector<ElementType, IndexType> & aControl,
                  locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        const IndexType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<IndexType>(2));

        ElementType tPower = 2;
        ElementType tDenominator = aControl(tVectorIndex, 1) + (static_cast<ElementType>(0.25) * aControl(tVectorIndex, 0));
        tDenominator = std::pow(tDenominator, tPower);
        ElementType tFirstElement = static_cast<ElementType>(0.375) / tDenominator;
        aGradient(tVectorIndex, 0) = tFirstElement;
        ElementType tSecondElement = static_cast<ElementType>(1.5) / tDenominator;
        aGradient(tVectorIndex, 1) = tSecondElement;
    }
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::Criterion<ElementType, IndexType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaInequalityTestTwo<ElementType, IndexType>>();
        return (tOutput);
    }

private:
    OptimalityCriteriaInequalityTestTwo(const locus::OptimalityCriteriaInequalityTestTwo<ElementType, IndexType>&);
    locus::OptimalityCriteriaInequalityTestTwo<ElementType, IndexType> & operator=(const locus::OptimalityCriteriaInequalityTestTwo<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaStageMng
{
public:
    virtual ~OptimalityCriteriaStageMng()
    {
    }

    virtual void updateStage(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng) = 0;
    virtual void evaluateInequality(const IndexType & aConstraintIndex,
                                    locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng) = 0;
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaStageMngTypeLP : public locus::OptimalityCriteriaStageMng<ElementType, IndexType>
{
public:
    OptimalityCriteriaStageMngTypeLP(const locus::Criterion<ElementType, IndexType> & aObjective,
                                     const locus::Criterion<ElementType, IndexType> & aInequality) :
            mObjective(aObjective.create()),
            mConstraint(aInequality.create())
    {
    }
    virtual ~OptimalityCriteriaStageMngTypeLP()
    {
    }

    void updateStage(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng)
    {
        const locus::MultiVector<ElementType, IndexType> & tState = aDataMng.getCurrentState();
        const locus::MultiVector<ElementType, IndexType> & tControl = aDataMng.getCurrentControl();
        ElementType tObjectiveValue = mObjective->value(tState, tControl);
        aDataMng.setCurrentObjectiveValue(tObjectiveValue);

        std::shared_ptr<locus::MultiVector<ElementType, IndexType>> tObjectiveGradient =
                aDataMng.getObjectiveGradient().create();
        const IndexType tNumVectors = tObjectiveGradient->getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            tObjectiveGradient->operator[](tIndex).fill(0.);
        }
        mObjective->gradient(tState, tControl, *tObjectiveGradient);
        aDataMng.setObjectiveGradient(*tObjectiveGradient);

        this->computeInequalityGradient(aDataMng);
    }
    void evaluateInequality(const IndexType & aConstraintIndex, locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng)
    {
        const locus::MultiVector<ElementType, IndexType> & tState = aDataMng.getCurrentState();
        const locus::MultiVector<ElementType, IndexType> & tControl = aDataMng.getCurrentControl();

        ElementType tInequalityValue = mConstraint->value(tState, tControl);
        aDataMng.setCurrentInequalityValue(aConstraintIndex, tInequalityValue);
    }
    void computeInequalityGradient(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng)
    {
        const locus::MultiVector<ElementType, IndexType> & tState = aDataMng.getCurrentState();
        const locus::MultiVector<ElementType, IndexType> & tControl = aDataMng.getCurrentControl();
        std::shared_ptr<locus::MultiVector<ElementType, IndexType>> tInequalityGradient =
                aDataMng.getInequalityGradient().create();

        mConstraint->gradient(tState, tControl, *tInequalityGradient);

        aDataMng.setInequalityGradient(*tInequalityGradient);
    }

private:
    std::shared_ptr<locus::Criterion<ElementType,IndexType>> mObjective;
    std::shared_ptr<locus::Criterion<ElementType,IndexType>> mConstraint;

private:
    OptimalityCriteriaStageMngTypeLP(const locus::OptimalityCriteriaStageMngTypeLP<ElementType, IndexType>&);
    locus::OptimalityCriteriaStageMngTypeLP<ElementType, IndexType> & operator=(const locus::OptimalityCriteriaStageMngTypeLP<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteriaSubProblem
{
public:
    virtual ~OptimalityCriteriaSubProblem(){}

    virtual void solve(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng,
                       locus::OptimalityCriteriaStageMng<ElementType, IndexType> & aStageMng) = 0;
};

template<typename ElementType, typename IndexType = size_t>
class SingleConstraintTypeLP : public locus::OptimalityCriteriaSubProblem<ElementType,IndexType>
{
public:
    explicit SingleConstraintTypeLP(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng) :
            mActiveSet(aDataMng.getCurrentControl().create()),
            mPassiveSet(aDataMng.getCurrentControl().create()),
            mWorkControl(aDataMng.getCurrentControl().create())
    {
        this->initialize();
    }
    virtual ~SingleConstraintTypeLP()
    {
    }

    void solve(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng,
               locus::OptimalityCriteriaStageMng<ElementType, IndexType> & aStageMng)
    {
        assert(aDataMng.getNumConstraints() == static_cast<IndexType>(1));

        const IndexType tConstraintIndex = 0;
        ElementType tDual = this->computeDual(aDataMng);
        aDataMng.setCurrentDual(tConstraintIndex, tDual);

        this->updateControl(tDual, aDataMng);

        aStageMng.evaluateInequality(tConstraintIndex, aDataMng);
    }

private:
    void initialize()
    {
        const IndexType tVectorIndex = 0;
        mActiveSet->operator [](tVectorIndex).fill(1);
        mPassiveSet->operator [](tVectorIndex).fill(0);
    }
    ElementType computeDual(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng)
    {

        // Compute linearized constraint, c_0 = g(\bm{x}) + (\frac{\partial{g}}{\partial{x}})^{T}\bm{x}
        const IndexType tVectorIndex = 0;
        const locus::Vector<ElementType, IndexType> & tInequalityGradient = aDataMng.getInequalityGradient(tVectorIndex);
        const locus::Vector<ElementType, IndexType> & tPreviousControl = aDataMng.getPreviousControl(tVectorIndex);
        ElementType tLinearizedConstraint = tInequalityGradient.dot(tPreviousControl);
        const IndexType tConstraintIndex = 0;
        tLinearizedConstraint += aDataMng.getCurrentInequalityValues(tConstraintIndex);

        /* Compute c_0^{\ast} = c_0 + \sum_{i\in{I}_p}\frac{\partial{g}}{\partial{y}_i}\frac{1}{x_i}, where I_p is the passive set
           and \frac{\partial{g}}{\partial{y}_i} = -x_i^2\frac{\partial{g}}{\partial{x}_i}\ \forall\ i=1\,dots,length(\bm{x})*/
        const locus::Vector<ElementType, IndexType> & tPassiveSet = mPassiveSet->operator [](tVectorIndex);
        mWorkControl->operator [](tVectorIndex).update(1., tInequalityGradient, 0.);
        mWorkControl->operator [](tVectorIndex).entryWiseProduct(tPassiveSet);
        ElementType tLinearizedConstraintStar = -(mWorkControl->operator [](tVectorIndex).dot(tPreviousControl));
        tLinearizedConstraintStar += tLinearizedConstraint;

        // Compute Active Inequality Constraint Gradient
        const locus::Vector<ElementType, IndexType> & tActiveSet = mActiveSet->operator [](tVectorIndex);
        mWorkControl->operator [](tVectorIndex).update(1., tInequalityGradient, 0.);
        mWorkControl->operator [](tVectorIndex).entryWiseProduct(tActiveSet);

        /* Compute Dual, \lambda=\left[\frac{1}{c_0^{ast}}\sum_{i\in{I}_a}\left(-\frac{\partial{f}}{\partial{x_i}}
           \frac{\partial{g}}{\partial{y_i}}\right)^{1/2}\right]^2, where y_i=1/x_i */
        ElementType tSum = 0;
        IndexType tNumControls = tPreviousControl.size();
        const locus::Vector<ElementType, IndexType> & tActiveInqGradient = mWorkControl->operator [](tVectorIndex);
        const locus::Vector<ElementType, IndexType> & tObjectiveGradient = aDataMng.getObjectiveGradient(tVectorIndex);
        for(IndexType tIndex = 0; tIndex < tNumControls; tIndex++)
        {
            ElementType tValue = std::pow(tPreviousControl[tIndex], static_cast<ElementType>(2)) * tObjectiveGradient[tIndex] * tActiveInqGradient[tIndex];
            tValue = std::sqrt(tValue);
            tSum += tValue;
        }
        ElementType tDual = (static_cast<ElementType>(1) / tLinearizedConstraintStar) * tSum;
        tDual = std::pow(tDual, static_cast<ElementType>(2));

        return (tDual);
    }
    void updateControl(const ElementType & aDual, locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng)
    {
        IndexType tNumControlVectors = aDataMng.getNumControlVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tPreviousControl = aDataMng.getPreviousControl(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tControlLowerBound = aDataMng.getControlLowerBounds(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tControlUpperBound = aDataMng.getControlUpperBounds(tVectorIndex);

            const locus::Vector<ElementType, IndexType> & tObjectiveGradient = aDataMng.getObjectiveGradient(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tInequalityGradient = aDataMng.getInequalityGradient(tVectorIndex);

            mActiveSet->operator [](tVectorIndex).fill(1);
            mPassiveSet->operator [](tVectorIndex).fill(0);

            ElementType tDampingPower = 0.5;
            IndexType tNumControls = tPreviousControl.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ElementType tTrialControl = (-aDual * tInequalityGradient[tControlIndex])
                        / (tObjectiveGradient[tControlIndex]);
                tTrialControl = -std::pow(tPreviousControl[tControlIndex], static_cast<ElementType>(2)) * tTrialControl;
                tTrialControl = std::pow(tTrialControl, tDampingPower);
                bool tOutsideBounds = (tControlLowerBound[tControlIndex] >= tTrialControl) || (tControlUpperBound[tControlIndex] <= tTrialControl);
                if(tOutsideBounds == true)
                {
                    mActiveSet->operator ()(tVectorIndex, tControlIndex) = 0;
                    mPassiveSet->operator ()(tVectorIndex, tControlIndex) = 1;
                }
                tTrialControl = tControlLowerBound[tControlIndex] >= tTrialControl ? tControlLowerBound[tControlIndex] : tTrialControl;
                tTrialControl = tControlUpperBound[tControlIndex] <= tTrialControl ? tControlUpperBound[tControlIndex] : tTrialControl;
                mWorkControl->operator ()(tVectorIndex, tControlIndex) = tTrialControl;
            }
            aDataMng.setCurrentControl(tVectorIndex, mWorkControl->operator [](tVectorIndex));
        }
    }

private:
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPassiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mWorkControl;

private:
    SingleConstraintTypeLP(const locus::SingleConstraintTypeLP<ElementType, IndexType>&);
    locus::SingleConstraintTypeLP<ElementType, IndexType> & operator=(const locus::SingleConstraintTypeLP<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class SynthesisOptimizationSubProblem : public locus::OptimalityCriteriaSubProblem<ElementType,IndexType>
{
public:
    explicit SynthesisOptimizationSubProblem(const locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng) :
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

    ElementType getMoveLimit() const
    {
        return (mMoveLimit);
    }
    ElementType getDampingPower() const
    {
        return (mDampingPower);
    }
    ElementType getDualLowerBound() const
    {
        return (mDualLowerBound);
    }
    ElementType getDualUpperBound() const
    {
        return (mDualUpperBound);
    }
    ElementType getBisectionTolerance() const
    {
        return (mBisectionTolerance);
    }

    void setMoveLimit(const ElementType & aInput)
    {
        mMoveLimit = aInput;
    }
    void setDampingPower(const ElementType & aInput)
    {
        mDampingPower = aInput;
    }
    void setDualLowerBound(const ElementType & aInput)
    {
        mDualLowerBound = aInput;
    }
    void setDualUpperBound(const ElementType & aInput)
    {
        mDualUpperBound = aInput;
    }
    void setBisectionTolerance(const ElementType & aInput)
    {
        mBisectionTolerance = aInput;
    }

    void solve(locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng,
               locus::OptimalityCriteriaStageMng<ElementType, IndexType> & aStageMng)
    {
        ElementType tDualLowerBound = this->getDualLowerBound();
        ElementType tDualUpperBound = this->getDualUpperBound();
        ElementType tBisectionTolerance = this->getBisectionTolerance();

        IndexType tNumConstraints = aDataMng.getNumConstraints();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            ElementType tDualMisfit = tDualUpperBound - tDualLowerBound;
            ElementType tTrialDual = std::numeric_limits<ElementType>::max();
            while(tDualMisfit >= tBisectionTolerance)
            {
                tTrialDual = static_cast<ElementType>(0.5) * (tDualUpperBound + tDualLowerBound);
                this->updateControl(tTrialDual, aDataMng);

                aStageMng.evaluateInequality(tConstraintIndex, aDataMng);
                const locus::Vector<ElementType, IndexType> & tInequalityValues = aDataMng.getCurrentInequalityValues();
                ElementType mFirstOrderTaylorApproximation = tInequalityValues[tConstraintIndex] + mInequalityGradientDotDeltaControl;
                if(mFirstOrderTaylorApproximation > static_cast<ElementType>(0.))
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
    void updateControl(const ElementType & aTrialDual, locus::OptimalityCriteriaDataMng<ElementType, IndexType> & aDataMng)
    {
        mInequalityGradientDotDeltaControl = 0;
        ElementType tMoveLimit = this->getMoveLimit();
        ElementType tDampingPower = this->getDampingPower();

        IndexType tNumControlVectors = aDataMng.getNumControlVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tPreviousControl = aDataMng.getPreviousControl(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tControlLowerBound = aDataMng.getControlLowerBounds(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tControlUpperBound = aDataMng.getControlUpperBounds(tVectorIndex);

            const locus::Vector<ElementType, IndexType> & tObjectiveGradient = aDataMng.getObjectiveGradient(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tInequalityGradient = aDataMng.getInequalityGradient(tVectorIndex);

            IndexType tNumControls = tPreviousControl.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ElementType tTrialControl = -tObjectiveGradient[tControlIndex]
                        / (aTrialDual * tInequalityGradient[tControlIndex]);
                ElementType tFabsValue = std::abs(tTrialControl);
                ElementType tSignValue = copysign(1.0, tTrialControl);
                tTrialControl = tPreviousControl[tControlIndex] * tSignValue * std::pow(tFabsValue, tDampingPower);
                ElementType tNewControl = tPreviousControl[tControlIndex] + tMoveLimit;
                tTrialControl = std::min(tNewControl, tTrialControl);
                tTrialControl = std::min(tControlUpperBound[tControlIndex], tTrialControl);
                tNewControl = tPreviousControl[tControlIndex] - tMoveLimit;
                tTrialControl = std::max(tNewControl, tTrialControl);
                mWorkControl->operator ()(tVectorIndex, tControlIndex) =
                        std::max(tControlLowerBound[tControlIndex], tTrialControl);
            }
            aDataMng.setCurrentControl(tVectorIndex, mWorkControl->operator [](tVectorIndex));
            mWorkControl->operator [](tVectorIndex).update(-1., tPreviousControl, 1.); /*Compute Delta Control*/
            mInequalityGradientDotDeltaControl += tInequalityGradient.dot(mWorkControl->operator [](tVectorIndex));
        }
    }

private:
    ElementType mMoveLimit;
    ElementType mDampingPower;
    ElementType mDualLowerBound;
    ElementType mDualUpperBound;
    ElementType mBisectionTolerance;
    ElementType mInequalityGradientDotDeltaControl;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mWorkControl;

private:
    SynthesisOptimizationSubProblem(const locus::SynthesisOptimizationSubProblem<ElementType, IndexType>&);
    locus::SynthesisOptimizationSubProblem<ElementType, IndexType> & operator=(const locus::SynthesisOptimizationSubProblem<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class OptimalityCriteria
{
public:
    explicit OptimalityCriteria(const std::shared_ptr<locus::OptimalityCriteriaDataMng<ElementType, IndexType>> & aDataMng,
                                const std::shared_ptr<locus::OptimalityCriteriaStageMng<ElementType, IndexType>> & aStageMng,
                                const std::shared_ptr<locus::OptimalityCriteriaSubProblem<ElementType, IndexType>> & aSubProblem) :
            mPrintDiagnostics(false),
            mOutputStream(),
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mStagnationTolerance(1e-2),
            mFeasibilityTolerance(1e-5),
            mObjectiveGradientTolerance(1e-8),
            mDataMng(aDataMng),
            mStageMng(aStageMng),
            mSubProblem(aSubProblem)
    {
    }
    ~OptimalityCriteria()
    {
    }

    bool printDiagnostics() const
    {
        return (mPrintDiagnostics);
    }

    void enableDiagnostics()
    {
        mPrintDiagnostics = true;
    }

    IndexType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    IndexType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    ElementType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    ElementType getFeasibilityTolerance() const
    {
        return (mFeasibilityTolerance);
    }
    ElementType getObjectiveGradientTolerance() const
    {
        return (mObjectiveGradientTolerance);
    }

    void setMaxNumIterations(const IndexType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStagnationTolerance(const ElementType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    void setFeasibilityTolerance(const ElementType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }
    void setObjectiveGradientTolerance(const ElementType & aInput)
    {
        mObjectiveGradientTolerance = aInput;
    }

    void gatherOuputStream(std::ostringstream & aOutput)
    {
        aOutput << mOutputStream.str().c_str();
    }

    void solve()
    {
        for(IndexType tIndex = 0; tIndex < mDataMng->getNumConstraints(); tIndex++)
        {
            mStageMng->evaluateInequality(tIndex, *mDataMng);
        }

        while(1)
        {
            mStageMng->updateStage(*mDataMng);

            mDataMng->computeStagnationMeasure();
            mDataMng->computeMaxInequalityValue();
            mDataMng->computeNormObjectiveGradient();

            this->printCurrentProgress();
            if(this->stoppingCriteriaSatisfied() == true)
            {
                break;
            }

            this->storeCurrentStageData();
            mSubProblem->solve(*mDataMng, *mStageMng);

            mNumIterationsDone++;
        }
    }

    bool stoppingCriteriaSatisfied()
    {
        bool tStoppingCriterionSatisfied = false;
        ElementType tStagnationMeasure = mDataMng->getStagnationMeasure();
        ElementType tMaxInequalityValue = mDataMng->getMaxInequalityValue();
        ElementType tNormObjectiveGradient = mDataMng->getNormObjectiveGradient();

        if(this->getNumIterationsDone() >= this->getMaxNumIterations())
        {
            tStoppingCriterionSatisfied = true;
        }
        else if(tStagnationMeasure < this->getStagnationTolerance())
        {
            tStoppingCriterionSatisfied = true;
        }
        else if(tNormObjectiveGradient < this->getObjectiveGradientTolerance()
                && tMaxInequalityValue < this->getFeasibilityTolerance())
        {
            tStoppingCriterionSatisfied = true;
        }

        return (tStoppingCriterionSatisfied);
    }

    void printCurrentProgress()
    {
        if(this->printDiagnostics() == false)
        {
            return;
        }

        IndexType tCurrentNumIterationsDone = this->getNumIterationsDone();

        if(tCurrentNumIterationsDone < 2)
        {
            mOutputStream << " Itr" << std::setw(14) << "   F(x)  " << std::setw(16) << " ||F'(x)||" << std::setw(16)
                    << "   Max(H) " << "\n" << std::flush;
            mOutputStream << "-----" << std::setw(14) << "----------" << std::setw(16) << "-----------" << std::setw(16)
                    << "----------" << "\n" << std::flush;
        }

        ElementType tObjectiveValue = mDataMng->getCurrentObjectiveValue();
        ElementType tMaxInequalityValue = mDataMng->getMaxInequalityValue();
        ElementType tNormObjectiveGradient = mDataMng->getNormObjectiveGradient();
        mOutputStream << std::setw(3) << tCurrentNumIterationsDone << std::setprecision(4) << std::fixed
                << std::scientific << std::setw(16) << tObjectiveValue << std::setw(16) << tNormObjectiveGradient
                << std::setw(16) << tMaxInequalityValue << "\n";
    }
    void storeCurrentStageData()
    {
        const ElementType tObjectiveValue = mDataMng->getCurrentObjectiveValue();
        mDataMng->setPreviousObjectiveValue(tObjectiveValue);

        IndexType tNumVectors = mDataMng->getNumControlVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tControl = mDataMng->getCurrentControl(tVectorIndex);
            mDataMng->setPreviousControl(tVectorIndex, tControl);
        }
    }

private:
    bool mPrintDiagnostics;
    std::ostringstream mOutputStream;

    IndexType mMaxNumIterations;
    IndexType mNumIterationsDone;

    ElementType mStagnationTolerance;
    ElementType mFeasibilityTolerance;
    ElementType mObjectiveGradientTolerance;

    std::shared_ptr<locus::OptimalityCriteriaDataMng<ElementType, IndexType>> mDataMng;
    std::shared_ptr<locus::OptimalityCriteriaStageMng<ElementType, IndexType>> mStageMng;
    std::shared_ptr<locus::OptimalityCriteriaSubProblem<ElementType, IndexType>> mSubProblem;

private:
    OptimalityCriteria(const locus::OptimalityCriteria<ElementType, IndexType>&);
    locus::OptimalityCriteria<ElementType, IndexType> & operator=(const locus::OptimalityCriteria<ElementType, IndexType>&);
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
        GRADIENT_TOLERANCE = 3,
        TRIAL_STEP_TOLERANCE = 4,
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

template<typename ElementType, typename IndexType = size_t>
void checkBounds(const locus::MultiVector<ElementType, IndexType> & aLowerBounds,
                 const locus::MultiVector<ElementType, IndexType> & aUpperBounds,
                 bool aPrintMessage = false)
{
    assert(aLowerBounds.getNumVectors() == aUpperBounds.getNumVectors());

    try
    {
        IndexType tNumVectors = aLowerBounds.getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            assert(aLowerBounds[tVectorIndex].size() == aUpperBounds[tVectorIndex].size());

            IndexType tNumElements = aLowerBounds[tVectorIndex].size();
            for(IndexType tElemIndex = 0; tElemIndex < tNumElements; tElemIndex++)
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

template<typename ElementType, typename IndexType = size_t>
void project(const locus::MultiVector<ElementType, IndexType> & aLowerBound,
             const locus::MultiVector<ElementType, IndexType> & aUpperBound,
             locus::MultiVector<ElementType, IndexType> & aInput)
{
    assert(aInput.getNumVectors() == aUpperBound.getNumVectors());
    assert(aLowerBound.getNumVectors() == aUpperBound.getNumVectors());

    IndexType tNumVectors = aInput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        locus::Vector<ElementType, IndexType> & tVector = aInput[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tLowerBound = aLowerBound[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tUpperBound = aUpperBound[tVectorIndex];

        assert(tVector.size() == tLowerBound.size());
        assert(tUpperBound.size() == tLowerBound.size());

        IndexType tNumElements = tVector.size();
        for(IndexType tIndex = 0; tIndex < tNumElements; tIndex++)
        {
            tVector[tIndex] = std::max(tVector[tIndex], tLowerBound[tIndex]);
            tVector[tIndex] = std::min(tVector[tIndex], tUpperBound[tIndex]);
        }
    }
} // function project

template<typename ElementType, typename IndexType = size_t>
void computeActiveAndInactiveSets(const locus::MultiVector<ElementType, IndexType> & aInput,
                                  const locus::MultiVector<ElementType, IndexType> & aLowerBound,
                                  const locus::MultiVector<ElementType, IndexType> & aUpperBound,
                                  locus::MultiVector<ElementType, IndexType> & aActiveSet,
                                  locus::MultiVector<ElementType, IndexType> & aInactiveSet)
{
    assert(aInput.getNumVectors() == aLowerBound.getNumVectors());
    assert(aInput.getNumVectors() == aInactiveSet.getNumVectors());
    assert(aActiveSet.getNumVectors() == aInactiveSet.getNumVectors());
    assert(aLowerBound.getNumVectors() == aUpperBound.getNumVectors());

    IndexType tNumVectors = aInput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        locus::Vector<ElementType, IndexType> & tActiveSet = aActiveSet[tVectorIndex];
        locus::Vector<ElementType, IndexType> & tInactiveSet = aInactiveSet[tVectorIndex];

        const locus::Vector<ElementType, IndexType> & tVector = aInput[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tLowerBound = aLowerBound[tVectorIndex];
        const locus::Vector<ElementType, IndexType> & tUpperBound = aUpperBound[tVectorIndex];

        assert(tVector.size() == tLowerBound.size());
        assert(tVector.size() == tInactiveSet.size());
        assert(tActiveSet.size() == tInactiveSet.size());
        assert(tUpperBound.size() == tLowerBound.size());

        tActiveSet.fill(0.);
        tInactiveSet.fill(0.);

        IndexType tNumElements = tVector.size();
        for(IndexType tIndex = 0; tIndex < tNumElements; tIndex++)
        {
            tActiveSet[tIndex] = static_cast<IndexType>((tVector[tIndex] >= tUpperBound[tIndex])
                    || (tVector[tIndex] <= tLowerBound[tIndex]));
            tInactiveSet[tIndex] = static_cast<IndexType>((tVector[tIndex] < tUpperBound[tIndex])
                    && (tVector[tIndex] > tLowerBound[tIndex]));
        }
    }
} // function computeActiveAndInactiveSets

} // namespace bounds

template<typename ElementType, typename IndexType = size_t>
class AlgorithmDataMng
{
public:
    virtual ~AlgorithmDataMng()
    {
    }

    virtual IndexType getNumControlVectors() const = 0;

    // NOTE: OBJECTIVE FUNCTION VALUE
    virtual ElementType getCurrentObjectiveFunctionValue() const = 0;
    virtual void setCurrentObjectiveFunctionValue(const ElementType & aInput) = 0;
    virtual ElementType getPreviousObjectiveFunctionValue() const = 0;
    virtual void setPreviousObjectiveFunctionValue(const ElementType & aInput) = 0;

    // NOTE: SET INITIAL GUESS
    virtual void setInitialGuess(const ElementType & aValue) = 0;
    virtual void setInitialGuess(const IndexType & aVectorIndex, const ElementType & aValue) = 0;
    virtual void setInitialGuess(const locus::MultiVector<ElementType, IndexType> & aInitialGuess) = 0;
    virtual void setInitialGuess(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInitialGuess) = 0;

    // NOTE: TRIAL STEP
    virtual const locus::MultiVector<ElementType, IndexType> & getTrialStep() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getTrialStep(const IndexType & aVectorIndex) const = 0;
    virtual void setTrialStep(const locus::MultiVector<ElementType, IndexType> & aTrialStep) = 0;
    virtual void setTrialStep(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aTrialStep) = 0;

    // NOTE: CURRENT CONTROL
    virtual const locus::MultiVector<ElementType, IndexType> & getCurrentControl() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getCurrentControl(const IndexType & aVectorIndex) const = 0;
    virtual void setCurrentControl(const locus::MultiVector<ElementType, IndexType> & aControl) = 0;
    virtual void setCurrentControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aControl) = 0;

    // NOTE: PREVIOUS CONTROL
    virtual const locus::MultiVector<ElementType, IndexType> & getPreviousControl() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getPreviousControl(const IndexType & aVectorIndex) const = 0;
    virtual void setPreviousControl(const locus::MultiVector<ElementType, IndexType> & aControl) = 0;
    virtual void setPreviousControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aControl) = 0;

    // NOTE: CURRENT GRADIENT
    virtual const locus::MultiVector<ElementType, IndexType> & getCurrentGradient() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getCurrentGradient(const IndexType & aVectorIndex) const = 0;
    virtual void setCurrentGradient(const locus::MultiVector<ElementType, IndexType> & aGradient) = 0;
    virtual void setCurrentGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aGradient) = 0;

    // NOTE: PREVIOUS GRADIENT
    virtual const locus::MultiVector<ElementType, IndexType> & getPreviousGradient() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getPreviousGradient(const IndexType & aVectorIndex) const = 0;
    virtual void setPreviousGradient(const locus::MultiVector<ElementType, IndexType> & aGradient) = 0;
    virtual void setPreviousGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aGradient) = 0;

    // NOTE: SET CONTROL LOWER BOUNDS
    virtual const locus::MultiVector<ElementType, IndexType> & getControlLowerBounds() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getControlLowerBounds(const IndexType & aVectorIndex) const = 0;
    virtual void setControlLowerBounds(const ElementType & aValue) = 0;
    virtual void setControlLowerBounds(const IndexType & aVectorIndex, const ElementType & aValue) = 0;
    virtual void setControlLowerBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aLowerBound) = 0;
    virtual void setControlLowerBounds(const locus::MultiVector<ElementType, IndexType> & aLowerBound) = 0;

    // NOTE: SET CONTROL UPPER BOUNDS
    virtual const locus::MultiVector<ElementType, IndexType> & getControlUpperBounds() const = 0;
    virtual const locus::Vector<ElementType, IndexType> & getControlUpperBounds(const IndexType & aVectorIndex) const = 0;
    virtual void setControlUpperBounds(const ElementType & aValue) = 0;
    virtual void setControlUpperBounds(const IndexType & aVectorIndex, const ElementType & aValue) = 0;
    virtual void setControlUpperBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aUpperBound) = 0;
    virtual void setControlUpperBounds(const locus::MultiVector<ElementType, IndexType> & aUpperBound) = 0;
};

template<typename ElementType, typename IndexType = size_t>
class TrustRegionAlgorithmDataMng : public AlgorithmDataMng<ElementType, IndexType>
{
public:
    explicit TrustRegionAlgorithmDataMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            mNumDualVectors(aDataFactory.dual().getNumVectors()),
            mNumControlVectors(aDataFactory.control().getNumVectors()),
            mStagnationMeasure(0),
            mStationarityMeasure(0),
            mNormProjectedGradient(0),
            mGradientInexactnessTolerance(0),
            mObjectiveInexactnessTolerance(0),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
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
    IndexType getNumControlVectors() const
    {
        return (mNumControlVectors);
    }
    // NOTE: NUMBER OF DUAL VECTORS
    IndexType getNumDualVectors() const
    {
        return (mNumDualVectors);
    }

    // NOTE: OBJECTIVE FUNCTION VALUE
    ElementType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ElementType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    ElementType getPreviousObjectiveFunctionValue() const
    {
        return (mPreviousObjectiveFunctionValue);
    }
    void setPreviousObjectiveFunctionValue(const ElementType & aInput)
    {
        mPreviousObjectiveFunctionValue = aInput;
    }

    // NOTE: SET INITIAL GUESS
    void setInitialGuess(const ElementType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mCurrentControl->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }

    // NOTE: DUAL VECTOR
    const locus::MultiVector<ElementType, IndexType> & getDual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getDual(const IndexType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        return (mDual->operator [](aVectorIndex));
    }
    void setDual(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mDual->getNumVectors());
        locus::update(1., aInput, 0., *mDual);
    }
    void setDual(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        mDual->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: TRIAL STEP
    const locus::MultiVector<ElementType, IndexType> & getTrialStep() const
    {
        assert(mTrialStep.get() != nullptr);

        return (mTrialStep.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getTrialStep(const IndexType & aVectorIndex) const
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());

        return (mTrialStep->operator [](aVectorIndex));
    }
    void setTrialStep(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mTrialStep->getNumVectors());
        locus::update(1., aInput, 0., *mTrialStep);
    }
    void setTrialStep(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mTrialStep.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mTrialStep->getNumVectors());

        mTrialStep->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: ACTIVE SET FUNCTIONS
    const locus::MultiVector<ElementType, IndexType> & getActiveSet() const
    {
        assert(mActiveSet.get() != nullptr);

        return (mActiveSet.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getActiveSet(const IndexType & aVectorIndex) const
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        return (mActiveSet->operator [](aVectorIndex));
    }
    void setActiveSet(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mActiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mActiveSet);
    }
    void setActiveSet(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        mActiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: INACTIVE SET FUNCTIONS
    const locus::MultiVector<ElementType, IndexType> & getInactiveSet() const
    {
        assert(mInactiveSet.get() != nullptr);

        return (mInactiveSet.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getInactiveSet(const IndexType & aVectorIndex) const
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        return (mInactiveSet->operator [](aVectorIndex));
    }
    void setInactiveSet(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mInactiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mInactiveSet);
    }
    void setInactiveSet(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        mInactiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONTROL
    const locus::MultiVector<ElementType, IndexType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);

        return (mCurrentControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentControl(const IndexType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ElementType>(1), aInput, static_cast<ElementType>(0), *mCurrentControl);
    }
    void setCurrentControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS CONTROL
    const locus::MultiVector<ElementType, IndexType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);

        return (mPreviousControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getPreviousControl(const IndexType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousControl->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousControl);
    }
    void setPreviousControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        mPreviousControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT GRADIENT
    const locus::MultiVector<ElementType, IndexType> & getCurrentGradient() const
    {
        assert(mCurrentGradient.get() != nullptr);

        return (mCurrentGradient.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentGradient(const IndexType & aVectorIndex) const
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());

        return (mCurrentGradient->operator [](aVectorIndex));
    }
    void setCurrentGradient(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentGradient->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentGradient);
    }
    void setCurrentGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentGradient->getNumVectors());

        mCurrentGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS GRADIENT
    const locus::MultiVector<ElementType, IndexType> & getPreviousGradient() const
    {
        assert(mPreviousGradient.get() != nullptr);

        return (mPreviousGradient.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getPreviousGradient(const IndexType & aVectorIndex) const
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());

        return (mPreviousGradient->operator [](aVectorIndex));
    }
    void setPreviousGradient(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousGradient->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousGradient);
    }
    void setPreviousGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mPreviousGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousGradient->getNumVectors());

        mPreviousGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ElementType, IndexType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);

        return (mControlLowerBounds.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getControlLowerBounds(const IndexType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        return (mControlLowerBounds->operator [](aVectorIndex));
    }
    void setControlLowerBounds(const ElementType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mControlLowerBounds->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlLowerBounds);
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ElementType, IndexType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);

        return (mControlUpperBounds.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getControlUpperBounds(const IndexType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        return (mControlUpperBounds->operator [](aVectorIndex));
    }
    void setControlUpperBounds(const ElementType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mControlUpperBounds->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ElementType, IndexType> & aInput)
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
        IndexType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ElementType> storage(tNumVectors, std::numeric_limits<ElementType>::min());
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkVector->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ElementType, IndexType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkVector->update(-1., tMyPreviousControl, 1.);
            mControlWorkVector->modulus();
            storage[tIndex] = mControlReductionOperations->max(*mControlWorkVector);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ElementType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: NORM OF CURRENT PROJECTED GRADIENT
    ElementType computeProjectedVectorNorm(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        ElementType tCummulativeDotProduct = 0.;
        IndexType tNumVectors = aInput.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ElementType, IndexType> & tMyInputVector = aInput[tIndex];

            mControlWorkVector->update(1., tMyInputVector, 0.);
            mControlWorkVector->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVector->dot(*mControlWorkVector);
        }
        ElementType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormProjectedGradient()
    {
        ElementType tCummulativeDotProduct = 0.;
        IndexType tNumVectors = mCurrentGradient->getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ElementType, IndexType> & tMyGradient = (*mCurrentGradient)[tIndex];

            mControlWorkVector->update(1., tMyGradient, 0.);
            mControlWorkVector->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkVector->dot(*mControlWorkVector);
        }
        mNormProjectedGradient = std::sqrt(tCummulativeDotProduct);
    }
    ElementType getNormProjectedGradient() const
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
    ElementType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }

    // NOTE: RESET AND STORE STAGE DATA
    void resetCurrentStageDataToPreviousStageData()
    {
        IndexType tNumVectors = mCurrentGradient->getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            locus::Vector<ElementType, IndexType> & tMyCurrentControl = (*mCurrentControl)[tIndex];
            const locus::Vector<ElementType, IndexType> & tMyPreviousControl = (*mPreviousControl)[tIndex];
            tMyCurrentControl.update(1., tMyPreviousControl, 0.);

            locus::Vector<ElementType, IndexType> & tMyCurrentGradient = (*mCurrentGradient)[tIndex];
            const locus::Vector<ElementType, IndexType> & tMyPreviousGradient = (*mPreviousGradient)[tIndex];
            tMyCurrentGradient.update(1., tMyPreviousGradient, 0.);
        }
        mCurrentObjectiveFunctionValue = mPreviousObjectiveFunctionValue;
    }
    void storeCurrentStageData()
    {
        const ElementType tCurrentObjectiveValue = this->getCurrentObjectiveFunctionValue();
        this->setPreviousObjectiveFunctionValue(tCurrentObjectiveValue);

        IndexType tNumVectors = mCurrentControl->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyCurrentControl = this->getCurrentControl(tVectorIndex);
            this->setPreviousControl(tVectorIndex, tMyCurrentControl);

            const locus::Vector<ElementType, IndexType> & tMyCurrentGradient = this->getCurrentGradient(tVectorIndex);
            this->setPreviousGradient(tVectorIndex, tMyCurrentGradient);
        }
    }

private:
    void initialize(const locus::DataFactory<ElementType, IndexType> & aDataFactory)
    {
        assert(aDataFactory.control().getNumVectors() > 0);

        const IndexType tVectorIndex = 0;
        mControlWorkVector = aDataFactory.control(tVectorIndex).create();
        locus::fill(static_cast<ElementType>(0), *mActiveSet);
        locus::fill(static_cast<ElementType>(1), *mInactiveSet);

        ElementType tScalarValue = std::numeric_limits<ElementType>::max();
        locus::fill(tScalarValue, *mControlUpperBounds);
        tScalarValue = -std::numeric_limits<ElementType>::max();
        locus::fill(tScalarValue, *mControlLowerBounds);
    }

private:
    IndexType mNumDualVectors;
    IndexType mNumControlVectors;

    ElementType mStagnationMeasure;
    ElementType mStationarityMeasure;
    ElementType mNormProjectedGradient;
    ElementType mGradientInexactnessTolerance;
    ElementType mObjectiveInexactnessTolerance;
    ElementType mCurrentObjectiveFunctionValue;
    ElementType mPreviousObjectiveFunctionValue;

    bool mIsInitialGuessSet;
    bool mGradientInexactnessToleranceExceeded;
    bool mObjectiveInexactnessToleranceExceeded;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDual;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentGradient;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPreviousGradient;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlUpperBounds;

    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkVector;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlWorkMultiVector;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mControlReductionOperations;

private:
    TrustRegionAlgorithmDataMng(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aRhs);
    locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & operator=(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aRhs);
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

template<typename ElementType, typename IndexType = size_t>
class GradientOperatorBase
{
public:
    virtual ~GradientOperatorBase()
    {
    }

    virtual void update(const locus::AlgorithmDataMng<ElementType, IndexType> & aDataMng) = 0;
    virtual void compute(const locus::MultiVector<ElementType, IndexType> & aState,
                         const locus::MultiVector<ElementType, IndexType> & aControl,
                         locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> create() const = 0;
};

template<typename ElementType, typename IndexType = size_t>
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

    IndexType size() const
    {
        return (mList.size());
    }
    void add(const locus::GradientOperatorBase<ElementType, IndexType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::GradientOperatorBase<ElementType, IndexType> & operator [](const IndexType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::GradientOperatorBase<ElementType, IndexType> & operator [](const IndexType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::GradientOperatorList<ElementType, IndexType>> create() const
    {
        assert(this->size() > static_cast<IndexType>(0));

        std::shared_ptr<locus::GradientOperatorList<ElementType, IndexType>> tOutput =
                std::make_shared<locus::GradientOperatorList<ElementType, IndexType>>();
        const IndexType tNumGradientOperators = this->size();
        for(IndexType tIndex = 0; tIndex < tNumGradientOperators; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);

            const std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> & tGradientOperator = mList[tIndex];
            tOutput->add(tGradientOperator);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> & ptr(const IndexType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>>> mList;

private:
    GradientOperatorList(const locus::GradientOperatorList<ElementType, IndexType>&);
    locus::GradientOperatorList<ElementType, IndexType> & operator=(const locus::GradientOperatorList<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class AnalyticalGradient : public locus::GradientOperatorBase<ElementType, IndexType>
{
public:
    explicit AnalyticalGradient(const locus::Criterion<ElementType, IndexType> & aCriterion) :
            mCriterion(aCriterion.create())
    {
    }
    explicit AnalyticalGradient(const std::shared_ptr<locus::Criterion<ElementType, IndexType>> & aCriterion) :
            mCriterion(aCriterion)
    {
    }
    virtual ~AnalyticalGradient()
    {
    }

    void update(const locus::AlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        return;
    }
    void compute(const locus::MultiVector<ElementType, IndexType> & aState,
                 const locus::MultiVector<ElementType, IndexType> & aControl,
                 locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        locus::fill(static_cast<ElementType>(0), aOutput);
        mCriterion->gradient(aState, aControl, aOutput);
    }
    std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> tOutput =
                std::make_shared<locus::AnalyticalGradient<ElementType, IndexType>>(mCriterion);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> mCriterion;

private:
    AnalyticalGradient(const locus::AnalyticalGradient<ElementType, IndexType> & aRhs);
    locus::AnalyticalGradient<ElementType, IndexType> & operator=(const locus::AnalyticalGradient<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class LinearOperatorBase
{
public:
    virtual ~LinearOperatorBase()
    {
    }

    virtual void update(const locus::AlgorithmDataMng<ElementType, IndexType> & aDataMng) = 0;
    virtual void apply(const locus::MultiVector<ElementType, IndexType> & aState,
                       const locus::MultiVector<ElementType, IndexType> & aControl,
                       const locus::MultiVector<ElementType, IndexType> & aVector,
                       locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> create() const = 0;
};

template<typename ElementType, typename IndexType = size_t>
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

    IndexType size() const
    {
        return (mList.size());
    }
    void add(const locus::LinearOperatorBase<ElementType, IndexType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::LinearOperatorBase<ElementType, IndexType> & operator [](const IndexType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::LinearOperatorBase<ElementType, IndexType> & operator [](const IndexType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::LinearOperatorList<ElementType, IndexType>> create() const
    {
        assert(this->size() > static_cast<IndexType>(0));

        std::shared_ptr<locus::LinearOperatorList<ElementType, IndexType>> tOutput =
                std::make_shared<locus::LinearOperatorList<ElementType, IndexType>>();
        const IndexType tNumLinearOperators = this->size();
        for(IndexType tIndex = 0; tIndex < tNumLinearOperators; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);

            const std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> & tLinearOperator = mList[tIndex];
            tOutput->add(tLinearOperator);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> & ptr(const IndexType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>>> mList;

private:
    LinearOperatorList(const locus::LinearOperatorList<ElementType, IndexType>&);
    locus::LinearOperatorList<ElementType, IndexType> & operator=(const locus::LinearOperatorList<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class AnalyticalHessian : public locus::LinearOperatorBase<ElementType, IndexType>
{
public:
    explicit AnalyticalHessian(const locus::Criterion<ElementType, IndexType> & aCriterion) :
            mCriterion(aCriterion.create())
    {
    }
    explicit AnalyticalHessian(const std::shared_ptr<locus::Criterion<ElementType, IndexType>> & aCriterion) :
            mCriterion(aCriterion)
    {
    }
    virtual ~AnalyticalHessian()
    {
    }

    void update(const locus::AlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        return;
    }
    void apply(const locus::MultiVector<ElementType, IndexType> & aState,
               const locus::MultiVector<ElementType, IndexType> & aControl,
               const locus::MultiVector<ElementType, IndexType> & aVector,
               locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        locus::fill(static_cast<ElementType>(0), aOutput);
        mCriterion->hessian(aState, aControl, aVector, aOutput);
    }
    std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> tOutput =
                std::make_shared<locus::AnalyticalHessian<ElementType, IndexType>>(mCriterion);
        return (tOutput);
    }

private:
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> mCriterion;

private:
    AnalyticalHessian(const locus::AnalyticalHessian<ElementType, IndexType> & aRhs);
    locus::AnalyticalHessian<ElementType, IndexType> & operator=(const locus::AnalyticalHessian<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class PreconditionerBase
{
public:
    virtual ~PreconditionerBase()
    {
    }

    virtual void update(const locus::AlgorithmDataMng<ElementType, IndexType> & aDataMng) = 0;
    virtual void applyPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                     const locus::MultiVector<ElementType, IndexType> & aVector,
                                     locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual void applyInvPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                        const locus::MultiVector<ElementType, IndexType> & aVector,
                                        locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual std::shared_ptr<locus::PreconditionerBase<ElementType, IndexType>> create() const = 0;
};

template<typename ElementType, typename IndexType = size_t>
class IdentityPreconditioner : public PreconditionerBase<ElementType, IndexType>
{
public:
    IdentityPreconditioner()
    {
    }
    virtual ~IdentityPreconditioner()
    {
    }
    void update(const locus::AlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        return;
    }
    void applyPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                             const locus::MultiVector<ElementType, IndexType> & aVector,
                             locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aVector.getNumVectors() == aOutput.getNumVectors());
        locus::update(1., aVector, 0., aOutput);
    }
    void applyInvPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                const locus::MultiVector<ElementType, IndexType> & aVector,
                                locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aVector.getNumVectors() == aOutput.getNumVectors());
        locus::update(1., aVector, 0., aOutput);
    }
    std::shared_ptr<locus::PreconditionerBase<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::PreconditionerBase<ElementType, IndexType>> tOutput =
                std::make_shared<locus::IdentityPreconditioner<ElementType, IndexType>>();
        return (tOutput);
    }

private:
    IdentityPreconditioner(const locus::IdentityPreconditioner<ElementType, IndexType> & aRhs);
    locus::IdentityPreconditioner<ElementType, IndexType> & operator=(const locus::IdentityPreconditioner<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class TrustRegionStageMngBase
{
public:
    virtual ~TrustRegionStageMngBase()
    {
    }

    virtual void update(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng) = 0;
    virtual ElementType evaluateObjective(const locus::MultiVector<ElementType, IndexType> & aControl,
                                          ElementType aTolerance = std::numeric_limits<ElementType>::max()) = 0;
    virtual void computeGradient(const locus::MultiVector<ElementType, IndexType> & aControl,
                                 locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ElementType, IndexType> & aControl,
                                      const locus::MultiVector<ElementType, IndexType> & aVector,
                                      locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual void applyVectorToPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                             const locus::MultiVector<ElementType, IndexType> & aVector,
                                             locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
    virtual void applyVectorToInvPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                                const locus::MultiVector<ElementType, IndexType> & aVector,
                                                locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
};

template<typename ElementType, typename IndexType = size_t>
class AugmentedLagrangianStageMng : public locus::TrustRegionStageMngBase<ElementType, IndexType>
{
public:
    AugmentedLagrangianStageMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory,
                                const locus::Criterion<ElementType, IndexType> & aObjective,
                                const locus::CriterionList<ElementType, IndexType> & aConstraints) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumObjectiveHessianEvaluations(0),
            mMinPenaltyValue(1e-10),
            mPenaltyScaleFactor(0.2),
            mCurrentFeasibilityMeasure(std::numeric_limits<ElementType>::max()),
            mCurrentLagrangeMultipliersPenalty(1),
            mNormObjectiveFunctionGradient(std::numeric_limits<ElementType>::max()),
            mNumConstraintEvaluations(std::vector<IndexType>(aConstraints.size())),
            mNumConstraintGradientEvaluations(std::vector<IndexType>(aConstraints.size())),
            mNumConstraintHessianEvaluations(std::vector<IndexType>(aConstraints.size())),
            mState(aDataFactory.state().create()),
            mDualWorkVec(aDataFactory.dual().create()),
            mControlWorkVec(aDataFactory.control().create()),
            mLagrangeMultipliers(aDataFactory.dual().create()),
            mWorkConstraintValues(aDataFactory.dual().create()),
            mCurrentConstraintValues(aDataFactory.dual().create()),
            mObjectiveFunctionGradient(aDataFactory.control().create()),
            mObjective(aObjective.create()),
            mConstraints(aConstraints.create()),
            mPreconditioner(std::make_shared<locus::IdentityPreconditioner<ElementType, IndexType>>()),
            mObjectiveGradient(nullptr),
            mConstraintGradients(nullptr),
            mObjectiveHessian(nullptr),
            mConstraintHessians(nullptr),
            mDualReductionOperations(aDataFactory.getDualReductionOperations().create())
    {
    }
    virtual ~AugmentedLagrangianStageMng()
    {
    }

    IndexType getNumObjectiveFunctionEvaluations() const
    {
        return (mNumObjectiveFunctionEvaluations);
    }
    IndexType getNumObjectiveGradientEvaluations() const
    {
        return (mNumObjectiveGradientEvaluations);
    }
    IndexType getNumObjectiveHessianEvaluations() const
    {
        return (mNumObjectiveHessianEvaluations);
    }
    IndexType getNumConstraintEvaluations(const IndexType & aIndex) const
    {
        assert(mNumConstraintEvaluations.empty() == false);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mNumConstraintEvaluations.size());
        return (mNumConstraintEvaluations[aIndex]);
    }
    IndexType getNumConstraintGradientEvaluations(const IndexType & aIndex) const
    {
        assert(mNumConstraintGradientEvaluations.empty() == false);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mNumConstraintGradientEvaluations.size());
        return (mNumConstraintGradientEvaluations[aIndex]);
    }
    IndexType getNumConstraintHessianEvaluations(const IndexType & aIndex) const
    {
        assert(mNumConstraintHessianEvaluations.empty() == false);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mNumConstraintHessianEvaluations.size());
        return (mNumConstraintHessianEvaluations[aIndex]);
    }

    ElementType getCurrentLagrangeMultipliersPenalty() const
    {
        return (mCurrentLagrangeMultipliersPenalty);
    }
    ElementType getNormObjectiveFunctionGradient() const
    {
        return (mNormObjectiveFunctionGradient);
    }

    void getLagrangeMultipliers(locus::MultiVector<ElementType, IndexType> & aInput) const
    {
        locus::update(1., *mLagrangeMultipliers, 0., aInput);
    }
    void getCurrentConstraintValues(locus::MultiVector<ElementType, IndexType> & aInput) const
    {
        locus::update(1., *mCurrentConstraintValues, 0., aInput);
    }

    void setObjectiveGradient(const locus::GradientOperatorBase<ElementType, IndexType> & aInput)
    {
        mObjectiveGradient = aInput.create();
    }
    void setConstraintGradients(const locus::GradientOperatorList<ElementType, IndexType> & aInput)
    {
        mConstraintGradients = aInput.create();
    }
    void setObjectiveHessian(const locus::LinearOperatorBase<ElementType, IndexType> & aInput)
    {
        mObjectiveHessian = aInput.create();
    }
    void setConstraintHessians(const locus::LinearOperatorList<ElementType, IndexType> & aInput)
    {
        mConstraintHessians = aInput.create();
    }
    void setPreconditioner(const locus::PreconditionerBase<ElementType, IndexType> & aInput)
    {
        mPreconditioner = aInput.create();
    }

    void update(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        mObjectiveHessian->update(aDataMng);
        const size_t tNumConstraints = mConstraints->size();
        for(size_t tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            (*mConstraintHessians)[tConstraintIndex].update(aDataMng);
        }
    }
    ElementType evaluateObjective(const locus::MultiVector<ElementType, IndexType> & aControl,
                                  ElementType aTolerance = std::numeric_limits<ElementType>::max())
    {
        // Evaluate objective function, f(\mathbf{z})
        ElementType tObjectiveValue = mObjective->value(*mState, aControl);
        this->increaseObjectiveFunctionEvaluationCounter();

        // Evaluate inequality constraints, h(\mathbf{u}(\mathbf{z}),\mathbf{z})
        this->evaluateConstraint(aControl);

        // Evaluate Lagrangian functional, \ell(\mathbf{u}(\mathbf{z}),\mathbf{z},\mu) =
        //   f(\mathbf{u}(\mathbf{z}),\mathbf{z}) + \mu^{T}h(\mathbf{u}(\mathbf{z}),\mathbf{z})
        ElementType tLagrangeMultipliersDotInequalityValue =
                locus::dot(*mLagrangeMultipliers, *mWorkConstraintValues);
        ElementType tLagrangianValue = tObjectiveValue + tLagrangeMultipliersDotInequalityValue;

        // Evaluate augmented Lagrangian functional, \mathcal{L}(\mathbf{z}),\mathbf{z},\mu) =
        //   \ell(\mathbf{u}(\mathbf{z}),\mathbf{z},\mu) +
        //   \frac{1}{2\beta}(h(\mathbf{u}(\mathbf{z}),\mathbf{z})^{T}h(\mathbf{u}(\mathbf{z}),\mathbf{z})),
        //   where \beta\in\mathbb{R} denotes a penalty parameter
        ElementType tInequalityValueDotInequalityValue =
                locus::dot(*mWorkConstraintValues, *mWorkConstraintValues);
        ElementType tAugmentedLagrangianValue = tLagrangianValue
                + ((static_cast<ElementType>(0.5) / mCurrentLagrangeMultipliersPenalty) * tInequalityValueDotInequalityValue);

        return (tAugmentedLagrangianValue);
    }

    void computeGradient(const locus::MultiVector<ElementType, IndexType> & aControl,
                         locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(mObjectiveGradient.get() != nullptr);

        locus::fill(static_cast<ElementType>(0), aOutput);
        // Compute objective function gradient: \frac{\partial f}{\partial\mathbf{z}}
        locus::fill(static_cast<ElementType>(0), *mObjectiveFunctionGradient);
        mObjectiveGradient->compute(*mState, aControl, *mObjectiveFunctionGradient);
        mNormObjectiveFunctionGradient = locus::norm(*mObjectiveFunctionGradient);
        this->increaseObjectiveGradientEvaluationCounter();

        // Compute inequality constraint gradient: \frac{\partial h_i}{\partial\mathbf{z}}
        const ElementType tOneOverPenalty = static_cast<ElementType>(1.) / mCurrentLagrangeMultipliersPenalty;
        const IndexType tNumConstraintVectors = mCurrentConstraintValues->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
        {
            const IndexType tNumConstraints = (*mCurrentConstraintValues)[tVectorIndex].size();
            for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                assert(mConstraintGradients.get() != nullptr);
                assert(mConstraintGradients->ptr(tConstraintIndex).get() != nullptr);
                // Add contribution from: \lambda_i\frac{\partial h_i}{\partial\mathbf{z}} to Lagrangian gradient
                locus::fill(static_cast<ElementType>(0), *mControlWorkVec);
                (*mConstraintGradients)[tConstraintIndex].compute(*mState, aControl, *mControlWorkVec);
                this->increaseConstraintGradientEvaluationCounter(tConstraintIndex);
                locus::update((*mLagrangeMultipliers)(tVectorIndex, tConstraintIndex), *mControlWorkVec, static_cast<ElementType>(1), aOutput);

                // Add contribution from \mu*h_i(\mathbf{u}(\mathbf{z}),\mathbf{z})\frac{\partial h_i}{\partial\mathbf{z}}.
                ElementType tAlpha = tOneOverPenalty * (*mCurrentConstraintValues)(tVectorIndex, tConstraintIndex);
                locus::update(tAlpha, *mControlWorkVec, static_cast<ElementType>(1), aOutput);
            }
        }
        // Compute Augmented Lagrangian gradient
        locus::update(static_cast<ElementType>(1), *mObjectiveFunctionGradient, static_cast<ElementType>(1), aOutput);
    }
    /*! Reduced space interface: Assemble the reduced space gradient operator. \n
        In: \n
            aControl = design variable vector, unchanged on exist. \n
            aVector = trial descent direction, unchanged on exist. \n
        Out: \n
            aOutput = application of the trial descent direction to the Hessian operator.
    */
    void applyVectorToHessian(const locus::MultiVector<ElementType, IndexType> & aControl,
                              const locus::MultiVector<ElementType, IndexType> & aVector,
                              locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(mObjectiveHessian.get() != nullptr);
        locus::fill(static_cast<ElementType>(0), aOutput);
        mObjectiveHessian->apply(*mState, aControl, aVector, aOutput);
        this->increaseObjectiveHessianEvaluationCounter();

        // Apply vector to inequality constraint Hessian operator and add contribution to total Hessian
        const ElementType tOneOverPenalty = static_cast<ElementType>(1.) / mCurrentLagrangeMultipliersPenalty;
        const IndexType tNumConstraintVectors = mCurrentConstraintValues->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
        {
            const IndexType tNumConstraints = (*mCurrentConstraintValues)[tVectorIndex].size();
            for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; ++ tConstraintIndex)
            {
                assert(mConstraintHessians.get() != nullptr);
                assert(mConstraintHessians->ptr(tConstraintIndex).get() != nullptr);
                // Add contribution from: \lambda_i\frac{\partial^2 h_i}{\partial\mathbf{z}^2}
                locus::fill(static_cast<ElementType>(0), *mControlWorkVec);
                (*mConstraintHessians)[tConstraintIndex].apply(*mState, aControl, aVector, *mControlWorkVec);
                this->increaseConstraintHessianEvaluationCounter(tConstraintIndex);
                locus::update((*mLagrangeMultipliers)(tVectorIndex, tConstraintIndex),
                              *mControlWorkVec,
                              static_cast<ElementType>(1),
                              aOutput);

                // Add contribution from: \mu\frac{\partial^2 h_i}{\partial\mathbf{z}^2}\h_i(\mathbf{z})
                ElementType tAlpha = tOneOverPenalty * (*mCurrentConstraintValues)(tVectorIndex, tConstraintIndex);
                locus::update(tAlpha, *mControlWorkVec, static_cast<ElementType>(1), aOutput);

                // Compute Jacobian, i.e. \frac{\partial h_i}{\partial\mathbf{z}}
                locus::fill(static_cast<ElementType>(0), *mControlWorkVec);
                (*mConstraintGradients)[tConstraintIndex].compute(*mState, aControl, *mControlWorkVec);
                this->increaseConstraintGradientEvaluationCounter(tConstraintIndex);

                ElementType tJacobianDotTrialDirection = locus::dot(*mControlWorkVec, aVector);
                ElementType tBeta = tOneOverPenalty * tJacobianDotTrialDirection;
                // Add contribution from: \mu\left(\frac{\partial h_i}{\partial\mathbf{z}}^{T}
                //                        \frac{\partial h_i}{\partial\mathbf{z}}\right)
                locus::update(tBeta, *mControlWorkVec, static_cast<ElementType>(1), aOutput);
            }
        }
    }
    void applyVectorToPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                     const locus::MultiVector<ElementType, IndexType> & aVector,
                                     locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(mPreconditioner.get() != nullptr);
        mPreconditioner->applyPreconditioner(aControl, aVector, aOutput);
    }
    void applyVectorToInvPreconditioner(const locus::MultiVector<ElementType, IndexType> & aControl,
                                        const locus::MultiVector<ElementType, IndexType> & aVector,
                                        locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(mPreconditioner.get() != nullptr);
        mPreconditioner->applyInvPreconditioner(aControl, aVector, aOutput);
    }

    void evaluateConstraint(const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        // Evaluate inequality constraints, h(\mathbf{u}(\mathbf{z}),\mathbf{z})
        const IndexType tNumConstraintVectors = mWorkConstraintValues->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
        {
            const IndexType tNumConstraints = (*mWorkConstraintValues)[tVectorIndex].size();
            for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                (*mWorkConstraintValues)(tVectorIndex, tConstraintIndex) = (*mConstraints)[tConstraintIndex].value(*mState, aControl);
                this->increaseConstraintEvaluationCounter(tConstraintIndex);
            }
        }
    }
    bool updateLagrangeMultipliers()
    {
        bool tIsPenaltyBelowTolerance = false;
        ElementType tPreviousPenalty = mCurrentLagrangeMultipliersPenalty;
        mCurrentLagrangeMultipliersPenalty = mPenaltyScaleFactor * mCurrentLagrangeMultipliersPenalty;
        if(mCurrentLagrangeMultipliersPenalty >= mMinPenaltyValue)
        {
            const IndexType tNumConstraintVectors = mCurrentConstraintValues->getNumVectors();
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumConstraintVectors; tVectorIndex++)
            {
                const IndexType tNumConstraints = (*mCurrentConstraintValues)[tVectorIndex].size();
                for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
                {
                    ElementType tAlpha = static_cast<ElementType>(1.) / tPreviousPenalty;
                    ElementType tBeta = tAlpha * (*mCurrentConstraintValues)(tVectorIndex, tConstraintIndex);
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
        locus::update(static_cast<ElementType>(1), *mWorkConstraintValues, static_cast<ElementType>(0), *mCurrentConstraintValues);
    }
    void computeCurrentFeasibilityMeasure()
    {
        locus::update(static_cast<ElementType>(1), *mCurrentConstraintValues, static_cast<ElementType>(0), *mDualWorkVec);
        const IndexType tNumVectors = mDualWorkVec->getNumVectors();
        std::vector<ElementType> tMaxValues(tNumVectors, 0.);
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ElementType, IndexType> & tMyVector = (*mDualWorkVec)[tVectorIndex];
            tMyVector.modulus();
            tMaxValues[tVectorIndex] = mDualReductionOperations->max(tMyVector);
        }

        mCurrentFeasibilityMeasure = *std::max_element(tMaxValues.begin(), tMaxValues.end());
    }
    ElementType getCurrentFeasibilityMeasure() const
    {
        return (mCurrentFeasibilityMeasure);
    }

private:
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
    void increaseConstraintEvaluationCounter(const IndexType & aIndex)
    {
        mNumConstraintEvaluations[aIndex] = mNumConstraintEvaluations[aIndex] + static_cast<IndexType>(1);
    }
    void increaseConstraintGradientEvaluationCounter(const IndexType & aIndex)
    {
        mNumConstraintGradientEvaluations[aIndex] = mNumConstraintGradientEvaluations[aIndex] + static_cast<IndexType>(1);
    }
    void increaseConstraintHessianEvaluationCounter(const IndexType & aIndex)
    {
        mNumConstraintHessianEvaluations[aIndex] = mNumConstraintHessianEvaluations[aIndex] + static_cast<IndexType>(1);
    }

private:
    IndexType mNumObjectiveFunctionEvaluations;
    IndexType mNumObjectiveGradientEvaluations;
    IndexType mNumObjectiveHessianEvaluations;

    ElementType mMinPenaltyValue;
    ElementType mPenaltyScaleFactor;
    ElementType mCurrentFeasibilityMeasure;
    ElementType mCurrentLagrangeMultipliersPenalty;
    ElementType mNormObjectiveFunctionGradient;

    std::vector<IndexType> mNumConstraintEvaluations;
    std::vector<IndexType> mNumConstraintGradientEvaluations;
    std::vector<IndexType> mNumConstraintHessianEvaluations;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mState;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDualWorkVec;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlWorkVec;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mLagrangeMultipliers;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mWorkConstraintValues;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentConstraintValues;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mObjectiveFunctionGradient;

    std::shared_ptr<locus::Criterion<ElementType, IndexType>> mObjective;
    std::shared_ptr<locus::CriterionList<ElementType, IndexType>> mConstraints;

    std::shared_ptr<locus::PreconditionerBase<ElementType, IndexType>> mPreconditioner;

    std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> mObjectiveGradient;
    std::shared_ptr<locus::GradientOperatorList<ElementType, IndexType>> mConstraintGradients;

    std::shared_ptr<locus::LinearOperatorBase<ElementType, IndexType>> mObjectiveHessian;
    std::shared_ptr<locus::LinearOperatorList<ElementType, IndexType>> mConstraintHessians;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mDualReductionOperations;

private:
    AugmentedLagrangianStageMng(const locus::AugmentedLagrangianStageMng<ElementType, IndexType>&);
    locus::AugmentedLagrangianStageMng<ElementType, IndexType> & operator=(const locus::AugmentedLagrangianStageMng<ElementType, IndexType>&);
};

template<typename ElementType, typename IndexType = size_t>
class SteihaugTointSolverBase
{
public:
    SteihaugTointSolverBase() :
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
    virtual ~SteihaugTointSolverBase()
    {
    }

    void setMaxNumIterations(const IndexType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    IndexType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setNumIterationsDone(const IndexType & aInput)
    {
        mNumIterationsDone = aInput;
    }
    IndexType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    void setSolverTolerance(const ElementType & aInput)
    {
        mTolerance = aInput;
    }
    ElementType getSolverTolerance() const
    {
        return (mTolerance);
    }
    void setTrustRegionRadius(const ElementType & aInput)
    {
        mTrustRegionRadius = aInput;
    }
    ElementType getTrustRegionRadius() const
    {
        return (mTrustRegionRadius);
    }
    void setNormResidual(const ElementType & aInput)
    {
        mNormResidual = aInput;
    }
    ElementType getNormResidual() const
    {
        return (mNormResidual);
    }
    void setRelativeTolerance(const ElementType & aInput)
    {
        mRelativeTolerance = aInput;
    }
    ElementType getRelativeTolerance() const
    {
        return (mRelativeTolerance);
    }
    void setRelativeToleranceExponential(const ElementType & aInput)
    {
        mRelativeToleranceExponential = aInput;
    }
    ElementType getRelativeToleranceExponential() const
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
    ElementType computeSteihaugTointStep(const locus::MultiVector<ElementType, IndexType> & aNewtonStep,
                                         const locus::MultiVector<ElementType, IndexType> & aConjugateDir,
                                         const locus::MultiVector<ElementType, IndexType> & aPrecTimesNewtonStep,
                                         const locus::MultiVector<ElementType, IndexType> & aPrecTimesConjugateDir)
    {
        assert(aNewtonStep.getNumVectors() == aConjugateDir.getNumVectors());
        assert(aNewtonStep.getNumVectors() == aPrecTimesNewtonStep.getNumVectors());
        assert(aNewtonStep.getNumVectors() == aPrecTimesConjugateDir.getNumVectors());

        // Dogleg trust region step
        IndexType tNumVectors = aNewtonStep.getNumVectors();
        ElementType tNewtonStepDotPrecTimesNewtonStep = 0;
        ElementType tNewtonStepDotPrecTimesConjugateDir = 0;
        ElementType tConjugateDirDotPrecTimesConjugateDir = 0;
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            tNewtonStepDotPrecTimesNewtonStep += aNewtonStep[tVectorIndex].dot(aPrecTimesNewtonStep[tVectorIndex]);
            tNewtonStepDotPrecTimesConjugateDir += aNewtonStep[tVectorIndex].dot(aPrecTimesConjugateDir[tVectorIndex]);
            tConjugateDirDotPrecTimesConjugateDir += aConjugateDir[tVectorIndex].dot(aPrecTimesConjugateDir[tVectorIndex]);
        }

        ElementType tTrustRegionRadius = this->getTrustRegionRadius();
        ElementType tAlpha = tNewtonStepDotPrecTimesConjugateDir * tNewtonStepDotPrecTimesConjugateDir;
        ElementType tBeta = tConjugateDirDotPrecTimesConjugateDir
                * (tTrustRegionRadius * tTrustRegionRadius - tNewtonStepDotPrecTimesNewtonStep);
        ElementType tAlphaPlusBeta = tAlpha + tBeta;
        ElementType tNumerator = -tNewtonStepDotPrecTimesConjugateDir + std::sqrt(tAlphaPlusBeta);
        ElementType tStep = tNumerator / tConjugateDirDotPrecTimesConjugateDir;

        return (tStep);
    }
    bool invalidCurvatureDetected(const ElementType & aInput)
    {
        bool tInvalidCurvatureDetected = false;

        if(aInput < static_cast<ElementType>(0.))
        {
            this->setStoppingCriterion(locus::solver::stop_t::NEGATIVE_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::abs(aInput) <= std::numeric_limits<ElementType>::min())
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
    bool toleranceSatisfied(const ElementType & aNormDescentDirection)
    {
        this->setNormResidual(aNormDescentDirection);
        ElementType tStoppingTolerance = this->getSolverTolerance();

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

    virtual void solve(locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
                       locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng) = 0;

private:
    IndexType mMaxNumIterations;
    IndexType mNumIterationsDone;

    ElementType mTolerance;
    ElementType mNormResidual;
    ElementType mTrustRegionRadius;
    ElementType mRelativeTolerance;
    ElementType mRelativeToleranceExponential;

    locus::solver::stop_t mStoppingCriterion;

private:
    SteihaugTointSolverBase(const locus::SteihaugTointSolverBase<ElementType, IndexType> & aRhs);
    locus::SteihaugTointSolverBase<ElementType, IndexType> & operator=(const locus::SteihaugTointSolverBase<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class ProjectedSteihaugTointPcg : public locus::SteihaugTointSolverBase<ElementType, IndexType>
{
public:
    explicit ProjectedSteihaugTointPcg(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            locus::SteihaugTointSolverBase<ElementType, IndexType>(),
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

    void solve(locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
               locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        IndexType tNumVectors = aDataMng.getNumControlVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            (*mNewtonStep)[tVectorIndex].fill(0);
            (*mConjugateDirection)[tVectorIndex].fill(0);

            const locus::Vector<ElementType, IndexType> & tCurrentGradient = aDataMng.getCurrentGradient(tVectorIndex);
            (*mResidual)[tVectorIndex].update(static_cast<ElementType>(-1.), tCurrentGradient, static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mResidual)[tVectorIndex].entryWiseProduct(tInactiveSet);
        }
        ElementType tNormResidual = locus::norm(*mResidual);
        this->setNormResidual(tNormResidual);

        this->iterate(aDataMng, aStageMng);

        ElementType tNormNewtonStep = locus::norm(*mNewtonStep);
        if(tNormNewtonStep <= static_cast<ElementType>(0.))
        {
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                const locus::Vector<ElementType, IndexType> & tCurrentGradient = aDataMng.getCurrentGradient(tVectorIndex);
                (*mNewtonStep)[tVectorIndex].update(static_cast<ElementType>(-1.), tCurrentGradient, static_cast<ElementType>(0.));
                const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
                (*mNewtonStep)[tVectorIndex].entryWiseProduct(tInactiveSet);
            }
        }
        aDataMng.setTrialStep(*mNewtonStep);
    }

private:
    void iterate(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                 locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng)
    {
        ElementType tPreviousTau = 0;
        ElementType tNormResidual = this->getNormResidual();
        ElementType tCurrentTrustRegionRadius = this->getTrustRegionRadius();

        IndexType tIteration = 0;
        IndexType tMaxNumIterations = this->getMaxNumIterations();
        while(this->toleranceSatisfied(tNormResidual) == false)
        {
            tIteration++;
            if(tIteration > tMaxNumIterations)
            {
                tIteration = tIteration - static_cast<IndexType>(1);
                this->setStoppingCriterion(locus::solver::stop_t::MAX_ITERATIONS);
                break;
            }
            this->applyVectorToInvPreconditioner(aDataMng, *mResidual, aStageMng, *mInvPrecTimesResidual);
            //compute scaling
            ElementType tCurrentTau = locus::dot(*mResidual, *mInvPrecTimesResidual);
            if(tIteration > 1)
            {
                ElementType tBeta = tCurrentTau / tPreviousTau;
                locus::update(static_cast<ElementType>(1.), *mInvPrecTimesResidual, tBeta, *mConjugateDirection);
            }
            else
            {
                locus::update(static_cast<ElementType>(1.), *mInvPrecTimesResidual, static_cast<ElementType>(0.), *mConjugateDirection);
            }
            this->applyVectorToHessian(aDataMng, *mConjugateDirection, aStageMng, *mHessTimesConjugateDirection);
            ElementType tCurvature = locus::dot(*mConjugateDirection, *mHessTimesConjugateDirection);
            if(this->invalidCurvatureDetected(tCurvature) == true)
            {
                // compute scaled inexact trial step
                ElementType tScaling = this->step(aDataMng, aStageMng);
                locus::update(tScaling, *mConjugateDirection, static_cast<ElementType>(1.), *mNewtonStep);
                break;
            }
            ElementType tRayleighQuotient = tCurrentTau / tCurvature;
            locus::update(-tRayleighQuotient, *mHessTimesConjugateDirection, static_cast<ElementType>(1.), *mResidual);
            tNormResidual = locus::norm(*mResidual);
            locus::update(tRayleighQuotient, *mConjugateDirection, static_cast<ElementType>(1.), *mNewtonStep);
            if(tIteration == static_cast<IndexType>(1))
            {
                locus::update(static_cast<ElementType>(1.), *mNewtonStep, static_cast<ElementType>(0.), *mCauchyStep);
            }
            ElementType tNormNewtonStep = locus::norm(*mNewtonStep);
            if(tNormNewtonStep > tCurrentTrustRegionRadius)
            {
                // compute scaled inexact trial step
                ElementType tScaleFactor = this->step(aDataMng, aStageMng);
                locus::update(tScaleFactor, *mConjugateDirection, static_cast<ElementType>(1), *mNewtonStep);
                this->setStoppingCriterion(locus::solver::stop_t::TRUST_REGION_RADIUS);
                break;
            }
            tPreviousTau = tCurrentTau;
        }
        this->setNumIterationsDone(tIteration);
    }
    ElementType step(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                     locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng)
    {
        this->applyVectorToPreconditioner(aDataMng, *mNewtonStep, aStageMng, *mPrecTimesNewtonStep);
        this->applyVectorToPreconditioner(aDataMng, *mConjugateDirection, aStageMng, *mPrecTimesConjugateDirection);

        ElementType tScaleFactor = this->computeSteihaugTointStep(*mNewtonStep,
                                                                  *mConjugateDirection,
                                                                  *mPrecTimesNewtonStep,
                                                                  *mPrecTimesConjugateDirection);

        return (tScaleFactor);
    }
    void applyVectorToHessian(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                              const locus::MultiVector<ElementType, IndexType> & aVector,
                              locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
                              locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aVector.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const IndexType tNumVectors = aVector.getNumVectors();

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            // Set Active Vector
            (*mActiveVector)[tVectorIndex].update(static_cast<ElementType>(1.), aVector[tVectorIndex], static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tActiveSet = aDataMng.getActiveSet(tVectorIndex);
            (*mActiveVector)[tVectorIndex].entryWiseProduct(tActiveSet);
            // Set Inactive Vector
            (*mInactiveVector)[tVectorIndex].update(static_cast<ElementType>(1.), aVector[tVectorIndex], static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mInactiveVector)[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].fill(0);
        }

        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToHessian(tCurrentControl, *mInactiveVector, aOutput);

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            aOutput[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].update(static_cast<ElementType>(1.), (*mActiveVector)[tVectorIndex], static_cast<ElementType>(1.));
        }
    }
    void applyVectorToPreconditioner(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                                     const locus::MultiVector<ElementType, IndexType> & aVector,
                                     locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
                                     locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aVector.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const IndexType tNumVectors = aVector.getNumVectors();

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            // Set Active Vector
            (*mActiveVector)[tVectorIndex].update(static_cast<ElementType>(1.), aVector[tVectorIndex], static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tActiveSet = aDataMng.getActiveSet(tVectorIndex);
            (*mActiveVector)[tVectorIndex].entryWiseProduct(tActiveSet);
            // Set Inactive Vector
            (*mInactiveVector)[tVectorIndex].update(static_cast<ElementType>(1.), aVector[tVectorIndex], static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mInactiveVector)[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].fill(0);
        }

        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToPreconditioner(tCurrentControl, *mInactiveVector, aOutput);

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            aOutput[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].update(static_cast<ElementType>(1.), (*mActiveVector)[tVectorIndex], static_cast<ElementType>(1.));
        }
    }
    void applyVectorToInvPreconditioner(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                                        const locus::MultiVector<ElementType, IndexType> & aVector,
                                        locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
                                        locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aVector.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const IndexType tNumVectors = aVector.getNumVectors();

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            // Set Active Vector
            (*mActiveVector)[tVectorIndex].update(static_cast<ElementType>(1.), aVector[tVectorIndex], static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tActiveSet = aDataMng.getActiveSet(tVectorIndex);
            (*mActiveVector)[tVectorIndex].entryWiseProduct(tActiveSet);
            // Set Inactive Vector
            (*mInactiveVector)[tVectorIndex].update(static_cast<ElementType>(1.), aVector[tVectorIndex], static_cast<ElementType>(0.));
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            (*mInactiveVector)[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].fill(0);
        }

        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToInvPreconditioner(tCurrentControl, *mInactiveVector, aOutput);

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            aOutput[tVectorIndex].entryWiseProduct(tInactiveSet);
            aOutput[tVectorIndex].update(static_cast<ElementType>(1.), (*mActiveVector)[tVectorIndex], static_cast<ElementType>(1.));
        }
    }

private:
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mResidual;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mNewtonStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCauchyStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mWorkVector;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveVector;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveVector;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConjugateDirection;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPrecTimesNewtonStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInvPrecTimesResidual;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPrecTimesConjugateDirection;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mHessTimesConjugateDirection;

private:
    ProjectedSteihaugTointPcg(const locus::ProjectedSteihaugTointPcg<ElementType, IndexType> & aRhs);
    locus::ProjectedSteihaugTointPcg<ElementType, IndexType> & operator=(const locus::ProjectedSteihaugTointPcg<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class TrustRegionStepMngBase
{
public:
    TrustRegionStepMngBase() :
            mActualReduction(0),
            mTrustRegionRadius(1e4),
            mPredictedReduction(0),
            mMinTrustRegionRadius(1e-4),
            mMaxTrustRegionRadius(1e4),
            mTrustRegionExpansion(2.),
            mTrustRegionContraction(0.5),
            mMinCosineAngleTolerance(1e-2),
            mGradientInexactnessTolerance(std::numeric_limits<ElementType>::max()),
            mObjectiveInexactnessTolerance(std::numeric_limits<ElementType>::max()),
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

    virtual ~TrustRegionStepMngBase()
    {
    }

    void setTrustRegionRadius(const ElementType & aInput)
    {
        mTrustRegionRadius = aInput;
    }
    ElementType getTrustRegionRadius() const
    {
        return (mTrustRegionRadius);
    }
    void setTrustRegionContraction(const ElementType & aInput)
    {
        mTrustRegionContraction = aInput;
    }
    ElementType getTrustRegionContraction() const
    {
        return (mTrustRegionContraction);
    }
    void setTrustRegionExpansion(const ElementType & aInput)
    {
        mTrustRegionExpansion = aInput;
    }
    ElementType getTrustRegionExpansion() const
    {
        return (mTrustRegionExpansion);
    }
    void setMinTrustRegionRadius(const ElementType & aInput)
    {
        mMinTrustRegionRadius = aInput;
    }
    ElementType getMinTrustRegionRadius() const
    {
        return (mMinTrustRegionRadius);
    }
    void setMaxTrustRegionRadius(const ElementType & aInput)
    {
        mMaxTrustRegionRadius = aInput;
    }
    ElementType getMaxTrustRegionRadius() const
    {
        return (mMaxTrustRegionRadius);
    }


    void setGradientInexactnessToleranceConstant(const ElementType & aInput)
    {
        mGradientInexactnessToleranceConstant = aInput;
    }
    ElementType getGradientInexactnessToleranceConstant() const
    {
        return (mGradientInexactnessToleranceConstant);
    }
    void updateGradientInexactnessTolerance(const ElementType & aInput)
    {
        ElementType tMinValue = std::min(mTrustRegionRadius, aInput);
        mGradientInexactnessTolerance = mGradientInexactnessToleranceConstant * tMinValue;
    }
    ElementType getGradientInexactnessTolerance() const
    {
        return (mGradientInexactnessTolerance);
    }


    void setObjectiveInexactnessToleranceConstant(const ElementType & aInput)
    {
        mObjectiveInexactnessToleranceConstant = aInput;
    }
    ElementType getObjectiveInexactnessToleranceConstant() const
    {
        return (mObjectiveInexactnessToleranceConstant);
    }
    void updateObjectiveInexactnessTolerance(const ElementType & aInput)
    {
        mObjectiveInexactnessTolerance = mObjectiveInexactnessToleranceConstant
                * mActualOverPredictedReductionLowerBound * std::abs(aInput);
    }
    ElementType getObjectiveInexactnessTolerance() const
    {
        return (mObjectiveInexactnessTolerance);
    }


    void setActualOverPredictedReductionMidBound(const ElementType & aInput)
    {
        mActualOverPredictedReductionMidBound = aInput;
    }
    ElementType getActualOverPredictedReductionMidBound() const
    {
        return (mActualOverPredictedReductionMidBound);
    }
    void setActualOverPredictedReductionLowerBound(const ElementType & aInput)
    {
        mActualOverPredictedReductionLowerBound = aInput;
    }
    ElementType getActualOverPredictedReductionLowerBound() const
    {
        return (mActualOverPredictedReductionLowerBound);
    }
    void setActualOverPredictedReductionUpperBound(const ElementType & aInput)
    {
        mActualOverPredictedReductionUpperBound = aInput;
    }
    ElementType getActualOverPredictedReductionUpperBound() const
    {
        return (mActualOverPredictedReductionUpperBound);
    }


    void setActualReduction(const ElementType & aInput)
    {
        mActualReduction = aInput;
    }
    ElementType getActualReduction() const
    {
        return (mActualReduction);
    }
    void setPredictedReduction(const ElementType & aInput)
    {
        mPredictedReduction = aInput;
    }
    ElementType getPredictedReduction() const
    {
        return (mPredictedReduction);
    }
    void setMinCosineAngleTolerance(const ElementType & aInput)
    {
        mMinCosineAngleTolerance = aInput;
    }
    ElementType getMinCosineAngleTolerance() const
    {
        return (mMinCosineAngleTolerance);
    }
    void setActualOverPredictedReduction(const ElementType & aInput)
    {
        mActualOverPredictedReduction = aInput;
    }
    ElementType getActualOverPredictedReduction() const
    {
        return (mActualOverPredictedReduction);
    }


    void setNumTrustRegionSubProblemItrDone(const IndexType & aInput)
    {
        mNumTrustRegionSubProblemItrDone = aInput;
    }
    void updateNumTrustRegionSubProblemItrDone()
    {
        mNumTrustRegionSubProblemItrDone++;
    }
    IndexType getNumTrustRegionSubProblemItrDone() const
    {
        return (mNumTrustRegionSubProblemItrDone);
    }
    void setMaxNumTrustRegionSubProblemItr(const IndexType & aInput)
    {
        mMaxNumTrustRegionSubProblemItr = aInput;
    }
    IndexType getMaxNumTrustRegionSubProblemItr() const
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

    virtual bool solveSubProblem(locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                                 locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
                                 locus::SteihaugTointSolverBase<ElementType, IndexType> & aSolver) = 0;

private:
    ElementType mActualReduction;
    ElementType mTrustRegionRadius;
    ElementType mPredictedReduction;
    ElementType mMinTrustRegionRadius;
    ElementType mMaxTrustRegionRadius;
    ElementType mTrustRegionExpansion;
    ElementType mTrustRegionContraction;
    ElementType mMinCosineAngleTolerance;
    ElementType mGradientInexactnessTolerance;
    ElementType mObjectiveInexactnessTolerance;

    ElementType mActualOverPredictedReduction;
    ElementType mActualOverPredictedReductionMidBound;
    ElementType mActualOverPredictedReductionLowerBound;
    ElementType mActualOverPredictedReductionUpperBound;

    ElementType mGradientInexactnessToleranceConstant;
    ElementType mObjectiveInexactnessToleranceConstant;

    IndexType mNumTrustRegionSubProblemItrDone;
    IndexType mMaxNumTrustRegionSubProblemItr;

    bool mIsInitialTrustRegionSetToNormProjectedGradient;

private:
    TrustRegionStepMngBase(const locus::TrustRegionStepMngBase<ElementType, IndexType> & aRhs);
    locus::TrustRegionStepMngBase<ElementType, IndexType> & operator=(const locus::TrustRegionStepMngBase<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class KelleySachsStepMng : public locus::TrustRegionStepMngBase<ElementType, IndexType>
{
public:
    explicit KelleySachsStepMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            locus::TrustRegionStepMngBase<ElementType, IndexType>(),
            mEta(0),
            mEpsilon(0),
            mNormInactiveGradient(0),
            mStationarityMeasureConstant(std::numeric_limits<ElementType>::min()),
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
    ElementType getEtaConstant() const
    {
        return (mEta);
    }
    //! Sets adaptive constants eta, which ensures superlinear convergence
    void setEtaConstant(const ElementType & aInput)
    {
        mEta = aInput;
    }
    //! Returns adaptive constants epsilon, which ensures superlinear convergence
    ElementType getEpsilonConstant() const
    {
        return (mEpsilon);
    }
    //! Sets adaptive constants epsilon, which ensures superlinear convergence
    void setEpsilonConstant(const ElementType &  aInput)
    {
        mEpsilon = aInput;
    }
    void setStationarityMeasureConstant(const ElementType &  aInput)
    {
        mStationarityMeasureConstant = aInput;
    }
    //! Sets objective function value computed with the control values at the mid-point
    void setMidPointObjectiveFunctionValue(const ElementType & aInput)
    {
        mMidPointObjectiveFunction = aInput;
    }
    //! Returns objective function value computed with the control values at the mid-point
    ElementType getMidPointObjectiveFunctionValue() const
    {
        return (mMidPointObjectiveFunction);
    }
    //! Sets control values at the mid-point
    void setMidPointControls(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(mMidControls.get() != nullptr);
        assert(aInput.getNumVectors() == mMidControls->getNumVectors());
        locus::update(static_cast<ElementType>(1), aInput, static_cast<ElementType>(0), *mMidControls);
    }
    //! Returns control values at the mid-point
    const locus::MultiVector<ElementType, IndexType> & getMidPointControls() const
    {
        return (mMidControls.operator*());
    }

    bool solveSubProblem(locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                         locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng,
                         locus::SteihaugTointSolverBase<ElementType, IndexType> & aSolver)
    {
        mTrustRegionRadiusFlag = false;
        bool tTrialControlAccepted = true;
        this->setNumTrustRegionSubProblemItrDone(1);
        ElementType  tMinTrustRegionRadius = this->getMinTrustRegionRadius();
        if(this->getTrustRegionRadius() < tMinTrustRegionRadius)
        {
            this->setTrustRegionRadius(tMinTrustRegionRadius);
        }

        IndexType tMaxNumSubProblemItr = this->getMaxNumTrustRegionSubProblemItr();
        while(this->getNumTrustRegionSubProblemItrDone() <= tMaxNumSubProblemItr)
        {
            // Compute active and inactive sets
            this->computeActiveAndInactiveSet(aDataMng);
            // Set solver tolerance
            this->setSolverTolerance(aDataMng, aSolver);
            // Compute descent direction
            ElementType tTrustRegionRadius = this->getTrustRegionRadius();
            aSolver.setTrustRegionRadius(tTrustRegionRadius);
            aSolver.solve(aStageMng, aDataMng);
            // Compute projected trial step
            this->computeProjectedTrialStep(aDataMng);
            // Apply projected trial step to Hessian operator
            this->applyProjectedTrialStepToHessian(aDataMng, aStageMng);
            // Compute predicted reduction based on mid trial control
            ElementType tPredictedReduction = this->computePredictedReduction(aDataMng);

            if(aDataMng.isObjectiveInexactnessToleranceExceeded() == true)
            {
                tTrialControlAccepted = false;
                break;
            }

            // Update objective function inexactness tolerance (bound)
            this->updateObjectiveInexactnessTolerance(tPredictedReduction);
            // Evaluate current mid objective function
            ElementType tTolerance = this->getObjectiveInexactnessTolerance();
            mMidPointObjectiveFunction = aStageMng.evaluateObjective(*mMidControls, tTolerance);
            // Compute actual reduction based on mid trial control
            ElementType tCurrentObjectiveFunctionValue = aDataMng.getCurrentObjectiveFunctionValue();
            ElementType tActualReduction = mMidPointObjectiveFunction - tCurrentObjectiveFunctionValue;
            this->setActualReduction(tActualReduction);
            // Compute actual over predicted reduction ratio
            ElementType tActualOverPredReduction = tActualReduction /
                    (tPredictedReduction + std::numeric_limits<ElementType>::epsilon());
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
    void setSolverTolerance(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                            locus::SteihaugTointSolverBase<ElementType, IndexType> & aSolver)
    {
        ElementType tCummulativeDotProduct = 0;
        const IndexType tNumVectors = aDataMng.getNumControlVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyInactiveSet = aDataMng.getInactiveSet(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyCurrentGradient = aDataMng.getCurrentGradient(tVectorIndex);
            locus::Vector<ElementType, IndexType> & tMyInactiveGradient = (*mInactiveGradient)[tVectorIndex];

            tMyInactiveGradient.update(static_cast<ElementType>(1), tMyCurrentGradient, static_cast<ElementType>(0));
            tMyInactiveGradient.entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += tMyInactiveGradient.dot(tMyInactiveGradient);
        }
        mNormInactiveGradient = std::sqrt(tCummulativeDotProduct);
        ElementType tSolverStoppingTolerance = this->getEtaConstant() * mNormInactiveGradient;
        aSolver.setSolverTolerance(tSolverStoppingTolerance);
    }
    void computeProjectedTrialStep(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        // Project trial control
        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mMidControls);
        const locus::MultiVector<ElementType, IndexType> & tTrialStep = aDataMng.getTrialStep();
        locus::update(static_cast<ElementType>(1), tTrialStep, static_cast<ElementType>(1), *mMidControls);
        const locus::MultiVector<ElementType, IndexType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ElementType, IndexType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, *mMidControls);

        // Compute projected trial step
        locus::update(static_cast<ElementType>(1), *mMidControls, static_cast<ElementType>(0), *mProjectedTrialStep);
        locus::update(static_cast<ElementType>(-1), tCurrentControl, static_cast<ElementType>(1), *mProjectedTrialStep);
    }
    ElementType computePredictedReduction(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        ElementType tProjTrialStepDotInactiveGradient = 0;
        ElementType tProjTrialStepDotHessTimesProjTrialStep = 0;
        const IndexType tNumVectors = aDataMng.getNumControlVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyInactiveGradient = mInactiveGradient->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyMatrixTimesVector = mMatrixTimesVector->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyProjectedTrialStep = mProjectedTrialStep->operator[](tVectorIndex);

            tProjTrialStepDotInactiveGradient += tMyProjectedTrialStep.dot(tMyInactiveGradient);
            tProjTrialStepDotHessTimesProjTrialStep += tMyProjectedTrialStep.dot(tMyMatrixTimesVector);
        }

        ElementType tPredictedReduction = tProjTrialStepDotInactiveGradient
                + (static_cast<ElementType>(0.5) * tProjTrialStepDotHessTimesProjTrialStep);
        this->setPredictedReduction(tPredictedReduction);

        return (tPredictedReduction);
    }
    bool updateTrustRegionRadius(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        ElementType tActualReduction = this->getActualReduction();
        ElementType tCurrentTrustRegionRadius = this->getTrustRegionRadius();
        ElementType tActualOverPredReduction = this->getActualOverPredictedReduction();
        ElementType tActualOverPredMidBound = this->getActualOverPredictedReductionMidBound();
        ElementType tActualOverPredLowerBound = this->getActualOverPredictedReductionLowerBound();
        ElementType tActualOverPredUpperBound = this->getActualOverPredictedReductionUpperBound();

        bool tStopTrustRegionSubProblem = false;
        ElementType tActualReductionLowerBound = this->computeActualReductionLowerBound(aDataMng);
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
                    static_cast<ElementType>(2) * this->getTrustRegionExpansion() * tCurrentTrustRegionRadius;
            tStopTrustRegionSubProblem = true;
        }
        else
        {
            ElementType tMaxTrustRegionRadius = this->getMaxTrustRegionRadius();
            tCurrentTrustRegionRadius = this->getTrustRegionExpansion() * tCurrentTrustRegionRadius;
            tCurrentTrustRegionRadius = std::min(tMaxTrustRegionRadius, tCurrentTrustRegionRadius);
        }
        this->setTrustRegionRadius(tCurrentTrustRegionRadius);

        return (tStopTrustRegionSubProblem);
    }

    void applyProjectedTrialStepToHessian(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                                          locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng)
    {
        // Compute active projected trial step
        locus::fill(static_cast<ElementType>(0), *mMatrixTimesVector);
        locus::update(static_cast<ElementType>(1),
                      *mProjectedTrialStep,
                      static_cast<ElementType>(0),
                      *mActiveProjectedTrialStep);
        const locus::MultiVector<ElementType, IndexType> & tActiveSet = aDataMng.getActiveSet();
        locus::entryWiseProduct(tActiveSet, *mActiveProjectedTrialStep);

        // Compute inactive projected trial step
        locus::update(static_cast<ElementType>(1),
                      *mProjectedTrialStep,
                      static_cast<ElementType>(0),
                      *mInactiveProjectedTrialStep);
        const locus::MultiVector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet();
        locus::entryWiseProduct(tInactiveSet, *mInactiveProjectedTrialStep);

        // Apply inactive projected trial step to Hessian
        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        aStageMng.applyVectorToHessian(tCurrentControl, *mInactiveProjectedTrialStep, *mMatrixTimesVector);

        // Compute Hessian times projected trial step, i.e. ( ActiveSet + (InactiveSet' * Hess * InactiveSet) ) * Vector
        locus::entryWiseProduct(tInactiveSet, *mMatrixTimesVector);
        locus::update(static_cast<ElementType>(1),
                      *mActiveProjectedTrialStep,
                      static_cast<ElementType>(1),
                      *mMatrixTimesVector);
    }

    ElementType computeActualReductionLowerBound(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        ElementType tConditionOne = this->getTrustRegionRadius()
                / (mNormInactiveGradient + std::numeric_limits<ElementType>::epsilon());
        ElementType tLambda = std::min(tConditionOne, static_cast<ElementType>(1.));

        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mWorkMultiVec);
        locus::update(-tLambda, *mInactiveGradient, static_cast<ElementType>(1), *mWorkMultiVec);

        const locus::MultiVector<ElementType, IndexType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ElementType, IndexType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, *mWorkMultiVec);

        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mProjectedCauchyStep);
        locus::update(static_cast<ElementType>(-1), *mWorkMultiVec, static_cast<ElementType>(1), *mProjectedCauchyStep);

        const ElementType tSLOPE_CONSTANT = 1e-4;
        ElementType tNormProjectedCauchyStep = locus::norm(*mProjectedCauchyStep);
        ElementType tLowerBound = -mStationarityMeasureConstant * tSLOPE_CONSTANT * tNormProjectedCauchyStep;

        return (tLowerBound);
    }

    ElementType computeLambdaScaleFactor(const locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        const locus::MultiVector<ElementType, IndexType> & tGradient = aDataMng.getCurrentGradient();
        locus::update(static_cast<ElementType>(1), tGradient, static_cast<ElementType>(0), *mWorkMultiVec);
        const locus::MultiVector<ElementType, IndexType> & tInactiveSet = aDataMng.getInactiveSet();
        locus::entryWiseProduct(tInactiveSet, *mWorkMultiVec);
        ElementType tNormCurrentProjectedGradient = locus::norm(*mWorkMultiVec);

        ElementType tCondition = 0;
        const ElementType tCurrentTrustRegionRadius = this->getTrustRegionRadius();
        if(tNormCurrentProjectedGradient > 0)
        {
            tCondition = tCurrentTrustRegionRadius / tNormCurrentProjectedGradient;
        }
        else
        {
            ElementType tNormProjectedGradient = aDataMng.getNormProjectedGradient();
            tCondition = tCurrentTrustRegionRadius / tNormProjectedGradient;
        }
        ElementType tLambda = std::min(tCondition, static_cast<ElementType>(1.));

        return (tLambda);
    }

    void computeActiveAndInactiveSet(locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng)
    {
        const locus::MultiVector<ElementType, IndexType> & tMyLowerBound = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ElementType, IndexType> & tMyUpperBound = aDataMng.getControlUpperBounds();
        // Compute current lower bound limit
        locus::fill(mEpsilon, *mWorkMultiVec);
        locus::update(static_cast<ElementType>(1), tMyLowerBound, static_cast<ElementType>(0), *mLowerBoundLimit);
        locus::update(static_cast<ElementType>(-1), *mWorkMultiVec, static_cast<ElementType>(1), *mLowerBoundLimit);
        // Compute current upper bound limit
        locus::update(static_cast<ElementType>(1), tMyUpperBound, static_cast<ElementType>(0), *mUpperBoundLimit);
        locus::update(static_cast<ElementType>(1), *mWorkMultiVec, static_cast<ElementType>(1), *mUpperBoundLimit);

        // Compute active and inactive sets
        const locus::MultiVector<ElementType, IndexType> & tMyCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ElementType>(1), tMyCurrentControl, static_cast<ElementType>(0), *mWorkMultiVec);

        ElementType tLambda = this->computeLambdaScaleFactor(aDataMng);
        const locus::MultiVector<ElementType, IndexType> & tMyGradient = aDataMng.getCurrentGradient();
        locus::update(-tLambda, tMyGradient, static_cast<ElementType>(1), *mWorkMultiVec);
        locus::bounds::computeActiveAndInactiveSets(*mWorkMultiVec,
                                                    *mLowerBoundLimit,
                                                    *mUpperBoundLimit,
                                                    *mActiveSet,
                                                    *mInactiveSet);
        aDataMng.setActiveSet(*mActiveSet);
        aDataMng.setInactiveSet(*mInactiveSet);
    }

private:
    ElementType mEta;
    ElementType mEpsilon;
    ElementType mNormInactiveGradient;
    ElementType mStationarityMeasureConstant;
    ElementType mMidPointObjectiveFunction;

    bool mTrustRegionRadiusFlag;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mMidControls;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mWorkMultiVec;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mLowerBoundLimit;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mUpperBoundLimit;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveGradient;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mMatrixTimesVector;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mProjectedTrialStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mProjectedCauchyStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveProjectedTrialStep;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveProjectedTrialStep;

private:
    KelleySachsStepMng(const locus::KelleySachsStepMng<ElementType, IndexType> & aRhs);
    locus::KelleySachsStepMng<ElementType, IndexType> & operator=(const locus::KelleySachsStepMng<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class KelleySachsBase
{
public:
    explicit KelleySachsBase(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
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
    virtual ~KelleySachsBase()
    {
    }

    void setGradientTolerance(const ElementType & aInput)
    {
        mGradientTolerance = aInput;
    }
    void setTrialStepTolerance(const ElementType & aInput)
    {
        mTrialStepTolerance = aInput;
    }
    void setObjectiveTolerance(const ElementType & aInput)
    {
        mObjectiveTolerance = aInput;
    }
    void setStagnationTolerance(const ElementType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    void setActualReductionTolerance(const ElementType & aInput)
    {
        mActualReductionTolerance = aInput;
    }

    void setMaxNumUpdates(const IndexType & aInput)
    {
        mMaxNumUpdates = aInput;
    }
    void setNumIterationsDone(const IndexType & aInput)
    {
        mNumOuterIterationsDone = aInput;
    }
    void setMaxNumIterations(const IndexType & aInput)
    {
        mMaxNumOuterIterations = aInput;
    }
    void setStoppingCriterion(const locus::algorithm::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    ElementType getStationarityMeasure() const
    {
        return (mStationarityMeasure);
    }
    ElementType getGradientTolerance() const
    {
        return (mGradientTolerance);
    }
    ElementType getTrialStepTolerance() const
    {
        return (mTrialStepTolerance);
    }
    ElementType getObjectiveTolerance() const
    {
        return (mObjectiveTolerance);
    }
    ElementType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    ElementType getActualReductionTolerance() const
    {
        return (mActualReductionTolerance);
    }

    IndexType getMaxNumUpdates() const
    {
        return (mMaxNumUpdates);
    }
    IndexType getNumIterationsDone() const
    {
        return (mNumOuterIterationsDone);
    }
    IndexType getMaxNumIterations() const
    {
        return (mMaxNumOuterIterations);
    }
    locus::algorithm::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }

    bool updateControl(const locus::MultiVector<ElementType, IndexType> & aMidGradient,
                       locus::KelleySachsStepMng<ElementType, IndexType> & aStepMng,
                       locus::TrustRegionAlgorithmDataMng<ElementType, IndexType> & aDataMng,
                       locus::TrustRegionStageMngBase<ElementType, IndexType> & aStageMng)
    {
        bool tControlUpdated = false;

        ElementType tXi = 1.;
        ElementType tBeta = 1e-2;
        ElementType tAlpha = tBeta;
        ElementType tMu = static_cast<ElementType>(1) - static_cast<ElementType>(1e-4);

        ElementType tMidActualReduction = aStepMng.getActualReduction();
        ElementType tMidObjectiveValue = aStepMng.getMidPointObjectiveFunctionValue();
        const locus::MultiVector<ElementType, IndexType> & tMidControl = aStepMng.getMidPointControls();
        const locus::MultiVector<ElementType, IndexType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ElementType, IndexType> & tUpperBounds = aDataMng.getControlUpperBounds();

        IndexType tIteration = 0;
        while(tIteration < mMaxNumUpdates)
        {
            // Compute trial point based on the mid gradient (i.e. mid steepest descent)
            ElementType tLambda = -tXi / tAlpha;
            locus::update(static_cast<ElementType>(1), tMidControl, static_cast<ElementType>(0), *mControlWorkVector);
            locus::update(tLambda, aMidGradient, static_cast<ElementType>(1), *mControlWorkVector);
            locus::bounds::project(tLowerBounds, tUpperBounds, *mControlWorkVector);

            // Compute trial objective function
            ElementType tTolerance = aStepMng.getObjectiveInexactnessTolerance();
            ElementType tTrialObjectiveValue = aStageMng.evaluateObjective(*mControlWorkVector, tTolerance);
            // Compute actual reduction
            ElementType tTrialActualReduction = tTrialObjectiveValue - tMidObjectiveValue;
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
    IndexType mMaxNumUpdates;
    IndexType mMaxNumOuterIterations;
    IndexType mNumOuterIterationsDone;

    ElementType mGradientTolerance;
    ElementType mTrialStepTolerance;
    ElementType mObjectiveTolerance;
    ElementType mStagnationTolerance;
    ElementType mStationarityMeasure;
    ElementType mActualReductionTolerance;

    locus::algorithm::stop_t mStoppingCriterion;

    std::shared_ptr<locus::MultiVector<ElementType,IndexType>> mControlWorkVector;

private:
    KelleySachsBase(const locus::KelleySachsBase<ElementType, IndexType> & aRhs);
    locus::KelleySachsBase<ElementType, IndexType> & operator=(const locus::KelleySachsBase<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class KelleySachsAugmentedLagrangian : public locus::KelleySachsBase<ElementType, IndexType>
{
public:
    KelleySachsAugmentedLagrangian(const std::shared_ptr<locus::DataFactory<ElementType, IndexType>> & aDataFactory,
                                   const std::shared_ptr<locus::TrustRegionAlgorithmDataMng<ElementType, IndexType>> & aDataMng,
                                   const std::shared_ptr<locus::AugmentedLagrangianStageMng<ElementType, IndexType>> & aStageMng) :
            locus::KelleySachsBase<ElementType, IndexType>(*aDataFactory),
            mGammaConstant(1e-3),
            mOptimalityTolerance(1e-5),
            mFeasibilityTolerance(1e-4),
            mGradient(aDataFactory->control().create()),
            mStepMng(std::make_shared<locus::KelleySachsStepMng<ElementType, IndexType>>(*aDataFactory)),
            mSolver(std::make_shared<locus::ProjectedSteihaugTointPcg<ElementType, IndexType>>(*aDataFactory)),
            mDataMng(aDataMng),
            mStageMng(aStageMng)
    {
    }
    virtual ~KelleySachsAugmentedLagrangian()
    {
    }

    void setOptimalityTolerance(const ElementType & aInput)
    {
        mOptimalityTolerance = aInput;
    }
    void setFeasibilityTolerance(const ElementType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }

    void solve()
    {
        assert(mDataMng->isInitialGuessSet() == true);

        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = mDataMng->getCurrentControl();
        ElementType tTolerance = mStepMng->getObjectiveInexactnessTolerance();
        ElementType tCurrentObjectiveFunctionValue = mStageMng->evaluateObjective(tCurrentControl, tTolerance);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mStageMng->updateCurrentConstraintValues();

        mStageMng->computeGradient(tCurrentControl, *mGradient);
        mDataMng->setCurrentGradient(*mGradient);
        mDataMng->computeNormProjectedGradient();
        mDataMng->storeCurrentStageData();

        if(mStepMng->isInitialTrustRegionRadiusSetToNormProjectedGradient() == true)
        {
            ElementType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
            mStepMng->setTrustRegionRadius(tNormProjectedGradient);
        }
        mDataMng->computeStationarityMeasure();

        IndexType tIteration = 0;
        while(1)
        {
            tIteration++;
            this->setNumIterationsDone(tIteration);
            // Compute adaptive constants to ensure superlinear convergence
            ElementType tStationarityMeasure = mDataMng->getStationarityMeasure();
            ElementType tValue = std::pow(tStationarityMeasure, static_cast<ElementType>(0.75));
            ElementType tEpsilon = std::min(static_cast<ElementType>(1e-3), tValue);
            mStepMng->setEpsilonConstant(tEpsilon);
            tValue = std::pow(tStationarityMeasure, static_cast<ElementType>(0.95));
            ElementType tEta = static_cast<ElementType>(0.1) * std::min(static_cast<ElementType>(1e-1), tValue);
            mStepMng->setEtaConstant(tEta);
            // Solve trust region subproblem
            mStepMng->solveSubProblem(*mDataMng, *mStageMng, *mSolver);
            // Update mid objective, control, and gradient information if necessary
            this->updateDataManager();
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
        const locus::MultiVector<ElementType, IndexType> & tMidControl = mStepMng->getMidPointControls();
        mStageMng->computeGradient(tMidControl, *mGradient);

        if(this->updateControl(*mGradient, *mStepMng, *mDataMng, *mStageMng) == true)
        {
            // Update new gradient and inequality constraint values since control
            // was successfully updated; else, keep mid gradient and thus mid control.
            mStageMng->updateCurrentConstraintValues();
            const locus::MultiVector<ElementType, IndexType> & tCurrentControl = mDataMng->getCurrentControl();
            mStageMng->computeGradient(tCurrentControl, *mGradient);
            mDataMng->setCurrentGradient(*mGradient);
        }
        else
        {
            // Keep current objective function, control, and gradient values at mid point
            const ElementType tMidObjectiveFunctionValue = mStepMng->getMidPointObjectiveFunctionValue();
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
        ElementType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
        mStepMng->updateGradientInexactnessTolerance(tNormProjectedGradient);
    }

    bool checkStoppingCriteria()
    {
        bool tStop = false;
        ElementType tCurrentLagrangeMultipliersPenalty = mStageMng->getCurrentLagrangeMultipliersPenalty();
        ElementType tTolerance = mGammaConstant * tCurrentLagrangeMultipliersPenalty;
        ElementType tNormAugmentedLagrangianGradient = mDataMng->getNormProjectedGradient();
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
            const IndexType tIterationCount = this->getNumIterationsDone();
            const ElementType tStagnationMeasure = mDataMng->getStagnationMeasure();
            const ElementType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
            const ElementType tOptimalityMeasure = mDataMng->getNormProjectedGradient();
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
            const ElementType tStagnationMeasure = mDataMng->getStagnationMeasure();
            const ElementType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
            const ElementType tStationarityMeasure = mDataMng->getStationarityMeasure();
            const ElementType tOptimalityMeasure = mDataMng->getNormProjectedGradient();

            if( tStationarityMeasure <= this->getTrialStepTolerance() )
            {
                tStop = true;
                this->setStoppingCriterion(locus::algorithm::stop_t::TRIAL_STEP_TOLERANCE);
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
        const ElementType tFeasibilityMeasure = mStageMng->getCurrentFeasibilityMeasure();
        const ElementType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ElementType tOptimalityMeasure = mDataMng->getNormProjectedGradient();
        const ElementType tNormProjectedAugmentedLagrangianGradient = mDataMng->getNormProjectedGradient();

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
    ElementType mGammaConstant;
    ElementType mOptimalityTolerance;
    ElementType mFeasibilityTolerance;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mGradient;

    std::shared_ptr<locus::KelleySachsStepMng<ElementType, IndexType>> mStepMng;
    std::shared_ptr<locus::ProjectedSteihaugTointPcg<ElementType,IndexType>> mSolver;
    std::shared_ptr<locus::TrustRegionAlgorithmDataMng<ElementType,IndexType>> mDataMng;
    std::shared_ptr<locus::AugmentedLagrangianStageMng<ElementType,IndexType>> mStageMng;

private:
    KelleySachsAugmentedLagrangian(const locus::KelleySachsAugmentedLagrangian<ElementType, IndexType> & aRhs);
    locus::KelleySachsAugmentedLagrangian<ElementType, IndexType> & operator=(const locus::KelleySachsAugmentedLagrangian<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class Circle : public locus::Criterion<ElementType, IndexType>
{
public:
    Circle()
    {
    }
    virtual ~Circle()
    {
    }

    /// \left(\mathbf{z}(0) - 1.\right)^2 + 2\left(\mathbf{z}(1) - 2\right)^2
    ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                      const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<IndexType>(0));

        const IndexType tVectorIndex = 0;
        ElementType tAlpha = aControl(tVectorIndex, 0) - static_cast<ElementType>(1.);
        ElementType tBeta = aControl(tVectorIndex, 1) - static_cast<ElementType>(2);
        tBeta = static_cast<ElementType>(2.) * std::pow(tBeta, static_cast<ElementType>(2));
        ElementType tOutput = std::pow(tAlpha, static_cast<ElementType>(2)) + tBeta;
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                  const locus::MultiVector<ElementType, IndexType> & aControl,
                  locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<IndexType>(0));
        assert(aControl.getNumVectors() > static_cast<IndexType>(0));
        assert(aControl.getNumVectors() == aOutput.getNumVectors());

        const IndexType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) =
                static_cast<ElementType>(2.) * (aControl(tVectorIndex, 0) - static_cast<ElementType>(1.));
        aOutput(tVectorIndex, 1) =
                static_cast<ElementType>(4.) * (aControl(tVectorIndex, 1) - static_cast<ElementType>(2.));

    }
    void hessian(const locus::MultiVector<ElementType, IndexType> & aState,
                 const locus::MultiVector<ElementType, IndexType> & aControl,
                 const locus::MultiVector<ElementType, IndexType> & aVector,
                 locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const IndexType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ElementType>(2.) * aVector(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ElementType>(4.) * aVector(tVectorIndex, 1);
    }
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::Criterion<ElementType, IndexType>> tOutput =
                std::make_shared<locus::Circle<ElementType, IndexType>>();
        return (tOutput);
    }

private:
    Circle(const locus::Circle<ElementType, IndexType> & aRhs);
    locus::Circle<ElementType, IndexType> & operator=(const locus::Circle<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class Radius : public locus::Criterion<ElementType, IndexType>
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
    ElementType value(const locus::MultiVector<ElementType, IndexType> & aState,
                      const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<IndexType>(0));

        const IndexType tVectorIndex = 0;
        ElementType tOutput = std::pow(aControl(tVectorIndex, 0), static_cast<ElementType>(2.)) +
                std::pow(aControl(tVectorIndex, 1), static_cast<ElementType>(2.));
        tOutput = tOutput - mLimit;
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ElementType, IndexType> & aState,
                  const locus::MultiVector<ElementType, IndexType> & aControl,
                  locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<IndexType>(0));
        assert(aControl.getNumVectors() > static_cast<IndexType>(0));
        assert(aControl.getNumVectors() == aOutput.getNumVectors());

        const IndexType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ElementType>(2.) * aControl(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ElementType>(2.) * aControl(tVectorIndex, 1);

    }
    void hessian(const locus::MultiVector<ElementType, IndexType> & aState,
                 const locus::MultiVector<ElementType, IndexType> & aControl,
                 const locus::MultiVector<ElementType, IndexType> & aVector,
                 locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() > static_cast<IndexType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const IndexType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ElementType>(2.) * aVector(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ElementType>(2.) * aVector(tVectorIndex, 1);
    }
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> create() const
    {
        std::shared_ptr<locus::Criterion<ElementType, IndexType>> tOutput =
                std::make_shared<locus::Radius<ElementType, IndexType>>();
        return (tOutput);
    }

private:
    ElementType mLimit;

private:
    Radius(const locus::Radius<ElementType, IndexType> & aRhs);
    locus::Radius<ElementType, IndexType> & operator=(const locus::Radius<ElementType, IndexType> & aRhs);
};

/**********************************************************************************************************/
/************************* CONSERVATIVE CONVEX SEPARABLE APPROXIMATION ALGORITHM **************************/
/**********************************************************************************************************/


template<typename ElementType, typename IndexType = size_t>
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

    IndexType size() const
    {
        return (mList.size());
    }
    void add(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::MultiVector<ElementType, IndexType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::MultiVector<ElementType, IndexType> & operator [](const IndexType & aInput)
    {
        assert(aInput < mList.size());
        assert(mList[aInput].get() != nullptr);
        return (mList[aInput].operator*());
    }
    const locus::MultiVector<ElementType, IndexType> & operator [](const IndexType & aInput) const
    {
        assert(aInput < mList.size());
        assert(mList[aInput].get() != nullptr);
        return (mList[aInput].operator*());
    }
    locus::Vector<ElementType, IndexType> & operator ()(const IndexType & aListIndex, const IndexType & aVectorIndex)
    {
        assert(aListIndex < mList.size());
        assert(mList[aListIndex].get() != nullptr);
        return (mList[aListIndex]->operator[](aVectorIndex));
    }
    const locus::Vector<ElementType, IndexType> & operator ()(const IndexType & aListIndex,
                                                              const IndexType & aVectorIndex) const
    {
        assert(aListIndex < mList.size());
        assert(mList[aListIndex].get() != nullptr);
        return (mList[aListIndex]->operator[](aVectorIndex));
    }
    std::shared_ptr<locus::MultiVectorList<ElementType, IndexType>> create() const
    {
        assert(this->size() > static_cast<IndexType>(0));
        std::shared_ptr<locus::MultiVectorList<ElementType, IndexType>> tOutput =
                std::make_shared<locus::MultiVectorList<ElementType, IndexType>>();
        const IndexType tListSize = this->size();
        for(IndexType tIndex = 0; tIndex < tListSize; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);
            const std::shared_ptr<locus::MultiVector<ElementType, IndexType>> & tMultiVector = mList[tIndex];
            tOutput->add(tMultiVector);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::MultiVector<ElementType, IndexType>> & ptr(const IndexType & aInput) const
    {
        assert(aInput < mList.size());
        assert(mList[aInput].get() != nullptr);
        return(mList[aInput]);
    }

private:
    std::vector<std::shared_ptr<locus::MultiVector<ElementType, IndexType>>> mList;

private:
    MultiVectorList(const locus::MultiVectorList<ElementType, IndexType>&);
    locus::MultiVectorList<ElementType, IndexType> & operator=(const locus::MultiVectorList<ElementType, IndexType>&);
};


template<typename ElementType, typename IndexType = size_t>
class ConservativeConvexSeparableAppxDataMng
{
public:
    explicit ConservativeConvexSeparableAppxDataMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            mIsInitialGuessSet(false),
            mStagnationMeasure(std::numeric_limits<ElementType>::max()),
            mFeasibilityMeasure(std::numeric_limits<ElementType>::max()),
            mStationarityMeasure(std::numeric_limits<ElementType>::max()),
            mNormProjectedGradient(std::numeric_limits<ElementType>::max()),
            mObjectiveStagnationMeasure(std::numeric_limits<ElementType>::max()),
            mDualProblemBoundsScaleFactor(0.5),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
            mPreviousObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
            mDualObjectiveGlobalizationFactor(1),
            mKarushKuhnTuckerConditionsInexactness(std::numeric_limits<ElementType>::max()),
            mDualWorkOne(),
            mDualWorkTwo(),
            mControlWorkOne(),
            mControlWorkTwo(),
            mDual(aDataFactory.dual().create()),
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
            mCurrentConstraintGradients()
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
    IndexType getNumControlVectors() const
    {
        return (mCurrentControl->getNumVectors());
    }
    // NOTE: NUMBER OF DUAL VECTORS
    IndexType getNumDualVectors() const
    {
        return (mDual->getNumVectors());
    }
    // NOTE :GET NUMBER OF CONSTRAINTS
    IndexType getNumConstraints() const
    {
        IndexType tNumConstraints = mCurrentConstraintGradients.size();
        return (tNumConstraints);
    }

    // NOTE: DUAL PROBLEM PARAMETERS
    ElementType getDualProblemBoundsScaleFactor() const
    {
        return (mDualProblemBoundsScaleFactor);
    }
    void setDualProblemBoundsScaleFactor(const ElementType & aInput)
    {
        mDualProblemBoundsScaleFactor = aInput;
    }
    ElementType getDualObjectiveGlobalizationFactor() const
    {
        return (mDualObjectiveGlobalizationFactor);
    }
    void setDualObjectiveGlobalizationFactor(const ElementType & aInput) const
    {
        mDualObjectiveGlobalizationFactor = aInput;
    }
    const locus::MultiVector<ElementType, IndexType> & getConstraintGlobalizationFactors() const
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        return (mConstraintGlobalizationFactors.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getConstraintGlobalizationFactors(const IndexType & aVectorIndex) const
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintGlobalizationFactors->getNumVectors());
        return (mConstraintGlobalizationFactors->operator [](aVectorIndex));
    }
    void setConstraintGlobalizationFactors(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintGlobalizationFactors->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintGlobalizationFactors);
    }
    void setConstraintGlobalizationFactors(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mConstraintGlobalizationFactors.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintGlobalizationFactors->getNumVectors());
        mConstraintGlobalizationFactors->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: OBJECTIVE FUNCTION VALUE
    ElementType getCurrentObjectiveFunctionValue() const
    {
        return (mCurrentObjectiveFunctionValue);
    }
    void setCurrentObjectiveFunctionValue(const ElementType & aInput)
    {
        mCurrentObjectiveFunctionValue = aInput;
    }
    ElementType getPreviousObjectiveFunctionValue() const
    {
        return (mPreviousObjectiveFunctionValue);
    }
    void setPreviousObjectiveFunctionValue(const ElementType & aInput)
    {
        mPreviousObjectiveFunctionValue = aInput;
    }

    // NOTE: SET INITIAL GUESS
    void setInitialGuess(const ElementType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(mCurrentControl->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mCurrentControl->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mCurrentControl->operator [](tVectorIndex).fill(aValue);
        }
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).fill(aValue);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
        mIsInitialGuessSet = true;
    }
    void setInitialGuess(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentControl);
        mIsInitialGuessSet = true;
    }

    // NOTE: DUAL VECTOR
    const locus::MultiVector<ElementType, IndexType> & getDual() const
    {
        assert(mDual.get() != nullptr);
        return (mDual.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getDual(const IndexType & aVectorIndex) const
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        return (mDual->operator [](aVectorIndex));
    }
    void setDual(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mDual->getNumVectors());
        locus::update(1., aInput, 0., *mDual);
    }
    void setDual(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mDual.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mDual->getNumVectors());
        mDual->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: ACTIVE SET FUNCTIONS
    const locus::MultiVector<ElementType, IndexType> & getActiveSet() const
    {
        assert(mActiveSet.get() != nullptr);

        return (mActiveSet.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getActiveSet(const IndexType & aVectorIndex) const
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        return (mActiveSet->operator [](aVectorIndex));
    }
    void setActiveSet(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mActiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mActiveSet);
    }
    void setActiveSet(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mActiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mActiveSet->getNumVectors());

        mActiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: INACTIVE SET FUNCTIONS
    const locus::MultiVector<ElementType, IndexType> & getInactiveSet() const
    {
        assert(mInactiveSet.get() != nullptr);

        return (mInactiveSet.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getInactiveSet(const IndexType & aVectorIndex) const
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        return (mInactiveSet->operator [](aVectorIndex));
    }
    void setInactiveSet(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mInactiveSet->getNumVectors());
        locus::update(1., aInput, 0., *mInactiveSet);
    }
    void setInactiveSet(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mInactiveSet.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mInactiveSet->getNumVectors());

        mInactiveSet->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONTROL
    const locus::MultiVector<ElementType, IndexType> & getCurrentControl() const
    {
        assert(mCurrentControl.get() != nullptr);

        return (mCurrentControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentControl(const IndexType & aVectorIndex) const
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        return (mCurrentControl->operator [](aVectorIndex));
    }
    void setCurrentControl(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentControl->getNumVectors());
        locus::update(static_cast<ElementType>(1), aInput, static_cast<ElementType>(0), *mCurrentControl);
    }
    void setCurrentControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentControl->getNumVectors());

        mCurrentControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: PREVIOUS CONTROL
    const locus::MultiVector<ElementType, IndexType> & getPreviousControl() const
    {
        assert(mPreviousControl.get() != nullptr);

        return (mPreviousControl.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getPreviousControl(const IndexType & aVectorIndex) const
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        return (mPreviousControl->operator [](aVectorIndex));
    }
    void setPreviousControl(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mPreviousControl->getNumVectors());
        locus::update(1., aInput, 0., *mPreviousControl);
    }
    void setPreviousControl(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mPreviousControl.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mPreviousControl->getNumVectors());

        mPreviousControl->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT OBJECTIVE GRADIENT
    const locus::MultiVector<ElementType, IndexType> & getCurrentObjectiveGradient() const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);

        return (mCurrentObjectiveGradient.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentObjectiveGradient(const IndexType & aVectorIndex) const
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentObjectiveGradient->getNumVectors());

        return (mCurrentObjectiveGradient->operator [](aVectorIndex));
    }
    void setCurrentObjectiveGradient(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentObjectiveGradient->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentObjectiveGradient);
    }
    void setCurrentObjectiveGradient(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentObjectiveGradient.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentObjectiveGradient->getNumVectors());

        mCurrentObjectiveGradient->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT SIGMA VECTOR
    const locus::MultiVector<ElementType, IndexType> & getCurrentSigma() const
    {
        assert(mCurrentSigma.get() != nullptr);

        return (mCurrentSigma.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentSigma(const IndexType & aVectorIndex) const
    {
        assert(mCurrentSigma.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentSigma->getNumVectors());

        return (mCurrentSigma->operator [](aVectorIndex));
    }
    void setCurrentSigma(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentSigma->getNumVectors());
        locus::update(1., aInput, 0., *mCurrentSigma);
    }
    void setCurrentSigma(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentSigma.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentSigma->getNumVectors());

        mCurrentSigma->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: SET CONTROL LOWER BOUNDS
    const locus::MultiVector<ElementType, IndexType> & getControlLowerBounds() const
    {
        assert(mControlLowerBounds.get() != nullptr);

        return (mControlLowerBounds.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getControlLowerBounds(const IndexType & aVectorIndex) const
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        return (mControlLowerBounds->operator [](aVectorIndex));
    }
    void setControlLowerBounds(const ElementType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(mControlLowerBounds->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mControlLowerBounds->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlLowerBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlLowerBounds(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlLowerBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mControlLowerBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlLowerBounds->getNumVectors());

        mControlLowerBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlLowerBounds(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlLowerBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlLowerBounds);
    }

    // NOTE: SET CONTROL UPPER BOUNDS
    const locus::MultiVector<ElementType, IndexType> & getControlUpperBounds() const
    {
        assert(mControlUpperBounds.get() != nullptr);

        return (mControlUpperBounds.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getControlUpperBounds(const IndexType & aVectorIndex) const
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        return (mControlUpperBounds->operator [](aVectorIndex));
    }
    void setControlUpperBounds(const ElementType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(mControlUpperBounds->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mControlUpperBounds->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlUpperBounds->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setControlUpperBounds(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).fill(aValue);
    }
    void setControlUpperBounds(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mControlUpperBounds.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mControlUpperBounds->getNumVectors());

        mControlUpperBounds->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setControlUpperBounds(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mControlUpperBounds->getNumVectors());
        locus::update(1., aInput, 0., *mControlUpperBounds);
    }

    // NOTE: CURRENT CONSTRAINT VALUES
    const locus::MultiVector<ElementType, IndexType> & getCurrentConstraintValues() const
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        return (mCurrentConstraintValues.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getCurrentConstraintValues(const IndexType & aVectorIndex) const
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentConstraintValues->getNumVectors());
        return (mCurrentConstraintValues->operator [](aVectorIndex));
    }
    void setCurrentConstraintValues(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mCurrentConstraintValues->getNumVectors());
        locus::update(static_cast<ElementType>(1), aInput, static_cast<ElementType>(0), *mCurrentConstraintValues);
    }
    void setCurrentConstraintValues(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentConstraintValues.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mCurrentConstraintValues->getNumVectors());
        mCurrentConstraintValues->operator [](aVectorIndex).update(1., aInput, 0.);
    }

    // NOTE: CURRENT CONSTRAINT GRADIENTS
    const locus::MultiVectorList<ElementType, IndexType> & getCurrentConstraintGradients() const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        return (mCurrentConstraintGradients.operator*());
    }
    const locus::MultiVector<ElementType, IndexType> & getCurrentConstraintGradients(const IndexType & aConstraintIndex) const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aConstraintIndex >= static_cast<IndexType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);

        return (mCurrentConstraintGradients->operator[](aConstraintIndex));
    }
    const locus::Vector<ElementType, IndexType> & getCurrentConstraintGradients(const IndexType & aConstraintIndex,
                                                                                const IndexType & aVectorIndex) const
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aConstraintIndex >= static_cast<IndexType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);
        assert(aVectorIndex < mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());

        return (mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex));
    }
    void getCurrentConstraintGradients(locus::MultiVectorList<ElementType, IndexType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aInput.size() == mCurrentConstraintGradients->size());

        const IndexType tNumConstraints = aInput.size();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            assert(aInput[tConstraintIndex].get() != nullptr);
            assert(mCurrentConstraintGradients->ptr(tConstraintIndex).get() != nullptr);
            locus::update(static_cast<ElementType>(1),
                          mCurrentConstraintGradients->operator[](tConstraintIndex),
                          static_cast<ElementType>(0),
                          aInput[tConstraintIndex]);
        }
    }
    void setCurrentConstraintGradients(const IndexType & aConstraintIndex,
                                       const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aConstraintIndex >= static_cast<IndexType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(mCurrentConstraintGradients->ptr(aConstraintIndex).get() != nullptr);
        assert(aInput.getNumVectors() == mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());

        locus::update(static_cast<ElementType>(1),
                      aInput,
                      static_cast<ElementType>(0),
                      mCurrentConstraintGradients->operator[](aConstraintIndex));
    }
    void setCurrentConstraintGradients(const IndexType & aConstraintIndex,
                                       const IndexType & aVectorIndex,
                                       const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mCurrentConstraintGradients.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aConstraintIndex >= static_cast<IndexType>(0));
        assert(aConstraintIndex < mCurrentConstraintGradients->size());
        assert(aVectorIndex < mCurrentConstraintGradients->operator[](aConstraintIndex).getNumVectors());
        assert(aInput.size() == mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex).size());

        const ElementType tAlpha = 1;
        const ElementType tBeta = 0;
        mCurrentConstraintGradients->operator()(aConstraintIndex, aVectorIndex).update(tAlpha, aInput, tBeta);
    }

    // NOTE: STAGNATION MEASURE CRITERION
    void computeStagnationMeasure()
    {
        IndexType tNumVectors = mCurrentControl->getNumVectors();
        std::vector<ElementType> storage(tNumVectors, std::numeric_limits<ElementType>::min());
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyCurrentControl = mCurrentControl->operator[](tIndex);
            mControlWorkOne->update(1., tMyCurrentControl, 0.);
            const locus::Vector<ElementType, IndexType> & tMyPreviousControl = mPreviousControl->operator[](tIndex);
            mControlWorkOne->update(-1., tMyPreviousControl, 1.);
            mControlWorkOne->modulus();
            storage[tIndex] = mControlReductions->max(*mControlWorkOne);
        }
        mStagnationMeasure = *std::max_element(storage.begin(), storage.end());
    }
    ElementType getStagnationMeasure() const
    {
        return (mStagnationMeasure);
    }

    // NOTE: NORM OF CURRENT PROJECTED GRADIENT
    ElementType computeProjectedVectorNorm(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        ElementType tCummulativeDotProduct = 0.;
        IndexType tNumVectors = aInput.getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ElementType, IndexType> & tMyInputVector = aInput[tIndex];

            mControlWorkOne->update(1., tMyInputVector, 0.);
            mControlWorkOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkOne->dot(*mControlWorkOne);
        }
        ElementType tOutput = std::sqrt(tCummulativeDotProduct);
        return(tOutput);
    }
    void computeNormProjectedGradient()
    {
        ElementType tCummulativeDotProduct = 0.;
        IndexType tNumVectors = mCurrentObjectiveGradient->getNumVectors();
        for(IndexType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyInactiveSet = (*mInactiveSet)[tIndex];
            const locus::Vector<ElementType, IndexType> & tMyGradient = (*mCurrentObjectiveGradient)[tIndex];

            mControlWorkOne->update(1., tMyGradient, 0.);
            mControlWorkOne->entryWiseProduct(tMyInactiveSet);
            tCummulativeDotProduct += mControlWorkOne->dot(*mControlWorkOne);
        }
        mNormProjectedGradient = std::sqrt(tCummulativeDotProduct);
    }
    ElementType getNormProjectedGradient() const
    {
        return (mNormProjectedGradient);
    }

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeObjectiveStagnationMeasure()
    {
        mObjectiveStagnationMeasure = mPreviousObjectiveFunctionValue - mCurrentObjectiveFunctionValue;
        mObjectiveStagnationMeasure = std::abs(mObjectiveStagnationMeasure);
    }
    ElementType getObjectiveStagnationMeasure() const
    {
        return (mObjectiveStagnationMeasure);
    }

    // NOTE: FEASIBILITY MEASURE CALCULATION
    void computeFeasibilityMeasure()
    {
        const IndexType tNumVectors = mCurrentConstraintValues->getNumVectors();
        std::vector<ElementType> tStorage(tNumVectors, static_cast<ElementType>(0));
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tDual = mDual->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tWork = mDualWorkOne->operator[](tVectorIndex);
            locus::update(static_cast<ElementType>(1), tDual, static_cast<ElementType>(0), tWork);
            tWork.modulus();
            tStorage[tVectorIndex] = mDualReductions->max(tWork);
        }
        const ElementType tInitialValue = 0;
        mFeasibilityMeasure = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
    }
    ElementType getFeasibilityMeasure() const
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

        locus::update(1., *mCurrentControl, 0., *mControlWorkMultiVector);
        locus::update(-1., *mCurrentObjectiveGradient, 1., *mControlWorkMultiVector);
        locus::bounds::project(*mControlLowerBounds, *mControlUpperBounds, *mControlWorkMultiVector);
        locus::update(1., *mCurrentControl, -1., *mControlWorkMultiVector);
        locus::entryWiseProduct(*mInactiveSet, *mControlWorkMultiVector);
        mStationarityMeasure = locus::norm(*mControlWorkMultiVector);
    }
    ElementType getStationarityMeasure() const
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
    void computeKarushKuhnTuckerConditionsInexactness(const locus::MultiVector<ElementType, IndexType> & aControl,
                                                      const locus::MultiVector<ElementType, IndexType> & aDual)
    {
        const IndexType tVectorIndex = 0;
        assert(aDual.getNumVectors() == mDual->getNumVectors());
        assert(aDual.getNumVectors() == static_cast<IndexType>(1));
        assert(aDual[tVectorIndex].size() == mDual->operator[](tVectorIndex).size());
        assert(aControl.getNumVectors() == mCurrentControl->getNumVectors());
        assert(aControl[tVectorIndex].size() == mCurrentControl->operator[](tVectorIndex).size());

        locus::fill(static_cast<ElementType>(0), mControlWorkMultiVector.operator*());
        const locus::Vector<ElementType, IndexType> & tDual = aDual[tVectorIndex];
        const IndexType tNumConstraints = tDual.size();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ElementType, IndexType> tConstraintGradients =
                    mCurrentConstraintGradients[tConstraintIndex].operator*();
            locus::MultiVector<ElementType, IndexType> tConstraintGradientsTimesDual =
                    mControlWorkMultiVector.operator*();
            locus::update(tDual[tConstraintIndex],
                          tConstraintGradients,
                          static_cast<ElementType>(1),
                          tConstraintGradientsTimesDual);
        }

        ElementType tConditioneOne = std::numeric_limits<ElementType>::max();
        ElementType tConditioneTwo = std::numeric_limits<ElementType>::max();
        this->computeConditionsOneAndTwo(aControl, aDual, tConditioneOne, tConditioneTwo);

        const ElementType tConditioneThree = std::numeric_limits<ElementType>::max();
        const ElementType tConditioneFour = std::numeric_limits<ElementType>::max();
        this->computeConditionsThreeAndFour(aControl, aDual, tConditioneThree, tConditioneFour);

        ElementType tNumControls = aControl[tVectorIndex].size();
        ElementType tSum = tConditioneOne + tConditioneTwo + tConditioneThree + tConditioneFour;
        mKarushKuhnTuckerConditionsInexactness = (static_cast<ElementType>(1) / tNumControls) * std::sqrt(tSum);
    }
    ElementType getKarushKuhnTuckerConditionsInexactness() const
    {
        return (mKarushKuhnTuckerConditionsInexactness);
    }


private:
    void initialize()
    {
        const IndexType tControlVectorIndex = 0;
        mControlWorkOne = mCurrentControl->operator[](tControlVectorIndex).create();
        mControlWorkTwo = mCurrentControl->operator[](tControlVectorIndex).create();
        locus::fill(static_cast<ElementType>(0), *mActiveSet);
        locus::fill(static_cast<ElementType>(1), *mInactiveSet);

        assert(mDual->getNumVectors() == static_cast<IndexType>(1));
        const IndexType tDualVectorIndex = 0;
        mDualWorkOne = mDual->operator[](tDualVectorIndex).create();
        mDualWorkTwo = mDual->operator[](tDualVectorIndex).create();

        const IndexType tNumConstraints = mDual->operator[](tDualVectorIndex).size();
        for(IndexType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            mCurrentConstraintGradients->add(mCurrentControl.operator*());
        }

        ElementType tScalarValue = std::numeric_limits<ElementType>::max();
        locus::fill(tScalarValue, *mControlUpperBounds);
        tScalarValue = -std::numeric_limits<ElementType>::max();
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
    void computeConditionsOneAndTwo(const locus::MultiVector<ElementType, IndexType> & aControl,
                                    const locus::MultiVector<ElementType, IndexType> & aDual,
                                    ElementType & aConditionOne,
                                    ElementType & aConditionTwo)
    {
        const IndexType tNumControlVectors = aControl.getNumVectors();
        std::vector<ElementType> tStorageOne(tNumControlVectors, static_cast<ElementType>(0));
        std::vector<ElementType> tStorageTwo(tNumControlVectors, static_cast<ElementType>(0));
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ElementType, IndexType> & tWorkOne = mControlWorkOne.operator*();
            locus::Vector<ElementType, IndexType> & tWorkTwo = mControlWorkTwo.operator*();
            const locus::Vector<ElementType, IndexType> & tControl = aControl[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tObjectiveGradient =
                    mCurrentObjectiveGradient->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tConstraintGradientTimesDual =
                    mControlWorkMultiVector->operator[](tVectorIndex);

            const IndexType tNumControls = tControl.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                tWorkOne[tControlIndex] = tObjectiveGradient[tControlIndex] + tConstraintGradientTimesDual[tControlIndex];
                tWorkOne[tControlIndex] = std::max(static_cast<ElementType>(0), tWorkOne[tControlIndex]);
                tWorkOne[tControlIndex] = (static_cast<ElementType>(1) + tControl[tControlIndex]) * tWorkOne[tControlIndex];
                tWorkOne[tControlIndex] = tWorkOne[tControlIndex] * tWorkOne[tControlIndex];

                tWorkTwo[tControlIndex] = tObjectiveGradient[tControlIndex] + tConstraintGradientTimesDual[tControlIndex];
                tWorkTwo[tControlIndex] = std::max(static_cast<ElementType>(0), -tWorkTwo[tControlIndex]);
                tWorkTwo[tControlIndex] = (static_cast<ElementType>(1) - tControl[tControlIndex]) * tWorkTwo[tControlIndex];
                tWorkTwo[tControlIndex] = tWorkTwo[tControlIndex] * tWorkTwo[tControlIndex];
            }

            tStorageOne[tVectorIndex] = mControlReductions->sum(tWorkOne);
            tStorageTwo[tVectorIndex] = mControlReductions->sum(tWorkTwo);
        }

        const ElementType tInitialValue = 0;
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
    void computeConditionsThreeAndFour(const locus::MultiVector<ElementType, IndexType> & aControl,
                                    const locus::MultiVector<ElementType, IndexType> & aDual,
                                    ElementType & aConditionThree,
                                    ElementType & aConditionFour)
    {
        const IndexType tNumDualVectors = aDual.getNumVectors();
        std::vector<ElementType> tStorageOne(tNumDualVectors, static_cast<ElementType>(0));
        std::vector<ElementType> tStorageTwo(tNumDualVectors, static_cast<ElementType>(0));
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumDualVectors; tVectorIndex++)
        {
            locus::Vector<ElementType, IndexType> & tWorkOne = mDualWorkOne.operator*();
            locus::Vector<ElementType, IndexType> & tWorkTwo = mDualWorkTwo.operator*();
            const locus::Vector<ElementType, IndexType> & tDual = aDual[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tConstraintValues =
                    mCurrentConstraintValues->operator[](tVectorIndex);

            const IndexType tNumConstraints = tDual.size();
            for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                tWorkOne[tConstraintIndex] = std::max(static_cast<ElementType>(0), tConstraintValues[tConstraintIndex]);
                tWorkOne[tConstraintIndex] = tWorkOne[tConstraintIndex] * tWorkOne[tConstraintIndex];

                tWorkTwo[tConstraintIndex] = tDual[tConstraintIndex]
                        * std::max(static_cast<ElementType>(0), -tConstraintValues[tConstraintIndex]);
                tWorkTwo[tConstraintIndex] = tWorkTwo[tConstraintIndex] * tWorkTwo[tConstraintIndex];
            }

            tStorageOne[tVectorIndex] = mDualReductions->sum(tWorkOne);
            tStorageTwo[tVectorIndex] = mDualReductions->sum(tWorkTwo);
        }

        const ElementType tInitialValue = 0;
        aConditionThree = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        aConditionFour = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
    }

private:
    bool mIsInitialGuessSet;

    ElementType mStagnationMeasure;
    ElementType mFeasibilityMeasure;
    ElementType mStationarityMeasure;
    ElementType mNormProjectedGradient;
    ElementType mObjectiveStagnationMeasure;
    ElementType mDualProblemBoundsScaleFactor;
    ElementType mCurrentObjectiveFunctionValue;
    ElementType mPreviousObjectiveFunctionValue;
    ElementType mDualObjectiveGlobalizationFactor;
    ElementType mKarushKuhnTuckerConditionsInexactness;

    std::shared_ptr<locus::Vector<ElementType, IndexType>> mDualWorkOne;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mDualWorkTwo;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkTwo;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDual;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentSigma;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPreviousControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlUpperBounds;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlWorkMultiVector;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentConstraintValues;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentObjectiveGradient;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConstraintGlobalizationFactors;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mDualReductions;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mControlReductions;

    std::shared_ptr<locus::MultiVectorList<ElementType, IndexType>> mCurrentConstraintGradients;

private:
    ConservativeConvexSeparableAppxDataMng(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aRhs);
    locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & operator=(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class ConservativeConvexSeparableAppxStageMng
{
public:
    virtual ~ConservativeConvexSeparableAppxStageMng()
    {
    }

    virtual void update(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng) = 0;
    virtual ElementType evaluateObjective(const locus::MultiVector<ElementType, IndexType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ElementType, IndexType> & aControl,
                                 locus::MultiVector<ElementType, IndexType> & aOutput) = 0;
};

template<typename ElementType, typename IndexType = size_t>
class PrimalProblemStageMng : public locus::ConservativeConvexSeparableAppxStageMng<ElementType, IndexType>
{
public:
    PrimalProblemStageMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumConstraintEvaluations(),
            mNumConstraintGradientEvaluations(),
            mState(aDataFactory.state().create()),
            mObjective(),
            mConstraints(),
            mObjectiveGradient(),
            mConstraintGradients()
    {
    }
    PrimalProblemStageMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory,
                          const locus::Criterion<ElementType, IndexType> & aObjective,
                          const locus::CriterionList<ElementType, IndexType> & aConstraints) :
            mNumObjectiveFunctionEvaluations(0),
            mNumObjectiveGradientEvaluations(0),
            mNumConstraintEvaluations(std::vector<IndexType>(aConstraints.size())),
            mNumConstraintGradientEvaluations(std::vector<IndexType>(aConstraints.size())),
            mState(aDataFactory.state().create()),
            mObjective(aObjective.create()),
            mConstraints(aConstraints.create()),
            mObjectiveGradient(),
            mConstraintGradients()
    {
    }
    virtual ~PrimalProblemStageMng()
    {
    }

    IndexType getNumObjectiveFunctionEvaluations() const
    {
        return (mNumObjectiveFunctionEvaluations);
    }
    IndexType getNumObjectiveGradientEvaluations() const
    {
        return (mNumObjectiveGradientEvaluations);
    }
    IndexType getNumConstraintEvaluations(const IndexType & aIndex) const
    {
        assert(mNumConstraintEvaluations.empty() == false);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mNumConstraintEvaluations.size());
        return (mNumConstraintEvaluations[aIndex]);
    }
    IndexType getNumConstraintGradientEvaluations(const IndexType & aIndex) const
    {
        assert(mNumConstraintGradientEvaluations.empty() == false);
        assert(aIndex >= static_cast<IndexType>(0));
        assert(aIndex < mNumConstraintGradientEvaluations.size());
        return (mNumConstraintGradientEvaluations[aIndex]);
    }

    void setObjectiveGradient(const locus::GradientOperatorBase<ElementType, IndexType> & aInput)
    {
        mObjectiveGradient = aInput.create();
    }
    void setConstraintGradients(const locus::GradientOperatorList<ElementType, IndexType> & aInput)
    {
        mConstraintGradients = aInput.create();
    }

    void update(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
    }
    ElementType evaluateObjective(const locus::MultiVector<ElementType, IndexType> & aControl)
    {
        assert(mObjective.get() != nullptr);

        ElementType tObjectiveFunctionValue = mObjective->value(mState.operator*(), aControl);
        mNumObjectiveFunctionEvaluations++;

        return (tObjectiveFunctionValue);
    }
    void computeGradient(const locus::MultiVector<ElementType, IndexType> & aControl,
                         locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(mObjectiveGradient.get() != nullptr);

        locus::fill(static_cast<ElementType>(0), aOutput);
        mObjectiveGradient->compute(mState.operator*(), aControl, aOutput);
        mNumObjectiveGradientEvaluations++;
    }
    void evaluateConstraints(const locus::MultiVector<ElementType, IndexType> & aControl,
                             locus::MultiVector<ElementType, IndexType> & aOutput)
    {
        assert(mConstraints.get() != nullptr);

        locus::fill(static_cast<ElementType>(0), aOutput);
        const IndexType tNumVectors = aOutput.getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ElementType, IndexType> & tMyOutput = aOutput[tVectorIndex];
            const IndexType tNumConstraints = tMyOutput.size();
            assert(tNumConstraints == mConstraints->size());

            for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
            {
                assert(mConstraints->ptr(tConstraintIndex).get() != nullptr);

                tMyOutput[tConstraintIndex] = mConstraints->operator[](tConstraintIndex).value(*mState, aControl);
                mNumConstraintEvaluations[tConstraintIndex] =
                        mNumConstraintEvaluations[tConstraintIndex] + static_cast<IndexType>(1);
            }
        }
    }
    void computeConstraintGradients(const locus::MultiVector<ElementType, IndexType> & aControl,
                                    locus::MultiVectorList<ElementType, IndexType> & aOutput)
    {
        assert(mConstraintGradients.get() != nullptr);
        assert(mConstraintGradients->size() == aOutput.size());

        const IndexType tNumConstraints = aOutput.size();
        for(IndexType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            assert(mConstraints->ptr(tIndex).get() != nullptr);

            locus::MultiVector<ElementType, IndexType> & tMyOutput = aOutput[tIndex];
            locus::fill(static_cast<ElementType>(0), tMyOutput);
            mConstraints->operator[](tIndex).computeConstraintGradients(*mState, aControl, tMyOutput);
            mNumConstraintGradientEvaluations[tIndex] =
                    mNumConstraintGradientEvaluations[tIndex] + static_cast<IndexType>(1);
        }
    }

private:
    IndexType mNumObjectiveFunctionEvaluations;
    IndexType mNumObjectiveGradientEvaluations;

    std::vector<IndexType> mNumConstraintEvaluations;
    std::vector<IndexType> mNumConstraintGradientEvaluations;

    std::shared_ptr<MultiVector<ElementType, IndexType>> mState;
    std::shared_ptr<locus::Criterion<ElementType, IndexType>> mObjective;
    std::shared_ptr<locus::CriterionList<ElementType, IndexType>> mConstraints;
    std::shared_ptr<locus::GradientOperatorBase<ElementType, IndexType>> mObjectiveGradient;
    std::shared_ptr<locus::GradientOperatorList<ElementType, IndexType>> mConstraintGradients;

private:
    PrimalProblemStageMng(const locus::PrimalProblemStageMng<ElementType, IndexType> & aRhs);
    locus::PrimalProblemStageMng<ElementType, IndexType> & operator=(const locus::PrimalProblemStageMng<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class DualProblemStageMng : public locus::ConservativeConvexSeparableAppxStageMng<ElementType, IndexType>
{
public:
    explicit DualProblemStageMng(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            mEpsilon(1e-6),
            mObjectiveCoefficientA(1),
            mObjectiveCoefficientR(1),
            mTrialAuxiliaryVariableZ(0),
            mCurrentObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
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
    const locus::MultiVector<ElementType, IndexType> & getConstraintCoefficientsA() const
    {
        assert(mConstraintCoefficientsA.get() != nullptr);

        return (mConstraintCoefficientsA.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getConstraintCoefficientsA(const IndexType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        return (mConstraintCoefficientsA->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsA(const ElementType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(mConstraintCoefficientsA->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mConstraintCoefficientsA->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsA->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsA(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        mConstraintCoefficientsA->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsA(const IndexType & aVectorIndex,
                                    const IndexType & aElementIndex,
                                    const ElementType & aValue)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());
        assert(aElementIndex >= static_cast<IndexType>(0));
        assert(aElementIndex < mConstraintCoefficientsA->operator [](aVectorIndex).size());

        mConstraintCoefficientsA->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsA(const IndexType & aVectorIndex,
                                    const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mConstraintCoefficientsA.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsA->getNumVectors());

        mConstraintCoefficientsA->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsA(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsA->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsA);
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS C
    const locus::MultiVector<ElementType, IndexType> & getConstraintCoefficientsC() const
    {
        assert(mConstraintCoefficientsC.get() != nullptr);

        return (mConstraintCoefficientsC.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getConstraintCoefficientsC(const IndexType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        return (mConstraintCoefficientsC->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsC(const ElementType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(mConstraintCoefficientsC->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mConstraintCoefficientsC->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsC->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsC(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        mConstraintCoefficientsC->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsC(const IndexType & aVectorIndex,
                                    const IndexType & aElementIndex,
                                    const ElementType & aValue)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());
        assert(aElementIndex >= static_cast<IndexType>(0));
        assert(aElementIndex < mConstraintCoefficientsC->operator [](aVectorIndex).size());

        mConstraintCoefficientsC->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsC(const IndexType & aVectorIndex,
                                    const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mConstraintCoefficientsC.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsC->getNumVectors());

        mConstraintCoefficientsC->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsC(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsC->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsC);
    }

    // NOTE: DUAL PROBLEM CONSTRAINT COEFFICIENTS D
    const locus::MultiVector<ElementType, IndexType> & getConstraintCoefficientsD() const
    {
        assert(mConstraintCoefficientsD.get() != nullptr);

        return (mConstraintCoefficientsD.operator *());
    }
    const locus::Vector<ElementType, IndexType> & getConstraintCoefficientsD(const IndexType & aVectorIndex) const
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        return (mConstraintCoefficientsD->operator [](aVectorIndex));
    }
    void setConstraintCoefficientsD(const ElementType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(mConstraintCoefficientsD->getNumVectors() > static_cast<IndexType>(0));

        IndexType tNumVectors = mConstraintCoefficientsD->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mConstraintCoefficientsD->operator [](tVectorIndex).fill(aValue);
        }
    }
    void setConstraintCoefficientsD(const IndexType & aVectorIndex, const ElementType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        mConstraintCoefficientsD->operator [](aVectorIndex).fill(aValue);
    }
    void setConstraintCoefficientsD(const IndexType & aVectorIndex,
                                    const IndexType & aElementIndex,
                                    const ElementType & aValue)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());
        assert(aElementIndex >= static_cast<IndexType>(0));
        assert(aElementIndex < mConstraintCoefficientsD->operator [](aVectorIndex).size());

        mConstraintCoefficientsD->operator*().operator()(aVectorIndex, aElementIndex) = aValue;
    }
    void setConstraintCoefficientsD(const IndexType & aVectorIndex, const locus::Vector<ElementType, IndexType> & aInput)
    {
        assert(mConstraintCoefficientsD.get() != nullptr);
        assert(aVectorIndex >= static_cast<IndexType>(0));
        assert(aVectorIndex < mConstraintCoefficientsD->getNumVectors());

        mConstraintCoefficientsD->operator [](aVectorIndex).update(1., aInput, 0.);
    }
    void setConstraintCoefficientsD(const locus::MultiVector<ElementType, IndexType> & aInput)
    {
        assert(aInput.getNumVectors() == mConstraintCoefficientsD->getNumVectors());
        locus::update(1., aInput, 0., *mConstraintCoefficientsD);
    }

    void update(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        // Update Current Objective & Constraint Values
        mCurrentObjectiveFunctionValue = aDataMng.getCurrentObjectiveFunctionValue();
        locus::update(static_cast<ElementType>(1),
                      aDataMng.getCurrentConstraintValues(),
                      static_cast<ElementType>(0),
                      mCurrentConstraintValues.operator*());

        // Update Moving Asymptotes
        const locus::MultiVector<ElementType, IndexType> & tCurrentSigma = aDataMng.getCurrentSigma();
        const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mLowerAsymptotes);
        locus::update(static_cast<ElementType>(-1), tCurrentSigma, static_cast<ElementType>(1), *mLowerAsymptotes);
        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mUpperAsymptotes);
        locus::update(static_cast<ElementType>(1), tCurrentSigma, static_cast<ElementType>(1), *mUpperAsymptotes);

        // Update Trial Control Bounds
        const ElementType tScaleFactor = aDataMng.getDualProblemBoundsScaleFactor();
        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mTrialControlLowerBounds);
        locus::update(-tScaleFactor, tCurrentSigma, static_cast<ElementType>(1), *mTrialControlLowerBounds);
        locus::update(static_cast<ElementType>(1), tCurrentControl, static_cast<ElementType>(0), *mTrialControlUpperBounds);
        locus::update(tScaleFactor, tCurrentSigma, static_cast<ElementType>(1), *mTrialControlUpperBounds);
    }
    ElementType evaluateObjective(const locus::MultiVector<ElementType, IndexType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<IndexType>(1));

        this->computeDualTimesConstraintCoefficientTerms(aDual);
        this->computeTrialControl(aDual);
        this->computeTrialAuxiliaryVariables(aDual);

        const ElementType tObjectiveTerm = mObjectiveCoefficientR + (mTrialAuxiliaryVariableZ * mObjectiveCoefficientA)
                + (mEpsilon * mTrialAuxiliaryVariableZ * mTrialAuxiliaryVariableZ);

        const ElementType tConstraintSummationTerm = this->computeConstraintContribution(aDual);

        ElementType tMovingAsymptotesTerm = this->computeMovingAsymptotesContribution();

        // Add all contributions to dual objective function
        ElementType tOutput = static_cast<ElementType>(-1)
                * (tObjectiveTerm + tConstraintSummationTerm + tMovingAsymptotesTerm);

        return (tOutput);
    }
    void computeGradient(const locus::MultiVector<ElementType, IndexType> & aDual,
                         locus::MultiVector<ElementType, IndexType> & aGradient)
    {
        assert(aDual.getNumVectors() == static_cast<IndexType>(1));

        const IndexType tDualVectorIndex = 0;
        locus::Vector<ElementType, IndexType> & tGradient = aGradient[tDualVectorIndex];
        const locus::Vector<ElementType, IndexType> & tDual = aDual[tDualVectorIndex];
        const locus::Vector<ElementType, IndexType> & tAuxiliaryVariableY = (*mTrialAuxiliaryVariableY)[tDualVectorIndex];
        const locus::Vector<ElementType, IndexType> & tCoefficientsR = (*mConstraintCoefficientsR)[tDualVectorIndex];
        const locus::Vector<ElementType, IndexType> & tCoefficientsA = (*mConstraintCoefficientsA)[tDualVectorIndex];

        IndexType tNumConstraints = tDual.size();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            tGradient[tConstraintIndex] = tCoefficientsR[tConstraintIndex] - tAuxiliaryVariableY[tConstraintIndex]
                    - (tCoefficientsA[tConstraintIndex] * mTrialAuxiliaryVariableZ);

            const locus::MultiVector<ElementType, IndexType> & tMyConstraintCoefficientsP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            const locus::MultiVector<ElementType, IndexType> & tMyConstraintCoefficientsQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();

            const IndexType tNumControlVectors = mTrialControl->getNumVectors();
            std::vector<ElementType> tMyStorageOne(tNumControlVectors);
            std::vector<ElementType> tMyStorageTwo(tNumControlVectors);
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                mControlWorkVectorOne->fill(static_cast<ElementType>(0));
                mControlWorkVectorTwo->fill(static_cast<ElementType>(0));
                const locus::Vector<ElementType, IndexType> & tMyTrialControl = (*mTrialControl)[tVectorIndex];
                const locus::Vector<ElementType, IndexType> & tMyLowerAsymptotes = (*mLowerAsymptotes)[tVectorIndex];
                const locus::Vector<ElementType, IndexType> & tMyUpperAsymptotes = (*mUpperAsymptotes)[tVectorIndex];
                const locus::Vector<ElementType, IndexType> & tMyCoefficientsP = tMyConstraintCoefficientsP[tVectorIndex];
                const locus::Vector<ElementType, IndexType> & tMyCoefficientsQ = tMyConstraintCoefficientsQ[tVectorIndex];

                const IndexType tNumControls = tMyTrialControl.size();
                for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    (*mControlWorkVectorOne)[tControlIndex] = tMyCoefficientsP[tControlIndex]
                            / (tMyUpperAsymptotes[tControlIndex] - tMyTrialControl[tControlIndex]);

                    (*mControlWorkVectorTwo)[tControlIndex] = tMyCoefficientsQ[tControlIndex]
                            / (tMyTrialControl[tControlIndex] - tMyLowerAsymptotes[tControlIndex]);
                }

                tMyStorageOne[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
                tMyStorageTwo[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorTwo.operator*());
            }

            const ElementType tInitialValue = 0;
            const ElementType tGlobalSumP = std::accumulate(tMyStorageOne.begin(), tMyStorageOne.end(), tInitialValue);
            const ElementType tGlobalSumQ = std::accumulate(tMyStorageTwo.begin(), tMyStorageTwo.end(), tInitialValue);
            // Add contribution to dual gradient
            tGradient[tConstraintIndex] = static_cast<ElementType>(-1)
                    * (tGradient[tConstraintIndex] + tGlobalSumP + tGlobalSumQ);
        }
    }

    void getTrialControl(locus::MultiVector<ElementType, IndexType> & aInput) const
    {
        locus::update(static_cast<ElementType>(1), *mTrialControl, static_cast<ElementType>(0), aInput);
    }
    void updateObjectiveCoefficients(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        mObjectiveCoefficientR = mCurrentObjectiveFunctionValue;
        const ElementType tGlobalizationFactor = aDataMng.getDualObjectiveGlobalizationFactor();
        const locus::MultiVector<ElementType, IndexType> & tCurrentSigma = aDataMng.getCurrentSigma();

        const IndexType tNumVectors = tCurrentSigma.getNumVectors();
        std::vector<ElementType> tStorage(tNumVectors);

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlWorkVectorOne->fill(static_cast<ElementType>(0));
            const locus::Vector<ElementType, IndexType> & tMyCurrentSigma = tCurrentSigma[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tMyCurrentObjectiveGradient =
                    aDataMng.getCurrentObjectiveGradient(tVectorIndex);

            IndexType tNumControls = tMyCurrentSigma.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ElementType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                        * tMyCurrentSigma[tControlIndex];
                (*mObjectiveCoefficientsP)(tVectorIndex, tControlIndex) = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ElementType>(0), tMyCurrentObjectiveGradient[tControlIndex]);
                +((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ElementType>(4));

                (*mObjectiveCoefficientsQ)(tVectorIndex, tControlIndex) = tCurrentSigmaTimesCurrentSigma
                        * std::max(static_cast<ElementType>(0), -tMyCurrentObjectiveGradient[tControlIndex])
                        + ((tGlobalizationFactor * tMyCurrentSigma[tControlIndex]) / static_cast<ElementType>(4));
                (*mControlWorkVectorOne)[tControlIndex] = ((*mObjectiveCoefficientsP)(tVectorIndex, tControlIndex)
                        + (*mObjectiveCoefficientsQ)(tVectorIndex, tControlIndex)) / tMyCurrentSigma[tControlIndex];
            }
            tStorage[tVectorIndex] = mControlReductionOperations->sum(*mControlWorkVectorOne);
        }

        const ElementType tInitialValue = 0;
        const ElementType tValue = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
        mObjectiveCoefficientR = mObjectiveCoefficientR - tValue;
    }
    void updateConstraintCoefficients(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<IndexType>(1));

        const IndexType tDualVectorIndex = 0;
        const locus::Vector<ElementType, IndexType> & tCurrentConstraintValues =
                mCurrentConstraintValues->operator[](tDualVectorIndex);
        const locus::Vector<ElementType, IndexType> & tGlobalizationFactor =
                aDataMng.getConstraintGlobalizationFactors(tDualVectorIndex);
        locus::Vector<ElementType, IndexType> & tConstraintCoefficientsR =
                mConstraintCoefficientsR.operator*()[tDualVectorIndex];

        const IndexType tNumConstraints = tConstraintCoefficientsR.size();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            tConstraintCoefficientsR[tConstraintIndex] = tCurrentConstraintValues[tConstraintIndex];
            const locus::MultiVector<ElementType, IndexType> & tCurrentConstraintGradients =
                    aDataMng.getCurrentConstraintGradients(tConstraintIndex);
            locus::MultiVector<ElementType, IndexType> & tMyConstraintCoeffP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            locus::MultiVector<ElementType, IndexType> & tMyConstraintCoeffQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();
            assert(tCurrentConstraintGradients.getNumVectors() == aDataMng.getNumControlVectors());

            const IndexType tNumControlVectors = tCurrentConstraintGradients.getNumVectors();
            std::vector<ElementType> tStorage(tNumControlVectors);
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                mControlWorkVectorOne->fill(static_cast<ElementType>(0));
                locus::Vector<ElementType, IndexType> & tMyCoeffP = tMyConstraintCoeffP[tVectorIndex];
                locus::Vector<ElementType, IndexType> & tMyCoeffQ = tMyConstraintCoeffQ[tVectorIndex];
                const locus::Vector<ElementType, IndexType> & tMyCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ElementType, IndexType> & tMyCurrentGradient = tCurrentConstraintGradients[tVectorIndex];

                const IndexType tNumControls = tMyCurrentGradient.size();
                for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ElementType tCurrentSigmaTimesCurrentSigma = tMyCurrentSigma[tControlIndex]
                            * tMyCurrentSigma[tControlIndex];
                    tMyCoeffP[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                            * std::max(static_cast<ElementType>(0), tMyCurrentGradient[tControlIndex])
                            + ((tGlobalizationFactor[tConstraintIndex] * tMyCurrentSigma[tControlIndex])
                                    / static_cast<ElementType>(4));

                    tMyCoeffQ[tControlIndex] = tCurrentSigmaTimesCurrentSigma
                            * std::max(static_cast<ElementType>(0), -tMyCurrentGradient[tControlIndex])
                            + ((tGlobalizationFactor[tConstraintIndex] * tMyCurrentSigma[tControlIndex])
                                    / static_cast<ElementType>(4));

                    (*mControlWorkVectorOne)[tControlIndex] = (tMyCoeffP[tControlIndex] + tMyCoeffQ[tControlIndex])
                            / tMyCurrentSigma[tControlIndex];
                }
                tStorage[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
            }

            const ElementType tInitialValue = 0;
            const ElementType tValue = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
            tConstraintCoefficientsR[tConstraintIndex] = tConstraintCoefficientsR[tConstraintIndex] - tValue;
        }
    }
    void initializeAuxiliaryVariables(const locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<IndexType>(1));

        const IndexType tDualVectorIndex = 0;
        locus::Vector<ElementType> & tAuxiliaryVariablesY =
                mTrialAuxiliaryVariableY->operator[](tDualVectorIndex);
        const locus::Vector<ElementType> & tCoefficientsA =
                mConstraintCoefficientsA->operator[](tDualVectorIndex);
        const locus::Vector<ElementType, IndexType> & tCurrentConstraintValues =
                aDataMng.getCurrentConstraintValues(tDualVectorIndex);

        const IndexType tNumConstraints = mDualWorkVector->size();
        const ElementType tMaxCoefficientA = mDualReductionOperations->max(tCoefficientsA);
        if(tMaxCoefficientA > static_cast<ElementType>(0))
        {
            locus::fill(static_cast<ElementType>(0), mDualWorkVector.operator*());
            for(IndexType tIndex = 0; tIndex < tNumConstraints; tIndex++)
            {
                if(tCoefficientsA[tIndex] > static_cast<ElementType>(0))
                {
                    ElementType tValue = std::max(static_cast<ElementType>(0), tCurrentConstraintValues[tIndex]);
                    (*mDualWorkVector)[tIndex] = tValue / tCoefficientsA[tIndex];
                    tAuxiliaryVariablesY[tIndex] = 0;
                }
                else
                {
                    tAuxiliaryVariablesY[tIndex] =
                            std::max(static_cast<ElementType>(0), tCurrentConstraintValues[tIndex]);
                }
            }
            mTrialAuxiliaryVariableZ = mDualReductionOperations->max(mDualWorkVector.operator*());
        }
        else
        {
            for(IndexType tIndex = 0; tIndex < tNumConstraints; tIndex++)
            {
                tAuxiliaryVariablesY[tIndex] = tCurrentConstraintValues[tIndex];
            }
            mTrialAuxiliaryVariableZ = 0;
        }
    }
    void checkConstraintCoefficients()
    {
        const IndexType tDualVectorIndex = 0;
        ElementType tMinCoeffA = mDualReductionOperations->min(mConstraintCoefficientsA->operator[](tDualVectorIndex));
        assert(tMinCoeffA >= static_cast<ElementType>(0));
        ElementType tMinCoeffC = mDualReductionOperations->min(mConstraintCoefficientsC->operator[](tDualVectorIndex));
        assert(tMinCoeffC >= static_cast<ElementType>(0));
        ElementType tMinCoeffD = mDualReductionOperations->min(mConstraintCoefficientsD->operator[](tDualVectorIndex));
        assert(tMinCoeffD >= static_cast<ElementType>(0));
    }

private:
    void initialize(const locus::DataFactory<ElementType, IndexType> & aDataFactory)
    {
        const IndexType tDualVectorIndex = 0;
        mDualWorkVector = aDataFactory.dual(tDualVectorIndex).create();

        const IndexType tControlVectorIndex = 0;
        mControlWorkVectorOne = aDataFactory.control(tControlVectorIndex).create();
        mControlWorkVectorTwo = aDataFactory.control(tControlVectorIndex).create();

        const IndexType tNumConstraints = aDataFactory.dual(tDualVectorIndex).size();
        mConstraintCoefficientsP.resize(tNumConstraints);
        mConstraintCoefficientsQ.resize(tNumConstraints);
        for(IndexType tIndex = 0; tIndex < tNumConstraints; tIndex++)
        {
            mConstraintCoefficientsP[tIndex] = aDataFactory.control().create();
            mConstraintCoefficientsQ[tIndex] = aDataFactory.control().create();
        }

        locus::fill(static_cast<ElementType>(0), mConstraintCoefficientsA.operator*());
        locus::fill(static_cast<ElementType>(1), mConstraintCoefficientsD.operator*());
        locus::fill(static_cast<ElementType>(1e3), mConstraintCoefficientsC.operator*());
    }
    ElementType computeMovingAsymptotesContribution()
    {
        const IndexType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ElementType> tMySumP(tNumVectors);
        std::vector<ElementType> tMySumQ(tNumVectors);

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mControlWorkVectorOne->fill(static_cast<ElementType>(0));
            mControlWorkVectorTwo->fill(static_cast<ElementType>(0));
            const locus::Vector<ElementType, IndexType> & tMyTrialControl = mTrialControl->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyLowerAsymptotes =
                    mLowerAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyUpperAsymptotes =
                    mUpperAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyObjectiveCoefficientsP =
                    mObjectiveCoefficientsP->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyObjectiveCoefficientsQ =
                    mObjectiveCoefficientsQ->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyDualTimesCoefficientsP =
                    mDualTimesCoefficientsP->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyDualTimesCoefficientsQ =
                    mDualTimesCoefficientsQ->operator[](tVectorIndex);

            const IndexType tNumControls = tMyTrialControl.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ElementType tNumerator = tMyObjectiveCoefficientsP[tControlIndex]
                        + tMyDualTimesCoefficientsP[tControlIndex];
                ElementType tDenominator = tMyUpperAsymptotes[tControlIndex] - tMyTrialControl[tControlIndex];
                (*mControlWorkVectorOne)[tControlIndex] = tNumerator / tDenominator;

                tNumerator = tMyObjectiveCoefficientsQ[tControlIndex] + tMyDualTimesCoefficientsQ[tControlIndex];
                tDenominator = tMyTrialControl[tControlIndex] - tMyLowerAsymptotes[tControlIndex];
                (*mControlWorkVectorTwo)[tControlIndex] = tNumerator / tDenominator;
            }

            tMySumP[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorOne.operator*());
            tMySumQ[tVectorIndex] = mControlReductionOperations->sum(mControlWorkVectorTwo.operator*());
        }

        const ElementType tInitialValue = 0;
        const ElementType tGlobalSumP = std::accumulate(tMySumP.begin(), tMySumP.end(), tInitialValue);
        const ElementType tGlobalSumQ = std::accumulate(tMySumQ.begin(), tMySumQ.end(), tInitialValue);
        const ElementType tMovingAsymptotesTerm = tGlobalSumP + tGlobalSumQ;

        return (tMovingAsymptotesTerm);
    }
    // Compute trial controls based on the following explicit expression:
    // \[ x(\lambda)=\frac{u_j^k\mathtt{b}^{1/2}+l_j^k\mathtt{a}^{1/2}}{(\mathtt{a}^{1/2}+\mathtt{b}^{1/2})} \],
    // where
    //      \[ \mathtt{a}=(p_{0j}+\lambda^{\intercal}p_j) \] and [ \mathtt{b}=(q_{0j}+\lambda^{\intercal}q_j) ]
    //      and j=1\dots,n_{x}
    // Here, x denotes the trial control vector
    void computeTrialControl(const locus::MultiVector<ElementType, IndexType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<IndexType>(1));

        locus::update(static_cast<ElementType>(1), *mObjectiveCoefficientsP, static_cast<ElementType>(0), *mTermA);
        locus::update(static_cast<ElementType>(1), *mDualTimesCoefficientsP, static_cast<ElementType>(1), *mTermA);
        locus::update(static_cast<ElementType>(1), *mObjectiveCoefficientsQ, static_cast<ElementType>(0), *mTermB);
        locus::update(static_cast<ElementType>(1), *mDualTimesCoefficientsQ, static_cast<ElementType>(1), *mTermB);

        const IndexType tNumControlVectors = mTrialControl->getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
        {
            locus::Vector<ElementType, IndexType> & tMyTrialControl = mTrialControl->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyLowerAsymptotes =
                    mLowerAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyUpperAsymptotes =
                    mUpperAsymptotes->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyLowerBounds =
                    mTrialControlLowerBounds->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tMyUpperBounds =
                    mTrialControlUpperBounds->operator[](tVectorIndex);

            IndexType tNumControls = tMyTrialControl.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ElementType tSqrtTermA = std::sqrt(mTermA->operator()(tVectorIndex, tControlIndex));
                ElementType tSqrtTermB = std::sqrt(mTermB->operator()(tVectorIndex, tControlIndex));
                ElementType tNumerator = (tMyLowerAsymptotes[tControlIndex] * tSqrtTermA)
                        + (tMyUpperAsymptotes[tControlIndex] * tSqrtTermB);
                ElementType tDenominator = (tSqrtTermA + tSqrtTermB);
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
    void computeTrialAuxiliaryVariables(const locus::MultiVector<ElementType, IndexType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<IndexType>(1));

        IndexType tNumVectors = aDual.getNumVectors();
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ElementType, IndexType> & tMyDual = aDual[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tMyCoefficientsC = (*mConstraintCoefficientsC)[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tMyCoefficientsD = (*mConstraintCoefficientsD)[tVectorIndex];
            locus::Vector<ElementType, IndexType> & tMyAuxiliaryVariablesY = (*mTrialAuxiliaryVariableY)[tVectorIndex];

            const IndexType tNumDual = tMyAuxiliaryVariablesY.size();
            for(IndexType tIndex = 0; tIndex < tNumDual; tIndex++)
            {
                ElementType tDualMinusConstraintCoefficientC = tMyDual[tIndex] - tMyCoefficientsC[tIndex];
                tMyAuxiliaryVariablesY[tIndex] = tDualMinusConstraintCoefficientC / tMyCoefficientsD[tIndex];
                // Project auxiliary variables Y to feasible set (Y >= 0)
                tMyAuxiliaryVariablesY[tIndex] = std::max(tMyAuxiliaryVariablesY[tIndex], static_cast<ElementType>(0));
            }
        }
        ElementType tDualDotConstraintCoefficientA = locus::dot(aDual, *mConstraintCoefficientsA);
        mTrialAuxiliaryVariableZ = (tDualDotConstraintCoefficientA - mObjectiveCoefficientA)
                / (static_cast<ElementType>(2) * mEpsilon);
        // Project auxiliary variables Z to feasible set (Z >= 0)
        mTrialAuxiliaryVariableZ = std::max(mTrialAuxiliaryVariableZ, static_cast<ElementType>(0));
    }
    /*! Compute: \sum_{i=1}^{m}\left( c_iy_i + \frac{1}{2}d_iy_i^2 \right) -
     * \lambda^{T}y - (\lambda^{T}a)z + \lambda^{T}r, where m is the number of constraints
     **/
    ElementType computeConstraintContribution(const locus::MultiVector<ElementType, IndexType> & aDual)
    {
        assert(aDual.getNumVectors() == static_cast<IndexType>(1));

        const IndexType tNumVectors = aDual.getNumVectors();
        std::vector<ElementType> tStorage(tNumVectors);
        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            mDualWorkVector->fill(static_cast<ElementType>(0));
            const locus::Vector<ElementType, IndexType> & tMyDual = aDual[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tMyCoefficientsC = (*mConstraintCoefficientsC)[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tMyCoefficientsD = (*mConstraintCoefficientsD)[tVectorIndex];
            const locus::Vector<ElementType, IndexType> & tMyAuxiliaryVariablesY =
                    (*mTrialAuxiliaryVariableY)[tVectorIndex];

            const IndexType tNumDuals = tMyDual.size();
            for(IndexType tIndex = 0; tIndex < tNumDuals; tIndex++)
            {
                ElementType tValueOne = tMyCoefficientsC[tIndex] * tMyAuxiliaryVariablesY[tIndex];
                ElementType tValueTwo = tMyCoefficientsD[tIndex] * tMyAuxiliaryVariablesY[tIndex]
                        * tMyAuxiliaryVariablesY[tIndex];
                (*mDualWorkVector)[tIndex] = tValueOne + tValueTwo;
            }

            tStorage[tVectorIndex] = mDualReductionOperations->sum(mDualWorkVector.operator*());
        }

        const ElementType tInitialValue = 0;
        ElementType tConstraintSummationTerm = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);

        // Add additional contributions to inequality summation term
        ElementType tDualDotConstraintCoeffR = locus::dot(aDual, *mConstraintCoefficientsR);
        ElementType tDualDotConstraintCoeffA = locus::dot(aDual, *mConstraintCoefficientsA);
        ElementType tDualDotTrialAuxiliaryVariableY = locus::dot(aDual, *mTrialAuxiliaryVariableY);
        ElementType tOutput = tConstraintSummationTerm - tDualDotTrialAuxiliaryVariableY
                - (tDualDotConstraintCoeffA * mTrialAuxiliaryVariableZ) + tDualDotConstraintCoeffR;

        return (tOutput);
    }
    /*
     * Compute \lambda_j\times{p}_j and \lambda_j\times{q}_j, where
     * j=1,\dots,N_{c}. Here, N_{c} denotes the number of constraints.
     **/
    void computeDualTimesConstraintCoefficientTerms(const locus::MultiVector<ElementType, IndexType> & aDual)
    {
        const IndexType tDualVectorIndex = 0;
        locus::fill(static_cast<ElementType>(0), mDualTimesCoefficientsP.operator*());
        locus::fill(static_cast<ElementType>(0), mDualTimesCoefficientsQ.operator*());
        const locus::Vector<ElementType, IndexType> & tDual = aDual[tDualVectorIndex];

        const ElementType tBeta = 1;
        const IndexType tNumConstraints = tDual.size();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            const locus::MultiVector<ElementType, IndexType> & tMyConstraintCoefficientsP =
                    mConstraintCoefficientsP[tConstraintIndex].operator*();
            const locus::MultiVector<ElementType, IndexType> & tMyConstraintCoefficientsQ =
                    mConstraintCoefficientsQ[tConstraintIndex].operator*();

            const IndexType tNumControlVectors = tMyConstraintCoefficientsP.getNumVectors();
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumControlVectors; tVectorIndex++)
            {
                const locus::Vector<ElementType, IndexType> & tMyCoefficientsP =
                        tMyConstraintCoefficientsP[tVectorIndex];
                locus::Vector<ElementType, IndexType> & tMyDualTimesCoefficientsP =
                        (*mDualTimesCoefficientsP)[tVectorIndex];
                tMyDualTimesCoefficientsP.update(tDual[tConstraintIndex], tMyCoefficientsP, tBeta);

                const locus::Vector<ElementType, IndexType> & tMyCoefficientsQ =
                        tMyConstraintCoefficientsQ[tVectorIndex];
                locus::Vector<ElementType, IndexType> & tMyDualTimesCoefficientsQ =
                        (*mDualTimesCoefficientsQ)[tVectorIndex];
                tMyDualTimesCoefficientsQ.update(tDual[tConstraintIndex], tMyCoefficientsQ, tBeta);
            }
        }
    }

private:
    ElementType mEpsilon;
    ElementType mObjectiveCoefficientA;
    ElementType mObjectiveCoefficientR;
    ElementType mTrialAuxiliaryVariableZ;
    ElementType mCurrentObjectiveFunctionValue;

    std::shared_ptr<locus::Vector<ElementType, IndexType>> mDualWorkVector;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkVectorOne;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkVectorTwo;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTermA;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTermB;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mLowerAsymptotes;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mUpperAsymptotes;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDualTimesCoefficientsP;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDualTimesCoefficientsQ;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mObjectiveCoefficientsP;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mObjectiveCoefficientsQ;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialControlLowerBounds;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialControlUpperBounds;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConstraintCoefficientsA;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConstraintCoefficientsC;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConstraintCoefficientsD;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConstraintCoefficientsR;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialAuxiliaryVariableY;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mCurrentConstraintValues;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mDualReductionOperations;
    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mControlReductionOperations;

    std::vector<std::shared_ptr<locus::MultiVector<ElementType, IndexType>>> mConstraintCoefficientsP;
    std::vector<std::shared_ptr<locus::MultiVector<ElementType, IndexType>>> mConstraintCoefficientsQ;

private:
    DualProblemStageMng(const locus::DualProblemStageMng<ElementType, IndexType> & aRhs);
    locus::DualProblemStageMng<ElementType, IndexType> & operator=(const locus::DualProblemStageMng<ElementType, IndexType> & aRhs);
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



template<typename ElementType, typename IndexType = size_t>
class ConservativeConvexSeparableApproximations
{
public:
    explicit ConservativeConvexSeparableApproximations(const locus::ccsa::method_t & aInput) :
            mMethod(aInput),
            mStoppingCriterion(locus::ccsa::stop_t::NOT_CONVERGED),
            mMaxNumIterations(10),
            mNumIterationsDone(0),
            mStagnationTolerance(1e-6),
            mKarushKuhnTuckerConditionsTolerance(1e-6)
    {
    }
    virtual ~ConservativeConvexSeparableApproximations()
    {
    }

    locus::ccsa::method_t  type() const
    {
        return (mMethod);
    }
    locus::ccsa::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    void setStoppingCriterion(const locus::ccsa::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }

    IndexType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    void setNumIterationsDone(const IndexType & aInput)
    {
        mNumIterationsDone = aInput;
    }
    IndexType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setMaxNumIterations(const IndexType & aInput)
    {
        mMaxNumIterations = aInput;
    }

    ElementType getKarushKuhnTuckerConditionsTolerance() const
    {
        return (mKarushKuhnTuckerConditionsTolerance);
    }
    void setKarushKuhnTuckerConditionsTolerance(const ElementType & aInput)
    {
        mKarushKuhnTuckerConditionsTolerance = aInput;
    }
    ElementType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    void setStagnationTolerance(const ElementType & aInput) const
    {
        mStagnationTolerance = aInput;
    }

    virtual void solve(locus::PrimalProblemStageMng<ElementType, IndexType> & aPrimalProblemStageMng,
                       locus::DualProblemStageMng<ElementType, IndexType> & aDualProblemStageMng,
                       locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng) = 0;

private:
    locus::ccsa::method_t mMethod;
    locus::ccsa::stop_t mStoppingCriterion;

    IndexType mMaxNumIterations;
    IndexType mNumIterationsDone;

    ElementType mStagnationTolerance;
    ElementType mKarushKuhnTuckerConditionsTolerance;

private:
    ConservativeConvexSeparableApproximations(const locus::ConservativeConvexSeparableApproximations<ElementType, IndexType> & aRhs);
    locus::ConservativeConvexSeparableApproximations<ElementType, IndexType> & operator=(const locus::ConservativeConvexSeparableApproximations<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class MethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximations<ElementType, IndexType>
{
public:
    explicit MethodMovingAsymptotes(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            locus::ConservativeConvexSeparableApproximations<ElementType, IndexType>(locus::ccsa::method_t::MMA),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mConstraintValues(aDataFactory.dual().create())
    {
    }
    virtual ~MethodMovingAsymptotes()
    {
    }

    void solve(locus::PrimalProblemStageMng<ElementType, IndexType> & aPrimalProblemStageMng,
               locus::DualProblemStageMng<ElementType, IndexType> & aDualProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        // NOTE: REMBER THAT THE GLOBALIZATION FACTORS FOR BOTH OBJECTIVE AND CONSTRAINTS ARE SET TO ZERO IF MMA.
        //       METHOD. THE METHOD SHOULD BE DETECTED AT THE OUTTER LOOP LEVEL AND THE VALUES SET INSIDE INITIALZE.
        //       DEFAULT VALUES ARE SET FOR GCMMA CASE.
        aDualProblemStageMng.update(aDataMng);
        aDualProblemStageMng.updateObjectiveCoefficients(aDataMng);
        aDualProblemStageMng.updateConstraintCoefficients(aDataMng);

        // RESET DUAL SOLVER DATA: mSolver->reset();
        // SOLVE DUAL PROBLEM: solve(aDualProblemStageMng);

        aDualProblemStageMng.getTrialControl(mTrialControl.operator*());
        const locus::MultiVector<ElementType, IndexType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ElementType, IndexType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());
        locus::bounds::computeActiveAndInactiveSets(mTrialControl.operator*(),
                                                    tLowerBounds,
                                                    tUpperBounds,
                                                    mActiveSet.operator*(),
                                                    mInactiveSet.operator*());
        aDataMng.setActiveSet(mActiveSet.operator*());
        aDataMng.setInactiveSet(mInactiveSet.operator*());

        ElementType tObjectiveFunctionValue = aPrimalProblemStageMng.evaluateObjective(mTrialControl.operator*());
        aDataMng.setCurrentObjectiveFunctionValue(tObjectiveFunctionValue);
        aPrimalProblemStageMng.evaluateConstraints(mTrialControl.operator*(), mConstraintValues.operator*());
        aDataMng.setCurrentConstraintValues(mConstraintValues.operator*());

        this->setStoppingCriterion(locus::ccsa::stop_t::MAX_NUMBER_ITERATIONS);
    }

private:
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mConstraintValues;

private:
    MethodMovingAsymptotes(const locus::MethodMovingAsymptotes<ElementType, IndexType> & aRhs);
    locus::MethodMovingAsymptotes<ElementType, IndexType> & operator=(const locus::MethodMovingAsymptotes<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class GloballyConvergentMethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximations<ElementType, IndexType>
{
public:
    explicit GloballyConvergentMethodMovingAsymptotes(const locus::DataFactory<ElementType, IndexType> & aDataFactory) :
            locus::ConservativeConvexSeparableApproximations<ElementType, IndexType>(locus::ccsa::method_t::GCMMA),
            mMinObjectiveGlobalizationFactor(1e-5),
            mCurrentTrialObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
            mPreviousTrialObjectiveFunctionValue(std::numeric_limits<ElementType>::max()),
            mControlWorkOne(),
            mControlWorkTwo(),
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mDeltaControl(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mTrialConstraintValues(aDataFactory.dual().create()),
            mMinConstraintGlobalizationFactors(aDataFactory.dual().create()),
            mControlReductionOperations(aDataFactory.getControlReductionOperations().create())
    {
        this->initialize();
    }
    virtual ~GloballyConvergentMethodMovingAsymptotes()
    {
    }

    void solve(locus::PrimalProblemStageMng<ElementType, IndexType> & aPrimalProblemStageMng,
               locus::DualProblemStageMng<ElementType, IndexType> & aDualProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        aDualProblemStageMng.update(aDataMng);
        IndexType tIterations = 0;
        const IndexType tMaxNumIterations = this->getMaxNumIterations();
        while(tIterations < tMaxNumIterations)
        {
            aDualProblemStageMng.updateObjectiveCoefficients(aDataMng);
            aDualProblemStageMng.updateConstraintCoefficients(aDataMng);

            // RESET DUAL SOLVER DATA: mSolver->reset();
            // SOLVE DUAL PROBLEM: solve(aDualProblemStageMng);

            aDualProblemStageMng.getTrialControl(mTrialControl.operator*());
            const locus::MultiVector<ElementType, IndexType> & tLowerBounds = aDataMng.getControlLowerBounds();
            const locus::MultiVector<ElementType, IndexType> & tUpperBounds = aDataMng.getControlUpperBounds();
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

            const locus::MultiVector<ElementType, IndexType> & tCurrentControl = aDataMng.getCurrentControl();
            locus::update(static_cast<ElementType>(1), *mTrialControl, static_cast<ElementType>(0), *mDeltaControl);
            locus::update(static_cast<ElementType>(-1), tCurrentControl, static_cast<ElementType>(1), *mDeltaControl);
            this->updateObjectiveGlobalizationFactor(aDataMng);
            this->updateConstraintGlobalizationFactors(aDataMng);

            tIterations++;
            if(this->checkStoppingCriteria(aDataMng) == true)
            {
                break;
            }
        }

        this->setNumIterationsDone(tIterations);
        aDataMng.setCurrentObjectiveFunctionValue(mCurrentTrialObjectiveFunctionValue);
        aDataMng.setCurrentConstraintValues(mTrialConstraintValues.operator*());
    }

private:
    void initialize()
    {
        locus::fill(static_cast<ElementType>(1e-5), mMinConstraintGlobalizationFactors.operator*());
        const IndexType tVectorIndex = 0;
        mControlWorkOne = mTrialControl->operator[](tVectorIndex).create();
        mControlWorkTwo = mTrialControl->operator[](tVectorIndex).create();
    }
    void updateObjectiveGlobalizationFactor(locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        locus::fill(static_cast<ElementType>(0), mControlWorkOne.operator*());
        locus::fill(static_cast<ElementType>(0), mControlWorkTwo.operator*());

        const IndexType tNumVectors = mTrialControl->getNumVectors();
        std::vector<ElementType> tStorageOne(tNumVectors, 0);
        std::vector<ElementType> tStorageTwo(tNumVectors, 0);

        for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            locus::Vector<ElementType, IndexType> & tWorkVectorOne = mControlWorkOne->operator[](tVectorIndex);
            locus::Vector<ElementType, IndexType> & tWorkVectorTwo = mControlWorkOne->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
            const locus::Vector<ElementType, IndexType> & tCurrentGradient = aDataMng.getCurrentObjectiveGradient(tVectorIndex);
            assert(tWorkVectorOne.size() == tWorkVectorTwo.size());

            const IndexType tNumControls = tWorkVectorOne.size();
            for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
            {
                ElementType tNumerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                ElementType tDenominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
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

        const ElementType tInitialValue = 0;
        ElementType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
        tFunctionEvaluationW = static_cast<ElementType>(0.5) * tFunctionEvaluationW;
        ElementType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
        const ElementType tCurrentObjectiveValue = aDataMng.getCurrentObjectiveFunctionValue();
        tFunctionEvaluationV = tCurrentObjectiveValue + tFunctionEvaluationV;

        ElementType tGlobalizationFactor = aDataMng.getDualObjectiveGlobalizationFactor();
        const ElementType tCcsaFunctionValue = tFunctionEvaluationV + (tGlobalizationFactor * tFunctionEvaluationW);

        const ElementType tActualOverPredictedReduction = (mCurrentTrialObjectiveFunctionValue - tCcsaFunctionValue)
                / tFunctionEvaluationW;
        if(tActualOverPredictedReduction > static_cast<ElementType>(0))
        {
            ElementType tValueOne = static_cast<ElementType>(10) * tGlobalizationFactor;
            ElementType tValueTwo = static_cast<ElementType>(1.1)
                    * (tGlobalizationFactor + tActualOverPredictedReduction);
            tGlobalizationFactor = std::min(tValueOne, tValueTwo);
        }

        aDataMng.setDualObjectiveGlobalizationFactor(tGlobalizationFactor);
    }
    void updateConstraintGlobalizationFactors(locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        assert(aDataMng.getNumDualVectors() == static_cast<IndexType>(1));
        const IndexType tNumDualVectors = 1;
        const locus::Vector<ElementType, IndexType> & tConstraintValues =
                aDataMng.getCurrentConstraintValues(tNumDualVectors);
        locus::Vector<ElementType, IndexType> & tGlobalizationFactors =
                aDataMng.getConstraintGlobalizationFactors(tNumDualVectors);

        const IndexType tNumConstraints = aDataMng.getNumConstraints();
        for(IndexType tConstraintIndex = 0; tConstraintIndex < tNumConstraints; tConstraintIndex++)
        {
            locus::fill(static_cast<ElementType>(0), mControlWorkOne.operator*());
            locus::fill(static_cast<ElementType>(0), mControlWorkTwo.operator*());

            const IndexType tNumVectors = mTrialControl->getNumVectors();
            std::vector<ElementType> tStorageOne(tNumVectors, 0);
            std::vector<ElementType> tStorageTwo(tNumVectors, 0);
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                locus::Vector<ElementType, IndexType> & tWorkVectorOne = mControlWorkOne->operator[](tVectorIndex);
                locus::Vector<ElementType, IndexType> & tWorkVectorTwo = mControlWorkOne->operator[](tVectorIndex);
                const locus::Vector<ElementType, IndexType> & tDeltaControl = mDeltaControl->operator[](tVectorIndex);
                const locus::Vector<ElementType, IndexType> & tCurrentSigma = aDataMng.getCurrentSigma(tVectorIndex);
                const locus::Vector<ElementType, IndexType> & tConstraintGradient =
                        aDataMng.getCurrentConstraintGradients(tConstraintIndex, tVectorIndex);
                assert(tDeltaControl.size() == tWorkVectorOne.size());
                assert(tWorkVectorOne.size() == tWorkVectorTwo.size());
                assert(tConstraintGradient.size() == tDeltaControl.size());

                const IndexType tNumControls = tWorkVectorOne.size();
                for(IndexType tControlIndex = 0; tControlIndex < tNumControls; tControlIndex++)
                {
                    ElementType numerator = tDeltaControl[tControlIndex] * tDeltaControl[tControlIndex];
                    ElementType denominator = (tCurrentSigma[tControlIndex] * tCurrentSigma[tControlIndex])
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

            const ElementType tInitialValue = 0;
            ElementType tFunctionEvaluationW = std::accumulate(tStorageOne.begin(), tStorageOne.end(), tInitialValue);
            tFunctionEvaluationW = static_cast<ElementType>(0.5) * tFunctionEvaluationW;

            ElementType tFunctionEvaluationV = std::accumulate(tStorageTwo.begin(), tStorageTwo.end(), tInitialValue);
            tFunctionEvaluationV = tFunctionEvaluationV + tConstraintValues[tConstraintIndex];

            ElementType tCcsaFunctionValue = tFunctionEvaluationV + (tGlobalizationFactors[tConstraintIndex] * tFunctionEvaluationW);
            ElementType tActualOverPredictedReduction = (tConstraintValues[tConstraintIndex] - tCcsaFunctionValue) / tFunctionEvaluationW;

            if(tActualOverPredictedReduction > static_cast<ElementType>(0))
            {
                ElementType tValueOne = static_cast<ElementType>(10) * tGlobalizationFactors[tConstraintIndex];
                ElementType tValueTwo = static_cast<ElementType>(1.1)
                        * (tGlobalizationFactors[tConstraintIndex] + tActualOverPredictedReduction);
                tGlobalizationFactors[tConstraintIndex] = std::min(tValueOne, tValueTwo);
            }
        }
    }
    bool checkStoppingCriteria(locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType> & aDataMng)
    {
        bool tStop = false;
        aDataMng.computeKarushKuhnTuckerConditionsInexactness(mTrialControl.operator*(), mTrialDual.operator*());
        const ElementType t_KKT_ConditionsInexactness = aDataMng.getKarushKuhnTuckerConditionsInexactness();

        const ElementType tObjectiveStagnation =
                std::abs(mCurrentTrialObjectiveFunctionValue - mPreviousTrialObjectiveFunctionValue);

        if(t_KKT_ConditionsInexactness < this->getKarushKuhnTuckerConditionsTolerance())
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
    ElementType mMinObjectiveGlobalizationFactor;
    ElementType mCurrentTrialObjectiveFunctionValue;
    ElementType mPreviousTrialObjectiveFunctionValue;

    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkOne;
    std::shared_ptr<locus::Vector<ElementType, IndexType>> mControlWorkTwo;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDeltaControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mTrialConstraintValues;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mMinConstraintGlobalizationFactors;

    std::shared_ptr<locus::ReductionOperations<ElementType, IndexType>> mControlReductionOperations;

private:
    GloballyConvergentMethodMovingAsymptotes(const locus::GloballyConvergentMethodMovingAsymptotes<ElementType, IndexType> & aRhs);
    locus::GloballyConvergentMethodMovingAsymptotes<ElementType, IndexType> & operator=(const locus::GloballyConvergentMethodMovingAsymptotes<ElementType, IndexType> & aRhs);
};

template<typename ElementType, typename IndexType = size_t>
class ConservativeConvexSeparableAppxAlgorithm
{
public:
    ConservativeConvexSeparableAppxAlgorithm(const std::shared_ptr<locus::DualProblemStageMng<ElementType, IndexType>> & aDualProblemStageMng,
                                             const std::shared_ptr<locus::PrimalProblemStageMng<ElementType, IndexType>> & aPrimalProblemStageMng,
                                             const std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType>> & aDataMng,
                                             const std::shared_ptr<locus::ConservativeConvexSeparableApproximations<ElementType, IndexType>> & aSubProblem) :
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
            mDualProblemStageMng(aDualProblemStageMng),
            mPrimalProblemStageMng(aPrimalProblemStageMng),
            mDataMng(aDataMng),
            mSubProblem(aSubProblem)
    {
        this->initialize();
    }
    ~ConservativeConvexSeparableAppxAlgorithm()
    {
    }

    IndexType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setMaxNumIterations(const IndexType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    IndexType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }

    ElementType getOptimalityTolerance() const
    {
        return (mOptimalityTolerance);
    }
    void setOptimalityTolerance(const ElementType & aInput)
    {
        mOptimalityTolerance = aInput;
    }
    ElementType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    void setStagnationTolerance(const ElementType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    ElementType getFeasibilityTolerance() const
    {
        return (mFeasibilityTolerance);
    }
    void setFeasibilityTolerance(const ElementType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }
    ElementType getStationarityTolerance() const
    {
        return (mStationarityTolerance);
    }
    void setStationarityTolerance(const ElementType & aInput)
    {
        mStationarityTolerance = aInput;
    }
    ElementType getObjectiveStagnationTolerance() const
    {
        return (mObjectiveStagnationTolerance);
    }
    void setObjectiveStagnationTolerance(const ElementType & aInput)
    {
        mObjectiveStagnationTolerance = aInput;
    }
    ElementType getMovingAsymptoteExpansionFactor() const
    {
        return (mMovingAsymptoteExpansionFactor);
    }
    void setMovingAsymptoteExpansionFactor(const ElementType & aInput)
    {
        mMovingAsymptoteExpansionFactor = aInput;
    }
    ElementType getMovingAsymptoteContractionFactor() const
    {
        return (mMovingAsymptoteContractionFactor);
    }
    void setMovingAsymptoteContractionFactor(const ElementType & aInput)
    {
        mMovingAsymptoteContractionFactor = aInput;
    }
    ElementType getKarushKuhnTuckerConditionsTolerance() const
    {
        return (mKarushKuhnTuckerConditionsTolerance);
    }
    void setKarushKuhnTuckerConditionsTolerance(const ElementType & aInput)
    {
        mKarushKuhnTuckerConditionsTolerance = aInput;
    }
    ElementType getMovingAsymptoteUpperBoundScaleFactor() const
    {
        return (mMovingAsymptoteUpperBoundScaleFactor);
    }
    void setMovingAsymptoteUpperBoundScaleFactor(const ElementType & aInput)
    {
        mMovingAsymptoteUpperBoundScaleFactor = aInput;
    }
    ElementType getMovingAsymptoteLowerBoundScaleFactor() const
    {
        return (mMovingAsymptoteLowerBoundScaleFactor);
    }
    void setMovingAsymptoteLowerBoundScaleFactor(const ElementType & aInput)
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
        mDualProblemStageMng->initializeAuxiliaryVariables(mDataMng.operator*());
        const locus::MultiVector<ElementType, IndexType> & tControl = mDataMng->getCurrentControl();
        const ElementType tCurrentObjectiveFunctionValue = mPrimalProblemStageMng->evaluateObjective(tControl);
        mDataMng->setCurrentObjectiveFunctionValue(tCurrentObjectiveFunctionValue);
        mPrimalProblemStageMng->evaluateConstraints(tControl, mDualWork.operator*());
        mDataMng->setDual(mDualWork.operator*());

        mNumIterationsDone = 0;
        while(1)
        {
            const locus::MultiVector<ElementType, IndexType> & tCurrentControl = mDataMng->getCurrentControl();
            mPrimalProblemStageMng->computeGradient(tCurrentControl, mControlWork.operator*());
            mDataMng->setCurrentObjectiveGradient(mControlWork.operator*());
            mPrimalProblemStageMng->computeConstraintGradients(tCurrentControl, mWorkMultiVectorList.operator*());
            mDataMng->setCurrentConstraintGradients(mWorkMultiVectorList.operator*());

            if(this->checkStoppingCriteria() == true)
            {
                break;
            }

            this->updateSigmaParameters();

            const locus::MultiVector<ElementType, IndexType> & tPreviousControl = mDataMng->getPreviousControl();
            locus::update(static_cast<ElementType>(1),
                          tPreviousControl,
                          static_cast<ElementType>(0),
                          mAntepenultimateControl.operator*());
            locus::update(static_cast<ElementType>(1),
                          tCurrentControl,
                          static_cast<ElementType>(0),
                          mControlWork.operator*());
            mDataMng->setPreviousControl(mControlWork.operator*());

            mSubProblem->solve(mPrimalProblemStageMng.operator*(),
                               mDualProblemStageMng.operator*(),
                               mDataMng.operator*());

            mNumIterationsDone++;
        }
    }

private:
    void initialize()
    {
        assert(mDataMng.get() != nullptr);

        mDualWork = mDataMng->getDual().create();
        mControlWork = mDataMng->getCurrentControl().create();
        mPreviousSigma = mDataMng->getCurrentControl().create();
        mAntepenultimateControl = mDataMng->getCurrentControl().create();
        mWorkMultiVectorList = mDataMng->getCurrentConstraintGradients().create();
    }
    bool checkStoppingCriteria()
    {
        bool tStop = false;

        mDataMng->computeStagnationMeasure();
        mDataMng->computeFeasibilityMeasure();
        mDataMng->computeStationarityMeasure();
        mDataMng->computeNormProjectedGradient();
        mDataMng->computeObjectiveStagnationMeasure();

        const locus::MultiVector<ElementType, IndexType> & tDual = mDataMng->getDual();
        const locus::MultiVector<ElementType, IndexType> & tControl = mDataMng->getCurrentControl();
        mDataMng->computeKarushKuhnTuckerConditionsInexactness(tControl, tDual);

        const ElementType tStagnationMeasure = mDataMng->getStagnationMeasure();
        const ElementType tFeasibilityMeasure = mDataMng->getFeasibilityMeasure();
        const ElementType tStationarityMeasure = mDataMng->getStationarityMeasure();
        const ElementType tNormProjectedGradient = mDataMng->getNormProjectedGradient();
        const ElementType tObjectiveStagnationMeasure = mDataMng->getObjectiveStagnationMeasure();
        const ElementType t_KKT_ConditionsInexactness = mDataMng->getKarushKuhnTuckerConditionsInexactness();

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

        const locus::MultiVector<ElementType, IndexType> & tCurrentSigma = mDataMng->getCurrentSigma();
        locus::update(static_cast<ElementType>(1), tCurrentSigma, static_cast<ElementType>(0), *mPreviousSigma);

        const IndexType tNumIterationsDone = this->getNumIterationsDone();
        if(tNumIterationsDone < static_cast<IndexType>(2))
        {
            const locus::MultiVector<ElementType, IndexType> & tUpperBounds = mDataMng->getControlUpperBounds();
            locus::update(static_cast<ElementType>(1), tUpperBounds, static_cast<ElementType>(0), *mControlWork);
            const locus::MultiVector<ElementType, IndexType> & tLowerBounds = mDataMng->getControlLowerBounds();
            locus::update(static_cast<ElementType>(-1), tLowerBounds, static_cast<ElementType>(1), *mControlWork);
            locus::scale(static_cast<ElementType>(0.5), mControlWork.operator*());
            mDataMng->setCurrentSigma(mControlWork.operator*());
        }
        else
        {
            const IndexType tNumVectors = mControlWork->getNumVectors();
            locus::fill(static_cast<ElementType>(0), mControlWork.operator*());
            for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
            {
                const ElementType tExpansionFactor = this->getMovingAsymptoteExpansionFactor();
                const ElementType tContractionFactor = this->getMovingAsymptoteContractionFactor();

                const locus::MultiVector<ElementType, IndexType> & tCurrentControl = mDataMng->getCurrentControl();
                const locus::MultiVector<ElementType, IndexType> & tUpperBounds = mDataMng->getControlUpperBounds();
                const locus::MultiVector<ElementType, IndexType> & tLowerBounds = mDataMng->getControlLowerBounds();
                const locus::MultiVector<ElementType, IndexType> & tPreviousControl = mDataMng->getPreviousControl();
                const locus::MultiVector<ElementType, IndexType> & tPreviousSigma = mPreviousSigma->operator[](tVectorIndex);
                const locus::MultiVector<ElementType, IndexType> & tAntepenultimateControl = mAntepenultimateControl->operator[](tVectorIndex);

                const IndexType tNumberControls = mControlWork->operator[](tVectorIndex).size();
                locus::MultiVector<ElementType, IndexType> & tCurrentSigma = mControlWork->operator[](tVectorIndex);
                for(IndexType tControlIndex = 0; tControlIndex < tNumberControls; tControlIndex++)
                {
                    ElementType tValue = (tCurrentControl[tControlIndex] - tPreviousControl[tControlIndex])
                            * (tPreviousControl[tControlIndex] - tAntepenultimateControl[tControlIndex]);
                    if(tValue > static_cast<ElementType>(0))
                    {
                        tCurrentSigma[tControlIndex] = tExpansionFactor * tPreviousSigma[tControlIndex];
                    }
                    else if(tValue < static_cast<ElementType>(0))
                    {
                        tCurrentSigma[tControlIndex] = tContractionFactor * tPreviousSigma[tControlIndex];
                    }
                    else
                    {
                        tCurrentSigma[tControlIndex] = tPreviousSigma[tControlIndex];
                    }
                    // check that lower bound is satisfied
                    const ElementType tLowerBoundScaleFactor = this->getMovingAsymptoteLowerBoundScaleFactor();
                    tValue = tLowerBoundScaleFactor * (tUpperBounds[tControlIndex] - tLowerBounds[tControlIndex]);
                    tCurrentSigma[tControlIndex] = std::max(tValue, tCurrentSigma[tControlIndex]);
                    // check that upper bound is satisfied
                    const ElementType tUpperBoundScaleFactor = this->getMovingAsymptoteUpperBoundScaleFactor();
                    tValue = tUpperBoundScaleFactor * (tUpperBounds[tControlIndex] - tLowerBounds[tControlIndex]);
                    tCurrentSigma[tControlIndex] = std::min(tValue, tCurrentSigma[tControlIndex]);
                }
            }
            mDataMng->setCurrentSigma(mControlWork.operator*());
        }
    }

private:
    IndexType mMaxNumIterations;
    IndexType mNumIterationsDone;

    ElementType mOptimalityTolerance;
    ElementType mStagnationTolerance;
    ElementType mFeasibilityTolerance;
    ElementType mStationarityTolerance;
    ElementType mObjectiveStagnationTolerance;
    ElementType mMovingAsymptoteExpansionFactor;
    ElementType mMovingAsymptoteContractionFactor;
    ElementType mKarushKuhnTuckerConditionsTolerance;
    ElementType mMovingAsymptoteUpperBoundScaleFactor;
    ElementType mMovingAsymptoteLowerBoundScaleFactor;

    locus::ccsa::stop_t mStoppingCriterion;

    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mDualWork;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mPreviousSigma;
    std::shared_ptr<locus::MultiVector<ElementType, IndexType>> mAntepenultimateControl;
    std::shared_ptr<locus::MultiVectorList<ElementType, IndexType>> mWorkMultiVectorList;

    std::shared_ptr<locus::DualProblemStageMng<ElementType, IndexType>> mDualProblemStageMng;
    std::shared_ptr<locus::PrimalProblemStageMng<ElementType, IndexType>> mPrimalProblemStageMng;
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<ElementType, IndexType>> mDataMng;
    std::shared_ptr<locus::ConservativeConvexSeparableApproximations<ElementType, IndexType>> mSubProblem;

private:
    ConservativeConvexSeparableAppxAlgorithm(const locus::ConservativeConvexSeparableAppxAlgorithm<ElementType, IndexType> & aRhs);
    locus::ConservativeConvexSeparableAppxAlgorithm<ElementType, IndexType> & operator=(const locus::ConservativeConvexSeparableAppxAlgorithm<ElementType, IndexType> & aRhs);
};

}

/**********************************************************************************************************/
/*********************************************** UNIT TESTS ***********************************************/
/**********************************************************************************************************/

namespace LocusTest
{

template<typename ElementType, typename IndexType>
void printMultiVector(const locus::MultiVector<ElementType, IndexType> & aInput)
{
    std::cout << "\nPRINT MULTI-VECTOR\n" << std::flush;
    const IndexType tNumVectors = aInput.getNumVectors();
    for(size_t tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        for(size_t tElementIndex = 0; tElementIndex < aInput[tVectorIndex].size(); tElementIndex++)
        {
            std::cout << "VectorIndex = " << tVectorIndex << ", Data(" << tVectorIndex << ", " << tElementIndex
                    << ") = " << aInput(tVectorIndex, tElementIndex) << "\n" << std::flush;
        }
    }
}

template<typename ElementType, typename IndexType>
void checkVectorData(const locus::Vector<ElementType, IndexType> & aInput,
                     const locus::Vector<ElementType, IndexType> & aGold,
                     ElementType aTolerance = 1e-6)
{
    assert(aInput.size() == aGold.size());

    IndexType tNumElements = aInput.size();
    for(IndexType tElemIndex = 0; tElemIndex < tNumElements; tElemIndex++)
    {
        EXPECT_NEAR(aInput[tElemIndex], aGold[tElemIndex], aTolerance);
    }
}

template<typename ElementType, typename IndexType>
void checkMultiVectorData(const locus::MultiVector<ElementType, IndexType> & aInput,
                          const locus::MultiVector<ElementType, IndexType> & aGold,
                          ElementType aTolerance = 1e-6)
{
    assert(aInput.getNumVectors() == aGold.getNumVectors());
    IndexType tNumVectors = aInput.getNumVectors();
    for(IndexType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        IndexType tNumElements = aInput[tVectorIndex].size();
        for(IndexType tElemIndex = 0; tElemIndex < tNumElements; tElemIndex++)
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
    // Default for second template typename is IndexType = size_t
    locus::StandardMultiVector<double> tMultiVector1(tNumVectors, tlocusVector);

    size_t tGoldNumVectors = 8;
    EXPECT_EQ(tMultiVector1.getNumVectors(), tGoldNumVectors);

    double tGoldSum = 0;
    size_t tGoldSize = 10;

    double tTolerance = 1e-6;
    // Default for second template typename is IndexType = size_t
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

    // Default for second template typename is IndexType = size_t
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
    locus::OptimalityCriteriaObjectiveTestOne<double,size_t> tObjective(tInterface);

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
    locus::OptimalityCriteriaInequalityTestOne<double,size_t> tInequality;

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
    tDataMng.setCurrentInequalityValue(tIndex, tValue);
    tGold = 0.345;
    EXPECT_NEAR(tDataMng.getCurrentInequalityValues()[tIndex], tGold, tTolerance);

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

    // ********* Test State Functions *********
    tValue = 0.11;
    locus::StandardMultiVector<double,size_t> tState(tNumVectors, tNumStates, tValue);
    tDataMng.setCurrentState(tState);
    LocusTest::checkMultiVectorData(tDataMng.getCurrentState(), tState, tTolerance);

    tValue = 0.14;
    tState[tVectorIndex].fill(tValue);
    tDataMng.setCurrentState(tVectorIndex, tState[tVectorIndex]);
    LocusTest::checkVectorData(tDataMng.getCurrentState(tVectorIndex), tState.operator [](tVectorIndex));

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
    locus::OptimalityCriteriaInequalityTestOne<double,size_t> tInequality;
    locus::OptimalityCriteriaObjectiveTestOne<double,size_t> tObjective(tReductionOperations);
    locus::OptimalityCriteriaStageMngTypeLP<double,size_t> tStageMng(tObjective, tInequality);

    // ********* Test Update Function *********
    std::vector<double> tData =
        { 6.0368603545003321, 5.2274861883466066, 4.5275369637457814, 3.5304415556495417, 2.1550569529294781 };
    locus::StandardVector<double,size_t> tControl(tData);

    size_t tVectorIndex = 0;
    tDataMng.setCurrentControl(tVectorIndex, tControl);
    tStageMng.updateStage(tDataMng);

    double tTolerance = 1e-6;
    double tGoldValue = 1.3401885069;
    double tObjectiveValue = tDataMng.getCurrentObjectiveValue();
    EXPECT_NEAR(tObjectiveValue, tGoldValue, tTolerance);

    std::fill(tData.begin(), tData.end(), 0.0624);
    locus::StandardVector<double,size_t> tGoldObjectiveGradient(tData);
    LocusTest::checkVectorData(tDataMng.getObjectiveGradient(tVectorIndex), tGoldObjectiveGradient);

    size_t tConstraintIndex = 0;
    tGoldValue = -5.07057774290498e-6;
    tStageMng.evaluateInequality(tConstraintIndex, tDataMng);
    double tInequalityValue = tDataMng.getCurrentInequalityValues(tConstraintIndex);
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

    // NOTE: Default IndexType = size_t
    std::shared_ptr<locus::ReductionOperations<double>> tReductionOperationsCopy = tReductionOperations.create();
    double tSumCopy = tReductionOperationsCopy->sum(tlocusVector);
    EXPECT_NEAR(tSumCopy, tGoldSum, tTolerance);
}

TEST(LocusTest, SynthesisOptimizationSubProblem)
{
    // ********* NOTE: Default IndexType = size_t *********
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
    // ********* NOTE: Default IndexType = size_t *********

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
    locus::OptimalityCriteriaObjectiveTestTwo<double> tObjective;
    locus::OptimalityCriteriaInequalityTestTwo<double> tInequality;
    std::shared_ptr<locus::OptimalityCriteriaStageMngTypeLP<double>> tStageMng =
            std::make_shared<locus::OptimalityCriteriaStageMngTypeLP<double>>(tObjective, tInequality);

    // ********* Allocate Optimality Criteria Algorithm *********
    std::shared_ptr<locus::SingleConstraintTypeLP<double>> tSubProlem =
            std::make_shared<locus::SingleConstraintTypeLP<double>>(*tDataMng);
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
    // Default for second template typename is IndexType = size_t
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
    // Default for second template typename is IndexType = size_t
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
    std::shared_ptr<locus::PreconditionerBase<double>> tCopy = tPreconditioner.create();
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

TEST(LocusTest, SteihaugTointSolverBase)
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

TEST(LocusTest, TrustRegionStepMngBase)
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

TEST(LocusTest, KelleySachsBase)
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
/* ************** METHOD OF MOVING ASYMPTOTES UNIT TESTS ************* */
/* ******************************************************************* */

TEST(LocusTest, DualProblemStageMng)
{
    locus::DataFactory<double> tDataFactory;
    const size_t tNumDuals = 1;
    const size_t tNumControls = 2;
    tDataFactory.allocateDual(tNumDuals);
    tDataFactory.allocateControl(tNumControls);

    std::shared_ptr<locus::MethodMovingAsymptotes<double>> tSubProblem =
            std::make_shared<locus::MethodMovingAsymptotes<double>>(tDataFactory);
    std::shared_ptr<locus::DualProblemStageMng<double>> tDualProblem =
            std::make_shared<locus::DualProblemStageMng<double>>(tDataFactory);
    std::shared_ptr<locus::PrimalProblemStageMng<double>> tPrimalProblem =
            std::make_shared<locus::PrimalProblemStageMng<double>>(tDataFactory);
    std::shared_ptr<locus::ConservativeConvexSeparableAppxDataMng<double>> tDataMng =
            std::make_shared<locus::ConservativeConvexSeparableAppxDataMng<double>>(tDataFactory);

    locus::ConservativeConvexSeparableAppxAlgorithm<double> tAlgorithm(tDualProblem, tPrimalProblem, tDataMng, tSubProblem);
}

}
