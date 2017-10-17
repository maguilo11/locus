/*
 * Locus_StandardVector.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef PLATO_STANDARDVECTOR_HPP_
#define PLATO_STANDARDVECTOR_HPP_

#include <vector>
#include <cassert>
#include <numeric>

#include "Locus_Vector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class StandardVector : public locus::Vector<ScalarType, OrdinalType>
{
public:
    explicit StandardVector(const std::vector<ScalarType> & aInput) :
            mData(aInput)
    {
    }
    StandardVector(const OrdinalType & aNumElements, ScalarType aValue = 0) :
            mData(std::vector<ScalarType>(aNumElements, aValue))
    {
    }
    virtual ~StandardVector()
    {
    }

    //! Scales a Vector by a ScalarType constant.
    void scale(const ScalarType & aInput)
    {
        OrdinalType tLength = this->size();
        for(OrdinalType tIndex = 0; tIndex < tLength; tIndex++)
        {
            mData[tIndex] = aInput * mData[tIndex];
        }
    }
    //! Element-wise multiplication of two vectors.
    void entryWiseProduct(const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
        OrdinalType tMyDataSize = mData.size();
        assert(aInput.size() == tMyDataSize);

        for(OrdinalType tIndex = 0; tIndex < tMyDataSize; tIndex++)
        {
            mData[tIndex] = aInput[tIndex] * mData[tIndex];
        }
    }
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const ScalarType & aAlpha,
                const locus::Vector<ScalarType, OrdinalType> & aInputVector,
                const ScalarType & aBeta)
    {
        OrdinalType tMyDataSize = mData.size();
        assert(aInputVector.size() == tMyDataSize);
        for(OrdinalType tIndex = 0; tIndex < tMyDataSize; tIndex++)
        {
            mData[tIndex] = aBeta * mData[tIndex] + aAlpha * aInputVector[tIndex];
        }
    }
    //! Computes the absolute value of each element in the container.
    void modulus()
    {
        OrdinalType tLength = this->size();
        for(OrdinalType tIndex = 0; tIndex < tLength; tIndex++)
        {
            mData[tIndex] = std::abs(mData[tIndex]);
        }
    }
    //! Returns the inner product of two vectors.
    ScalarType dot(const locus::Vector<ScalarType, OrdinalType> & aInputVector) const
    {
        assert(aInputVector.size() == static_cast<OrdinalType>(mData.size()));

        const locus::StandardVector<ScalarType, OrdinalType>& tInputVector =
                dynamic_cast<const locus::StandardVector<ScalarType, OrdinalType>&>(aInputVector);

        ScalarType tBaseValue = 0;
        ScalarType tOutput = std::inner_product(mData.begin(), mData.end(), tInputVector.mData.begin(), tBaseValue);
        return (tOutput);
    }
    //! Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & aValue)
    {
        std::fill(mData.begin(), mData.end(), aValue);
    }
    //! Returns the number of local elements in the Vector.
    OrdinalType size() const
    {
        OrdinalType tOutput = mData.size();
        return (tOutput);
    }
    //! Creates object of type locus::Vector
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> create() const
    {
        const ScalarType tBaseValue = 0;
        const OrdinalType tNumElements = this->size();
        std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::StandardVector<ScalarType, OrdinalType>>(tNumElements, tBaseValue);
        return (tOutput);
    }
    //! Operator overloads the square bracket operator
    ScalarType & operator [](const OrdinalType & aIndex)
    {
        assert(aIndex < this->size());
        assert(aIndex >= static_cast<OrdinalType>(0));

        return (mData[aIndex]);
    }
    //! Operator overloads the square bracket operator
    const ScalarType & operator [](const OrdinalType & aIndex) const
    {
        assert(aIndex < this->size());
        assert(aIndex >= static_cast<OrdinalType>(0));

        return (mData[aIndex]);
    }

private:
    std::vector<ScalarType> mData;

private:
    StandardVector(const locus::StandardVector<ScalarType, OrdinalType> &);
    locus::StandardVector<ScalarType, OrdinalType> & operator=(const locus::StandardVector<ScalarType, OrdinalType> &);
};

}

#endif /* PLATO_STANDARDVECTOR_HPP_ */
