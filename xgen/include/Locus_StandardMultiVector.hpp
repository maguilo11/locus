/*
 * Locus_StandardMultiVector.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_STANDARDMULTIVECTOR_HPP_
#define LOCUS_STANDARDMULTIVECTOR_HPP_

#include <vector>
#include <cassert>

#include "Locus_MultiVector.hpp"
#include "Locus_StandardVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class StandardMultiVector : public locus::MultiVector<ScalarType, OrdinalType>
{
public:
    StandardMultiVector(const OrdinalType & aNumVectors, const std::vector<ScalarType> & aStandardVectorTemplate) :
        mData(std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>>(aNumVectors))
    {
        this->initialize(aStandardVectorTemplate);
    }
    StandardMultiVector(const OrdinalType & aNumVectors, const locus::Vector<ScalarType, OrdinalType> & aVectorTemplate) :
        mData(std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>>(aNumVectors))
    {
        this->initialize(aVectorTemplate);
    }
    explicit StandardMultiVector(const std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>>& aMultiVectorTemplate) :
        mData(std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>>(aMultiVectorTemplate.size()))
    {
        this->initialize(aMultiVectorTemplate);
    }
    StandardMultiVector(const OrdinalType & aNumVectors, const OrdinalType & aNumElementsPerVector, ScalarType aValue = 0) :
        mData(std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>>(aNumVectors))
    {
        this->initialize(aNumElementsPerVector, aValue);
    }
    virtual ~StandardMultiVector()
    {
    }

    //! Creates a copy of type MultiVector
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> create() const
    {
        const OrdinalType tVectorIndex = 0;
        const OrdinalType tNumVectors = this->getNumVectors();
        std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> tOutput;
        const locus::Vector<ScalarType, OrdinalType> & tVectorTemplate = *mData[tVectorIndex];
        tOutput = std::make_shared<locus::StandardMultiVector<ScalarType, OrdinalType>>(tNumVectors, tVectorTemplate);
        return (tOutput);
    }
    //! Number of vectors
    OrdinalType getNumVectors() const
    {
        OrdinalType tNumVectors = mData.size();
        return (tNumVectors);
    }
    //! Operator overloads the square bracket operator
    virtual locus::Vector<ScalarType, OrdinalType> & operator [](const OrdinalType & aVectorIndex)
    {
        assert(mData.empty() == false);
        assert(aVectorIndex < this->getNumVectors());

        return (mData[aVectorIndex].operator *());
    }
    //! Operator overloads the square bracket operator
    virtual const locus::Vector<ScalarType, OrdinalType> & operator [](const OrdinalType & aVectorIndex) const
    {
        assert(mData.empty() == false);
        assert(mData[aVectorIndex].get() != nullptr);
        assert(aVectorIndex < this->getNumVectors());

        return (mData[aVectorIndex].operator *());
    }
    //! Operator overloads the square bracket operator
    virtual ScalarType & operator ()(const OrdinalType & aVectorIndex, const OrdinalType & aElementIndex)
    {
        assert(aVectorIndex < this->getNumVectors());
        assert(aElementIndex < mData[aVectorIndex]->size());

        return (mData[aVectorIndex].operator *().operator [](aElementIndex));
    }
    //! Operator overloads the square bracket operator
    virtual const ScalarType & operator ()(const OrdinalType & aVectorIndex, const OrdinalType & aElementIndex) const
    {
        assert(aVectorIndex < this->getNumVectors());
        assert(aElementIndex < mData[aVectorIndex]->size());

        return (mData[aVectorIndex].operator *().operator [](aElementIndex));
    }

private:
    void initialize(const OrdinalType & aNumElementsPerVector, const ScalarType & aValue)
    {
        locus::StandardVector<ScalarType,OrdinalType> tVector(aNumElementsPerVector);

        OrdinalType tNumVectors = mData.size();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            mData[tIndex] = tVector.create();
            mData[tIndex]->fill(aValue);
        }
    }
    void initialize(const std::vector<ScalarType> & aVectorTemplate)
    {
        locus::StandardVector<ScalarType,OrdinalType> tVector(aVectorTemplate);

        OrdinalType tNumVectors = mData.size();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            mData[tIndex] = tVector.create();
        }
    }
    void initialize(const locus::Vector<ScalarType, OrdinalType> & aVectorTemplate)
    {
        OrdinalType tNumVectors = mData.size();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            mData[tIndex] = aVectorTemplate.create();
        }
    }
    void initialize(const std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>> & aMultiVectorTemplate)
    {
        assert(mData.size() > 0);
        assert(aMultiVectorTemplate.size() > 0);
        OrdinalType tNumVectors = aMultiVectorTemplate.size();
        for(OrdinalType tIndex = 0; tIndex < tNumVectors; tIndex++)
        {
            assert(aMultiVectorTemplate[tIndex]->size() > 0);
            mData[tIndex] = aMultiVectorTemplate[tIndex]->create();
            mData[tIndex]->update(static_cast<ScalarType>(1.), *aMultiVectorTemplate[tIndex], static_cast<ScalarType>(0.));
        }
    }

private:
    std::vector<std::shared_ptr<locus::Vector<ScalarType, OrdinalType>>> mData;

private:
    StandardMultiVector(const locus::StandardMultiVector<ScalarType, OrdinalType>&);
    locus::StandardMultiVector<ScalarType, OrdinalType> & operator=(const locus::StandardMultiVector<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_STANDARDMULTIVECTOR_HPP_ */
