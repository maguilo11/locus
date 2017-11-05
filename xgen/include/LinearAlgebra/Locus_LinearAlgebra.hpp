/*
 * Locus_LinearAlgebra.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_LINEARALGEBRA_HPP_
#define LOCUS_LINEARALGEBRA_HPP_

#include <cmath>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType>
ScalarType dot(const locus::MultiVector<ScalarType, OrdinalType> & aVectorOne,
               const locus::MultiVector<ScalarType, OrdinalType> & aVectorTwo)
{
    assert(aVectorOne.getNumVectors() > static_cast<OrdinalType>(0));
    assert(aVectorOne.getNumVectors() == aVectorTwo.getNumVectors());

    ScalarType tCummulativeSum = 0;
    OrdinalType tNumVectors = aVectorOne.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aVectorOne[tVectorIndex].size() > static_cast<OrdinalType>(0));
        assert(aVectorOne[tVectorIndex].size() == aVectorTwo[tVectorIndex].size());
        const locus::Vector<ScalarType, OrdinalType> & tVectorOne = aVectorOne[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tVectorTwo = aVectorTwo[tVectorIndex];
        tCummulativeSum += tVectorOne.dot(tVectorTwo);
    }
    return(tCummulativeSum);
}

template<typename ScalarType, typename OrdinalType>
void entryWiseProduct(const locus::MultiVector<ScalarType, OrdinalType> & aInput,
                      locus::MultiVector<ScalarType, OrdinalType> & aOutput)
{
    assert(aInput.getNumVectors() > static_cast<OrdinalType>(0));
    assert(aInput.getNumVectors() == aOutput.getNumVectors());

    OrdinalType tNumVectors = aInput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aInput[tVectorIndex].size() > static_cast<OrdinalType>(0));
        assert(aInput[tVectorIndex].size() == aOutput[tVectorIndex].size());
        locus::Vector<ScalarType, OrdinalType> & tOutput = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tInput = aInput[tVectorIndex];
        tOutput.entryWiseProduct(tInput);
    }
}

template<typename ScalarType, typename OrdinalType>
void fill(const ScalarType & aScalar, locus::MultiVector<ScalarType, OrdinalType> & aOutput)
{
    assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));

    OrdinalType tNumVectors = aOutput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aOutput[tVectorIndex].size() > static_cast<OrdinalType>(0));
        locus::Vector<ScalarType, OrdinalType> & tVector = aOutput[tVectorIndex];
        tVector.fill(aScalar);
    }
}

template<typename ScalarType, typename OrdinalType>
void scale(const ScalarType & aScalar, locus::MultiVector<ScalarType, OrdinalType> & aOutput)
{
    assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));

    OrdinalType tNumVectors = aOutput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aOutput[tVectorIndex].size() > static_cast<OrdinalType>(0));
        locus::Vector<ScalarType, OrdinalType> & tVector = aOutput[tVectorIndex];
        tVector.scale(aScalar);
    }
}

template<typename ScalarType, typename OrdinalType>
ScalarType norm(const locus::MultiVector<ScalarType, OrdinalType> & aInput)
{
    ScalarType tDotProduct = locus::dot(aInput, aInput);
    ScalarType tNorm = std::sqrt(tDotProduct);
    return(tNorm);
}

//! Update vector values with scaled values of A, this = beta*this + alpha*A.
template<typename ScalarType, typename OrdinalType>
void update(const ScalarType & aAlpha,
            const locus::MultiVector<ScalarType, OrdinalType> & aInput,
            const ScalarType & aBeta,
            locus::MultiVector<ScalarType, OrdinalType> & aOutput)
{
    assert(aInput.getNumVectors() > static_cast<OrdinalType>(0));
    assert(aInput.getNumVectors() == aOutput.getNumVectors());

    OrdinalType tNumVectors = aInput.getNumVectors();
    for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
    {
        assert(aInput[tVectorIndex].size() > static_cast<OrdinalType>(0));
        assert(aInput[tVectorIndex].size() == aOutput[tVectorIndex].size());
        locus::Vector<ScalarType, OrdinalType> & tOutputVector = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tInputVector = aInput[tVectorIndex];
        tOutputVector.update(aAlpha, tInputVector, aBeta);
    }
}

template<typename ScalarType, typename OrdinalType>
void gemv(const ScalarType & aAlpha,
          const locus::MultiVector<ScalarType, OrdinalType> & aMatrix,
          const locus::Vector<ScalarType, OrdinalType> & aVector,
          const ScalarType & aBeta,
          locus::Vector<ScalarType, OrdinalType> & aOutput,
          bool aTranspose = false)
{
    const OrdinalType tNumElements = aMatrix[0].size();
    const OrdinalType tNumVectors = aMatrix.getNumVectors();

    if(aTranspose == false)
    {
        assert(tNumVectors == aOutput.size());
        assert(tNumElements == aVector.size());

        aOutput.scale(aBeta);
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            ScalarType tOutputRowValue = aMatrix[tVectorIndex].dot(aVector);
            tOutputRowValue = aAlpha * tOutputRowValue;
            aOutput[tVectorIndex] = aOutput[tVectorIndex] + tOutputRowValue;
        }
    }
    else
    {
        assert(tNumVectors == aVector.size());
        assert(tNumElements == aOutput.size());

        aOutput.scale(aBeta);
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            ScalarType tScaleFactor = aAlpha * aVector[tVectorIndex];
            const locus::Vector<ScalarType, OrdinalType> & tVector = aMatrix[tVectorIndex];
            aOutput.update(tScaleFactor, tVector, static_cast<ScalarType>(1));
        }
    }
}

}

#endif /* LOCUS_LINEARALGEBRA_HPP_ */
