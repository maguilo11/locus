/*
 * Locus_Bounds.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_BOUNDS_HPP_
#define LOCUS_BOUNDS_HPP_

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"

namespace locus
{

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
void computeProjectedVector(const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                            const locus::MultiVector<ScalarType, OrdinalType> & aCurrentControl,
                            locus::MultiVector<ScalarType, OrdinalType> & aProjectedVector)
{
    assert(aVector.getNumVectors() == aCurrentControl.getNumVectors());
    assert(aCurrentControl.getNumVectors() == aProjectedVector.getNumVectors());

    locus::update(static_cast<ScalarType>(1), aVector, static_cast<ScalarType>(0), aProjectedVector);
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

}

#endif /* LOCUS_BOUNDS_HPP_ */
