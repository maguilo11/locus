/*
 * Locus_UnitTestUtils.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_UNITTESTUTILS_HPP_
#define LOCUS_UNITTESTUTILS_HPP_

#include <cassert>
#include <iostream>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"

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

} // namespace LocusTest

#endif /* LOCUS_UNITTESTUTILS_HPP_ */
