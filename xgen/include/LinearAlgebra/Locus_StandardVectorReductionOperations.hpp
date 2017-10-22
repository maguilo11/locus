/*
 * Locus_StandardVectorReductionOperations.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_STANDARDVECTORREDUCTIONOPERATIONS_HPP_
#define LOCUS_STANDARDVECTORREDUCTIONOPERATIONS_HPP_

#include <vector>
#include <cassert>
#include <algorithm>

#include "Locus_Vector.hpp"
#include "Locus_ReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class StandardVectorReductionOperations : public locus::ReductionOperations<ScalarType, OrdinalType>
{
public:
    StandardVectorReductionOperations()
    {
    }
    virtual ~StandardVectorReductionOperations()
    {
    }

    //! Returns the maximum element in range
    ScalarType max(const locus::Vector<ScalarType, OrdinalType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ScalarType tValue = 0;
        const OrdinalType tSize = aInput.size();
        std::vector<ScalarType> tCopy(tSize, tValue);
        for(OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ScalarType aMaxValue = *std::max_element(tCopy.begin(), tCopy.end());
        return (aMaxValue);
    }
    //! Returns the minimum element in range
    ScalarType min(const locus::Vector<ScalarType, OrdinalType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ScalarType tValue = 0;
        const OrdinalType tSize = aInput.size();
        std::vector<ScalarType> tCopy(tSize, tValue);
        for(OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ScalarType aMinValue = *std::min_element(tCopy.begin(), tCopy.end());
        return (aMinValue);
    }
    //! Returns the sum of all the elements in container.
    ScalarType sum(const locus::Vector<ScalarType, OrdinalType> & aInput) const
    {
        assert(aInput.size() > 0);

        const ScalarType tValue = 0;
        const OrdinalType tSize = aInput.size();
        std::vector<ScalarType> tCopy(tSize, tValue);
        for(OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
        {
            tCopy[tIndex] = aInput[tIndex];
        }

        ScalarType tBaseValue = 0;
        ScalarType tSum = std::accumulate(tCopy.begin(), tCopy.end(), tBaseValue);
        return (tSum);
    }
    //! Creates an instance of type locus::ReductionOperations
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> tCopy =
                std::make_shared<StandardVectorReductionOperations<ScalarType, OrdinalType>>();
        return (tCopy);
    }

private:
    StandardVectorReductionOperations(const locus::StandardVectorReductionOperations<ScalarType, OrdinalType> &);
    locus::StandardVectorReductionOperations<ScalarType, OrdinalType> & operator=(const locus::StandardVectorReductionOperations<ScalarType, OrdinalType> &);
};

}

#endif /* LOCUS_STANDARDVECTORREDUCTIONOPERATIONS_HPP_ */
