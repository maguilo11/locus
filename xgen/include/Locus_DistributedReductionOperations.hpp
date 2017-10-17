/*
 * Locus_DistributedReductionOperations.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_DISTRIBUTEDREDUCTIONOPERATIONS_HPP_
#define LOCUS_DISTRIBUTEDREDUCTIONOPERATIONS_HPP_

#include <mpi.h>

#include <vector>
#include <cassert>
#include <algorithm>

#include "Locus_Vector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class DistributedReductionOperations : public locus::ReductionOperations<ScalarType, OrdinalType>
{
public:
    DistributedReductionOperations()
    {
    }
    virtual ~DistributedReductionOperations()
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
        ScalarType aLocalMaxValue = *std::max_element(tCopy.begin(), tCopy.end());

        ScalarType aGlobalMaxValue = aLocalMaxValue;
        MPI_Allreduce(&aGlobalMaxValue, &aLocalMaxValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        return (aLocalMaxValue);
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
        ScalarType aLocalMinValue = *std::min_element(tCopy.begin(), tCopy.end());

        ScalarType aGlobalMinValue = aLocalMinValue;
        MPI_Allreduce(&aGlobalMinValue, &aLocalMinValue, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        return (aGlobalMinValue);
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
        ScalarType tLocalSum = std::accumulate(tCopy.begin(), tCopy.end(), tBaseValue);

        ScalarType tGlobalSum = tLocalSum;
        MPI_Allreduce(&tGlobalSum, &tLocalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return (tLocalSum);
    }
    //! Creates an instance of type locus::ReductionOperations
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> tCopy =
                std::make_shared<DistributedReductionOperations<ScalarType, OrdinalType>>();
        return (tCopy);
    }
    //! Return number of ranks (i.e. processes)
    OrdinalType getNumRanks() const
    {
        int tNumRanks = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &tNumRanks);
        assert(tNumRanks > static_cast<int>(0));
        return (tNumRanks);
    }

private:
    DistributedReductionOperations(const locus::DistributedReductionOperations<ScalarType, OrdinalType> &);
    locus::DistributedReductionOperations<ScalarType, OrdinalType> & operator=(const locus::DistributedReductionOperations<ScalarType, OrdinalType> &);
};

}

#endif /* LOCUS_DISTRIBUTEDREDUCTIONOPERATIONS_HPP_ */
