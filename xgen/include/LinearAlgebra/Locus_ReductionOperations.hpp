/*
 * Locus_ReductionOperations.hpp
 *
 *  Created on: Oct 6, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_REDUCTIONOPERATIONS_HPP_
#define LOCUS_REDUCTIONOPERATIONS_HPP_

#include <memory>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class Vector;

template<typename ScalarType, typename OrdinalType = size_t>
class ReductionOperations
{
public:
    virtual ~ReductionOperations()
    {
    }

    //! Returns the maximum element in range
    virtual ScalarType max(const locus::Vector<ScalarType, OrdinalType> & aInput) const = 0;
    //! Returns the minimum element in range
    virtual ScalarType min(const locus::Vector<ScalarType, OrdinalType> & aInput) const = 0;
    //! Returns the sum of all the elements in container.
    virtual ScalarType sum(const locus::Vector<ScalarType, OrdinalType> & aInput) const = 0;
    //! Creates object of type locus::ReductionOperations
    virtual std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> create() const = 0;
};

}

#endif /* LOCUS_REDUCTIONOPERATIONS_HPP_ */
