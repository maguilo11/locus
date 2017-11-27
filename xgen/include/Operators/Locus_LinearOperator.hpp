/*
 * Locus_LinearOperator.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_LINEAROPERATOR_HPP_
#define LOCUS_LINEAROPERATOR_HPP_

#include <memory>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class StateData;
template<typename ScalarType, typename OrdinalType>
class MultiVector;

template<typename ScalarType, typename OrdinalType = size_t>
class LinearOperator
{
public:
    virtual ~LinearOperator()
    {
    }

    virtual void update(const locus::StateData<ScalarType, OrdinalType> & aStateData) = 0;
    virtual void apply(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                       const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                       locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> create() const = 0;
};

} // namespace locus

#endif /* LOCUS_LINEAROPERATOR_HPP_ */
