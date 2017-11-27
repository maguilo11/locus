/*
 * Locus_GradientOperator.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_GRADIENTOPERATOR_HPP_
#define LOCUS_GRADIENTOPERATOR_HPP_

#include <memory>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class StateData;
template<typename ScalarType, typename OrdinalType>
class MultiVector;

template<typename ScalarType, typename OrdinalType = size_t>
class GradientOperator
{
public:
    virtual ~GradientOperator()
    {
    }

    virtual void update(const locus::StateData<ScalarType, OrdinalType> & aStateData) = 0;
    virtual void compute(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                         locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> create() const = 0;
};

} // namespace locus

#endif /* LOCUS_GRADIENTOPERATOR_HPP_ */
