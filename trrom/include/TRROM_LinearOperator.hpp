/*
 * TRROM_LinearOperator.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_LINEAROPERATOR_HPP_
#define TRROM_LINEAROPERATOR_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class LinearOperator
{
public:
    virtual ~LinearOperator()
    {
    }

    virtual trrom::types::linear_operator_t type() const = 0;
    virtual void update(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_) = 0;
    virtual void apply(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                       const std::tr1::shared_ptr<trrom::Vector<double> > & output_) = 0;
};

}

#endif /* TRROM_LINEAROPERATOR_HPP_ */
