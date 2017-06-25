/*
 * TRROM_ReducedHessian.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_REDUCEDHESSIAN_HPP_
#define TRROM_REDUCEDHESSIAN_HPP_

#include "TRROM_LinearOperator.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class OptimizationDataMng;

class ReducedHessian : public trrom::LinearOperator
{
public:
    ReducedHessian();
    virtual ~ReducedHessian();

    virtual trrom::types::linear_operator_t type() const;
    virtual void update(const std::shared_ptr<trrom::OptimizationDataMng> & mng_);
    virtual void apply(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                       const std::shared_ptr<trrom::Vector<double> > & input_,
                       const std::shared_ptr<trrom::Vector<double> > & output_);

private:
    ReducedHessian(const trrom::ReducedHessian &);
    trrom::ReducedHessian & operator=(const trrom::ReducedHessian &);
};

}

#endif /* TRROM_REDUCEDHESSIAN_HPP_ */
