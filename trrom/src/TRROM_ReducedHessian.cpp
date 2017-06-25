/*
 * TRROM_ReducedHessian.cpp
 *
 *  Created on: Aug 18, 2016
 */

#include <limits>

#include "TRROM_Vector.hpp"
#include "TRROM_ReducedHessian.hpp"
#include "TRROM_AssemblyManager.hpp"
#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

ReducedHessian::ReducedHessian()
{
}

ReducedHessian::~ReducedHessian()
{
}

trrom::types::linear_operator_t ReducedHessian::type() const
{
    return (trrom::types::REDUCED_HESSIAN);
}

void ReducedHessian::update(const std::shared_ptr<trrom::OptimizationDataMng> & mng_)
{
    return;
}

void ReducedHessian::apply(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                           const std::shared_ptr<trrom::Vector<double> > & input_,
                           const std::shared_ptr<trrom::Vector<double> > & output_)
{
    mng_->applyVectorToHessian(input_, output_);
}

}
