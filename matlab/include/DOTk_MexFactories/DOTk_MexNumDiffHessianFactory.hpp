/*
 * DOTk_MexNumDiffHessianFactory.hpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXNUMDIFFHESSIANFACTORY_HPP_
#define DOTK_MEXNUMDIFFHESSIANFACTORY_HPP_

#include <mex.h>
#include <memory>

namespace dotk
{

class NumericallyDifferentiatedHessian;

template<typename SacalarType>
class Vector;

namespace mex
{

void buildNumericallyDifferentiatedHessian(const mxArray* options_,
                                           const Vector<double> & input_,
                                           std::shared_ptr<dotk::NumericallyDifferentiatedHessian> & output_);

}

}

#endif /* DOTK_MEXNUMDIFFHESSIANFACTORY_HPP_ */
