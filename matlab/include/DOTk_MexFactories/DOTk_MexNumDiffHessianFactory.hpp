/*
 * DOTk_MexNumDiffHessianFactory.hpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXNUMDIFFHESSIANFACTORY_HPP_
#define DOTK_MEXNUMDIFFHESSIANFACTORY_HPP_

#include <mex.h>
#include <tr1/memory>

namespace dotk
{

class DOTk_Primal;
class NumericallyDifferentiatedHessian;

namespace mex
{

void buildNumericallyDifferentiatedHessian(const mxArray* options_,
                                           const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> & hessian_);

}

}

#endif /* DOTK_MEXNUMDIFFHESSIANFACTORY_HPP_ */
