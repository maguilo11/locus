/*
 * DOTk_MexHessianFactory.hpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXHESSIANFACTORY_HPP_
#define DOTK_MEXHESSIANFACTORY_HPP_

#include <mex.h>

namespace dotk
{

class DOTk_Primal;
class DOTk_Hessian;

namespace mex
{

void buildHessian(const mxArray* options_,
                  const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                  std::tr1::shared_ptr<dotk::DOTk_Hessian> & hessian_);

}

}

#endif /* DOTK_MEXHESSIANFACTORY_HPP_ */
