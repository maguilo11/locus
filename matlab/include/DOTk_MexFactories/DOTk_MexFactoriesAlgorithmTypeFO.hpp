/*
 * DOTk_MexFactoriesAlgorithmTypeFO.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXFACTORIESALGORITHMTYPEFO_HPP_
#define DOTK_MEXFACTORIESALGORITHMTYPEFO_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_LineSearchQuasiNewton;

namespace mex
{

void buildQuasiNewtonMethod(const mxArray* options_,dotk::DOTk_LineSearchQuasiNewton & algorithm_);

}

}

#endif /* DOTK_MEXFACTORIESALGORITHMTYPEFO_HPP_ */
