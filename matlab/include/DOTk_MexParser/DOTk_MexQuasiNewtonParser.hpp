/*
 * DOTk_MexQuasiNewtonParser.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXQUASINEWTONPARSER_HPP_
#define DOTK_MEXQUASINEWTONPARSER_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

namespace mex
{

size_t parseQuasiNewtonStorage(const mxArray* input_);
dotk::types::invhessian_t parseQuasiNewtonMethod(const mxArray* input_);

}

}

#endif /* DOTK_MEXQUASINEWTONPARSER_HPP_ */
