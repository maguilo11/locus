/*
 * DOTk_MexQuasiNewtonParser.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXQUASINEWTONPARSER_HPP_
#define DOTK_MEXQUASINEWTONPARSER_HPP_

#include <mex.h>
#include <cstddef>

#include "DOTk_Types.hpp"
#include "DOTk_MexArrayPtr.hpp"

namespace dotk
{

namespace mex
{

void parseQuasiNewtonStorage(const mxArray* options_, size_t & output_);
void parseQuasiNewtonMethod(const mxArray* options_, dotk::types::invhessian_t & output_);

}

}

#endif /* DOTK_MEXQUASINEWTONPARSER_HPP_ */
