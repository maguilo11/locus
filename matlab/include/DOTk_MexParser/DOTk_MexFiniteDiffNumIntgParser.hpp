/*
 * DOTk_MexFiniteDiffNumIntgParser.hpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXFINITEDIFFNUMINTGPARSER_HPP_
#define DOTK_MEXFINITEDIFFNUMINTGPARSER_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

namespace mex
{

double parseNumericalDifferentiationEpsilon(const mxArray* input_);
dotk::types::numerical_integration_t getFiniteDiffNumIntgMethod(const mxArray* input_);
dotk::types::numerical_integration_t parseNumericalDifferentiationMethod(const mxArray* input_);

}

}

#endif /* DOTK_MEXFINITEDIFFNUMINTGPARSER_HPP_ */
