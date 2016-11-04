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

class DOTk_MexArrayPtr;

namespace mex
{

void parseNumericalDifferentiationEpsilon(const mxArray* options_, double & output_);
void parseNumericalDifferentiationMethod(const mxArray* options_, dotk::types::numerical_integration_t & output_);

dotk::types::numerical_integration_t getFiniteDiffNumIntgMethod(dotk::DOTk_MexArrayPtr & ptr_);

}

}

#endif /* DOTK_MEXFINITEDIFFNUMINTGPARSER_HPP_ */
