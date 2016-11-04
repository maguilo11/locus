/*
 * DOTk_MexOptimalityCriteriaParser.hpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXOPTIMALITYCRITERIAPARSER_HPP_
#define DOTK_MEXOPTIMALITYCRITERIAPARSER_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"
#include "DOTk_MexArrayPtr.hpp"

namespace dotk
{

namespace mex
{

void parseOptCriteriaMoveLimit(const mxArray* options_, double & output_);
void parseOptCriteriaDualLowerBound(const mxArray* options_, double & output_);
void parseOptCriteriaDualUpperBound(const mxArray* options_, double & output_);
void parseOptCriteriaDampingParameter(const mxArray* options_, double & output_);
void parseOptCriteriaBisectionTolerance(const mxArray* options_, double & output_);

}

}

#endif /* DOTK_MEXOPTIMALITYCRITERIAPARSER_HPP_ */
