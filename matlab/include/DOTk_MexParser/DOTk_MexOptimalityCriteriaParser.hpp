/*
 * DOTk_MexOptimalityCriteriaParser.hpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXOPTIMALITYCRITERIAPARSER_HPP_
#define DOTK_MEXOPTIMALITYCRITERIAPARSER_HPP_

#include <mex.h>

namespace dotk
{

namespace mex
{

double parseOptCriteriaMoveLimit(const mxArray* input_);
double parseOptCriteriaDualLowerBound(const mxArray* input_);
double parseOptCriteriaDualUpperBound(const mxArray* input_);
double parseOptCriteriaDampingParameter(const mxArray* input_);
double parseOptCriteriaBisectionTolerance(const mxArray* input_);

}

}

#endif /* DOTK_MEXOPTIMALITYCRITERIAPARSER_HPP_ */
