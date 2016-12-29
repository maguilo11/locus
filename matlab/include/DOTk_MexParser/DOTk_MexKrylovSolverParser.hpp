/*
 * DOTk_MexKrylovSolverParser.hpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXKRYLOVSOLVERPARSER_HPP_
#define DOTK_MEXKRYLOVSOLVERPARSER_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

namespace mex
{

size_t parseMaxNumKrylovSolverItr(const mxArray* input_);
double parseKrylovSolverFixTolerance(const mxArray* input_);
double parseRelativeToleranceExponential(const mxArray* input_);
double parseKrylovSolverRelativeTolerance(const mxArray* input_);
dotk::types::krylov_solver_t getKrylovSolverMethod(const mxArray* input_);
dotk::types::krylov_solver_t parseKrylovSolverMethod(const mxArray* input_);

}

}

#endif /* DOTK_MEXKRYLOVSOLVERPARSER_HPP_ */
