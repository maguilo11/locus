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

class DOTk_MexArrayPtr;

namespace mex
{

dotk::types::krylov_solver_t getKrylovSolverMethod(dotk::DOTk_MexArrayPtr & ptr_);

void parseMaxNumKrylovSolverItr(const mxArray* options_, size_t & output_);
void parseKrylovSolverFixTolerance(const mxArray* options_, double & output_);
void parseRelativeToleranceExponential(const mxArray* options_, double & output_);
void parseKrylovSolverRelativeTolerance(const mxArray* options_, double & output_);
void parseKrylovSolverMethod(const mxArray* options_, dotk::types::krylov_solver_t & output_);

}

}

#endif /* DOTK_MEXKRYLOVSOLVERPARSER_HPP_ */
