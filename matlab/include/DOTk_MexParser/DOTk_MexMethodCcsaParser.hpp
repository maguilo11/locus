/*
 * DOTk_MexMethodCcsaParser.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXMETHODCCSAPARSER_HPP_
#define DOTK_MEXMETHODCCSAPARSER_HPP_

#include <mex.h>
#include "DOTk_UtilsCCSA.hpp"

namespace dotk
{

namespace mex
{

void parseMovingAsymptoteUpperBoundScale(const mxArray* options_, double & output_);
void parseMovingAsymptoteLowerBoundScale(const mxArray* options_, double & output_);
void parseMovingAsymptoteExpansionParameter(const mxArray* options_, double & output_);
void parseMovingAsymptoteContractionParameter(const mxArray* options_, double & output_);

void parseDualSolverGradientTolerance(const mxArray* options_, double & output_);
void parseDualSolverTrialStepTolerance(const mxArray* options_, double & output_);
void parseDualSolverMaxNumberIterations(const mxArray* options_, size_t & output_);
void parseDualObjectiveEpsilonParameter(const mxArray* options_, double & output_);
void parseDualSolverType(const mxArray* options_, dotk::ccsa::dual_solver_t & output_);
void parseDualObjectiveTrialControlBoundScaling(const mxArray* options_, double & output_);
void parseDualSolverObjectiveStagnationTolerance(const mxArray* options_, double & output_);
void parseDualSolverTypeNLCG(const mxArray* options_, dotk::types::nonlinearcg_t & output_);

void parseLineSearchStepLowerBound(const mxArray* options_, double & output_);
void parseLineSearchStepUpperBound(const mxArray* options_, double & output_);
void parseSubProblemResidualTolerance(const mxArray* options_, double & output_);
void parseSubProblemStagnationTolerance(const mxArray* options_, double & output_);
void parseMaxNumberSubProblemIterations(const mxArray* options_, size_t & output_);

}

}

#endif /* DOTK_MEXMETHODCCSAPARSER_HPP_ */
