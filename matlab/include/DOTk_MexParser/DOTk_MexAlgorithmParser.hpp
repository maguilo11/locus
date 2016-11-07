/*
 * DOTk_MexAlgorithmParser.hpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXALGORITHMPARSER_HPP_
#define DOTK_MEXALGORITHMPARSER_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_VectorTypes.hpp"

namespace dotk
{

template<typename Type>
class vector;

namespace mex
{

void parseThreadCount(const mxArray* options_, size_t & output_);
void parseNumberDuals(const mxArray* options_, size_t & output_);
void parseNumberStates(const mxArray* options_, size_t & output_);
void parseMaxNumUpdates(const mxArray* options_, size_t & output_);
void parseNumberControls(const mxArray* options_, size_t & output_);
void parseMaxNumFeasibleItr(const mxArray* options_, size_t & output_);
void parseMaxNumAlgorithmItr(const mxArray* options_, size_t & output_);
void parseMaxNumLineSearchItr(const mxArray* options_, size_t & output_);
void parseMaxNumTrustRegionSubProblemItr(const mxArray* options_, size_t & output_);

void parseFiniteDifferenceDiagnosticsUpperSuperScripts(const mxArray* options_, int & output_);
void parseFiniteDifferenceDiagnosticsLowerSuperScripts(const mxArray* options_, int & output_);

void parseGradientTolerance(const mxArray* options_, double & output_);
void parseResidualTolerance(const mxArray* options_, double & output_);
void parseObjectiveTolerance(const mxArray* options_, double & output_);
void parseTrialStepTolerance(const mxArray* options_, double & output_);
void parseOptimalityTolerance(const mxArray* options_, double & output_);
void parseFeasibilityTolerance(const mxArray* options_, double & output_);
void parseActualReductionTolerance(const mxArray* options_, double & output_);
void parseControlStagnationTolerance(const mxArray* options_, double & output_);

void parseMaxTrustRegionRadius(const mxArray* options_, double & output_);
void parseMinTrustRegionRadius(const mxArray* options_, double & output_);
void parseBoundConstraintStepSize(const mxArray* options_, double & output_);
void parseInitialTrustRegionRadius(const mxArray* options_, double & output_);
void parseTrustRegionExpansionFactor(const mxArray* options_, double & output_);
void parseGoldsteinLineSearchConstant(const mxArray* options_, double & output_);
void parseLineSearchContractionFactor(const mxArray* options_, double & output_);
void parseTrustRegionContractionFactor(const mxArray* options_, double & output_);
void parseLineSearchStagnationTolerance(const mxArray* options_, double & output_);
void parseBoundConstraintContractionFactor(const mxArray* options_, double & output_);
void parseMinActualOverPredictedReductionRatio(const mxArray* options_, double & output_);
void parseMidActualOverPredictedReductionRatio(const mxArray* options_, double & output_);
void parseMaxActualOverPredictedReductionRatio(const mxArray* options_, double & output_);
void parseSetInitialTrustRegionRadiusToNormGradFlag(const mxArray* options_, bool & output_);

void parseObjectiveFunction(const mxArray* options_, dotk::DOTk_MexArrayPtr & ptr_);
void parseEqualityConstraint(const mxArray* options_, dotk::DOTk_MexArrayPtr & ptr_);
void parseInequalityConstraint(const mxArray* options_, dotk::DOTk_MexArrayPtr & ptr_);

void parseProblemType(const mxArray* options_, dotk::types::problem_t & output_);
void parseContainerType(const mxArray* options_, dotk::types::container_t & output_);
void parseLineSearchMethod(const mxArray* options_, dotk::types::line_search_t & output_);
void parseTrustRegionMethod(const mxArray* options_, dotk::types::trustregion_t & output_);
void parseNonlinearCgMethod(const mxArray* options_, dotk::types::nonlinearcg_t & output_);
void parseHessianComputationMethod(const mxArray* options_, dotk::types::hessian_t & output_);
void parseGradientComputationMethod(const mxArray* options_, dotk::types::gradient_t & output_);
void parseBoundConstraintMethod(const mxArray* options_, dotk::types::constraint_method_t & output_);

void parseDualData(const mxArray* options_, dotk::vector<double> & output_);
void parseStateData(const mxArray* options_, dotk::vector<double> & output_);
void parseControlData(const mxArray* options_, dotk::vector<double> & output_);
void parseDualLowerBound(const mxArray* options_, dotk::vector<double> & output_);
void parseDualUpperBound(const mxArray* options_, dotk::vector<double> & output_);
void parseStateLowerBound(const mxArray* options_, dotk::vector<double> & output_);
void parseStateUpperBound(const mxArray* options_, dotk::vector<double> & output_);
void parseControlLowerBound(const mxArray* options_, dotk::vector<double> & output_);
void parseControlUpperBound(const mxArray* options_, dotk::vector<double> & output_);
void parseFiniteDifferencePerturbation(const mxArray* options_, dotk::vector<double> & output_);

}

}

#endif /* DOTK_MEXALGORITHMPARSER_HPP_ */
