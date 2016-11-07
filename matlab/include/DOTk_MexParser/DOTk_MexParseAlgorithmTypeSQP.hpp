/*
 * DOTk_MexParseAlgorithmTypeSQP.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXPARSEALGORITHMTYPESQP_HPP_
#define DOTK_MEXPARSEALGORITHMTYPESQP_HPP_

#include <mex.h>
#include <cstddef>

#include "DOTk_Types.hpp"
#include "DOTk_MexArrayPtr.hpp"

namespace dotk
{

namespace mex
{

void parseSqpMaxNumDualProblemItr(const mxArray* options_, size_t & output_);
void parseSqpMaxNumTangentialProblemItr(const mxArray* options_, size_t & output_);
void parseSqpMaxNumQuasiNormalProblemItr(const mxArray* options_, size_t & output_);
void parseSqpMaxNumTangentialSubProblemItr(const mxArray* options_, size_t & output_);

void parseTangentialTolerance(const mxArray* options_, double & output_);
void parseDualProblemTolerance(const mxArray* options_, double & output_);
void parseDualDotGradientTolerance(const mxArray* options_, double & output_);
void parseToleranceContractionFactor(const mxArray* options_, double & output_);
void parsePredictedReductionParameter(const mxArray* options_, double & output_);
void parseMeritFunctionPenaltyParameter(const mxArray* options_, double & output_);
void parseQuasiNormalProblemRelativeTolerance(const mxArray* options_, double & output_);
void parseTangentialToleranceContractionFactor(const mxArray* options_, double & output_);
void parseActualOverPredictedReductionTolerance(const mxArray* options_, double & output_);
void parseMaxEffectiveTangentialOverTrialStepRatio(const mxArray* options_, double & output_);
void parseTangentialSubProbLeftPrecProjectionTolerance(const mxArray* options_, double & output_);
void parseQuasiNormalProblemTrustRegionRadiusPenaltyParameter(const mxArray* options_, double & output_);

}

}

#endif /* DOTK_MEXPARSEALGORITHMTYPESQP_HPP_ */
