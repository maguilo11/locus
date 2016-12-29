/*
 * DOTk_MexParseAlgorithmTypeSQP.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXPARSEALGORITHMTYPESQP_HPP_
#define DOTK_MEXPARSEALGORITHMTYPESQP_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

namespace mex
{

//! @name Free functions that return integer type
//@{
/*!
 * Parses the maximum number of dual problem iterations.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return iterations.
 **/
size_t parseSqpMaxNumDualProblemItr(const mxArray* input_);
/*!
 * Parses the maximum number of tangential problem iterations.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return iterations.
 **/
size_t parseSqpMaxNumTangentialProblemItr(const mxArray* input_);
/*!
 * Parses the maximum number of quasi-normal problem iterations.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return iterations.
 **/
size_t parseSqpMaxNumQuasiNormalProblemItr(const mxArray* input_);
/*!
 * Parses the maximum number of tangential subproblem iterations.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return iterations.
 **/
size_t parseSqpMaxNumTangentialSubProblemItr(const mxArray* input_);
//@}

//! @name Free functions that return scalar type
//@{
/*!
 * Parses tolerance on the norm of the tangential step.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseTangentialTolerance(const mxArray* input_);
/*!
 * Parses tolerance on dual problem.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseDualProblemTolerance(const mxArray* input_);
/*!
 * Parses tolerance on dual problem inexactness.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseDualDotGradientTolerance(const mxArray* input_);
/*!
 * Parses contraction parameter on stopping tolerances.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseToleranceContractionFactor(const mxArray* input_);
/*!
 * Parses penalty parameter on predicted reduction.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parsePredictedReductionParameter(const mxArray* input_);
/*!
 * Parses merit function's penalty parameter.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseMeritFunctionPenaltyParameter(const mxArray* input_);
/*!
 * Parses relative tolerance on quasi-normal problem.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseQuasiNormalProblemRelativeTolerance(const mxArray* input_);
/*!
 * Parses contraction parameter on tangential tolerances.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseTangentialToleranceContractionFactor(const mxArray* input_);
/*!
 * Parses tolerance on actual over predicted reduction ratio.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseActualOverPredictedReductionTolerance(const mxArray* input_);
/*!
 * Parses upper bound on tangential step norm over trial step norm ratio.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseMaxEffectiveTangentialOverTrialStepRatio(const mxArray* input_);
/*!
 * Parses projection tolerance on left preconditioner times tangential step
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseTangentialSubProbLeftPrecProjectionTolerance(const mxArray* input_);
/*!
 * Parses trust region radius penalty parameter for quasi-normal problem
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseQuasiNormalProblemTrustRegionRadiusPenaltyParameter(const mxArray* input_);
//@}

}

}

#endif /* DOTK_MEXPARSEALGORITHMTYPESQP_HPP_ */
