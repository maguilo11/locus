/*
 * DOTk_MexMethodCcsaParser.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXMETHODCCSAPARSER_HPP_
#define DOTK_MEXMETHODCCSAPARSER_HPP_

#include <mex.h>
#include <cstddef>

#include "DOTk_UtilsCCSA.hpp"

namespace dotk
{

namespace mex
{

//! @name Free functions that return scalar type
//@{
/*!
 * Parses lower bound on line search step (dual problem only).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseLineSearchStepLowerBound(const mxArray* input_);
/*!
 * Parses upper bound on line search step (dual problem only).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseLineSearchStepUpperBound(const mxArray* input_);
/*!
 * Parses convex conservative separable approximation (CCSA)
 * subproblem tolerance on the norm of the Karush-Kuhn-Tucker
 * (KKT) residual.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseSubProblemResidualTolerance(const mxArray* input_);
/*!
 * Parses tolerance on the norm of the gradient for the dual problem.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseDualSolverGradientTolerance(const mxArray* input_);
/*!
 * Parses tolerance on the norm of the step for the dual problem.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseDualSolverStepTolerance(const mxArray* input_);
/*!
 * Parses the CCSA subproblem stagnation tolerance.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseSubProblemStagnationTolerance(const mxArray* input_);
/*!
 * Parses relaxation parameter for the dual objective function.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseDualObjectiveRelaxationParameter(const mxArray* input_);
/*!
 * Parses upper bound on the moving asymptotes.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseMovingAsymptoteUpperBoundScale(const mxArray* input_);
/*!
 * Parses lower bound on the moving asymptotes.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseMovingAsymptoteLowerBoundScale(const mxArray* input_);
/*!
 * Parses moving asymptotes' expansion parameter.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseMovingAsymptoteExpansionParameter(const mxArray* input_);
/*!
 * Parses moving asymptotes' contraction parameter.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseMovingAsymptoteContractionParameter(const mxArray* input_);
/*!
 * Parses scaling factor on the trial control (dual problem specific).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseDualObjectiveControlBoundsScaling(const mxArray* input_);
/*!
 * Parses stagnation tolerance on the dual objective function.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return value.
 **/
double parseDualSolverObjectiveStagnationTolerance(const mxArray* input_);
//@}

//! @name Free functions that return integer type
//@{
/*!
 * Parses the maximum number of outer iterations for the dual problem.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of iterations.
 **/
size_t parseDualSolverMaxNumberIterations(const mxArray* input_);
/*!
 * Parses the maximum number convex conservative separable approximation
 * (CCSA) subproblem iterations.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of iterations.
 **/
size_t parseMaxNumberSubProblemIterations(const mxArray* input_);
//@}

//! @name Free functions that return dotk::flag type
//@{
/*!
 * Parses dual solver type (dual problem specific).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::ccsa::dual_solver_t parseDualSolverType(const mxArray* input_);
/*!
 * Parses nonlinear conjugate gradient type (dual problem specific).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::nonlinearcg_t parseDualSolverTypeNLCG(const mxArray* input_);
}

}

#endif /* DOTK_MEXMETHODCCSAPARSER_HPP_ */
