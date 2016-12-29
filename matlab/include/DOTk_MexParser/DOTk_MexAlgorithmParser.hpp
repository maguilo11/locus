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

namespace dotk
{

namespace mex
{

//! @name Free functions that return integer type
//@{
/*!
 * Parses the length of the array of dual variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of dual variables.
 **/
size_t parseNumberDuals(const mxArray* input_);
/*!
 * Parses the length of the array of the state variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of state variables.
 **/
size_t parseNumberStates(const mxArray* input_);
/*!
 * Parses number of line search iterations during primal variables
 *  update routine inside the Kelley-Sachs trust region algorithm
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of updates.
 **/
size_t parseMaxNumUpdates(const mxArray* input_);
/*!
 * Parses the length of the array of control variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of control variables.
 **/
size_t parseNumberControls(const mxArray* input_);
/*!
 * Parses maximum number of feasible iterations (bound constraint problem)
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of feasible iterations.
 **/
size_t parseMaxNumFeasibleItr(const mxArray* input_);
/*!
 * Parses maximum number of outer optimization iterations
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of outer optimization iterations.
 **/
size_t parseMaxNumOuterIterations(const mxArray* input_);
/*!
 * Parses maximum number of line search iterations
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number line search iterations.
 **/
size_t parseMaxNumLineSearchItr(const mxArray* input_);
/*!
 * Parses maximum number of trust region subproblem iterations
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number trust region subproblem iterations.
 **/
size_t parseMaxNumTrustRegionSubProblemItr(const mxArray* input_);
/*!
 * Parses finite difference diagnostics upper bound on super script
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return upper bound on super script.
 **/
int parseFiniteDifferenceDiagnosticsUpperSuperScripts(const mxArray* input_);
/*!
 * Parses finite difference diagnostics lower bound on super script
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return lower bound on super script.
 **/
int parseFiniteDifferenceDiagnosticsLowerSuperScripts(const mxArray* input_);
//@}

//! @name Free functions that return scalar type
//@{
/*!
 * Parses tolerance on the norm of the gradient.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseGradientTolerance(const mxArray* input_);
/*!
 * Parses tolerance on norm of the residual.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseResidualTolerance(const mxArray* input_);
/*!
 * Parses tolerance on objective function.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseObjectiveTolerance(const mxArray* input_);
/*!
 * Parses tolerance on the norm of the step.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseStepTolerance(const mxArray* input_);
/*!
 * Parses tolerance on feasibility.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseFeasibilityTolerance(const mxArray* input_);
/*!
 * Parses tolerance on actual reduction (trust region algorithm only).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseActualReductionTolerance(const mxArray* input_);
/*!
 * Parses tolerance on stagnation between two subsequent solutions
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseControlStagnationTolerance(const mxArray* input_);
/*!
 * Parses maximum trust region radius
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return upper bound.
 **/
double parseMaxTrustRegionRadius(const mxArray* input_);
/*!
 * Parses minimum trust region radius
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return lower bound.
 **/
double parseMinTrustRegionRadius(const mxArray* input_);
/*!
 * Parses initial trust region radius
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return radius.
 **/
double parseInitialTrustRegionRadius(const mxArray* input_);
/*!
 * Parses trust region radius expansion scalar
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseTrustRegionExpansionFactor(const mxArray* input_);
/*!
 * Parses line search step contraction scalar
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseLineSearchContractionFactor(const mxArray* input_);
/*!
 * Parses trust region radius contraction scalar
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseTrustRegionContractionFactor(const mxArray* input_);
/*!
 * Parses line search step stagnation tolerance
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseLineSearchStagnationTolerance(const mxArray* input_);
/*!
 * Parses feasible step contraction scalar
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return scalar.
 **/
double parseFeasibleStepContractionFactor(const mxArray* input_);
/*!
 * Parses lower bound on actual over predicted reduction ratio
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseMinActualOverPredictedReductionRatio(const mxArray* input_);
/*!
 * Parses mid bound on actual over predicted reduction ratio
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseMidActualOverPredictedReductionRatio(const mxArray* input_);
/*!
 * Parses upper bound on actual over predicted reduction ratio
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseMaxActualOverPredictedReductionRatio(const mxArray* input_);
//@}

//! @name Free functions that return boolean type
//@{
/*!
 * Parses flag on initial trust region radius: set to norm of gradient or
 * user defined value.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
bool parseSetInitialTrustRegionRadiusToNormGradFlag(const mxArray* input_);
//@}

//! @name Free functions that return dotk::flag type
//@{
/*!
 * Parses optimization problem type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::problem_t parseProblemType(const mxArray* input_);
/*!
 * Parses line search method type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::line_search_t parseLineSearchMethod(const mxArray* input_);
/*!
 * Parses trust region method type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::trustregion_t parseTrustRegionMethod(const mxArray* input_);
/*!
 * Parses nonlinear conjugate gradient method type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::nonlinearcg_t parseNonlinearCgMethod(const mxArray* input_);
/*!
 * Parses Hassian computation method type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::hessian_t parseHessianComputationMethod(const mxArray* input_);
/*!
 * Parses gradient computation method type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::gradient_t parseGradientComputationMethod(const mxArray* input_);/*!
 * Parses bound constraint method type
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return flag.
 **/
dotk::types::constraint_method_t parseBoundConstraintMethod(const mxArray* input_);
//@}

//! @name Free functions that return mxArray pointer type
//@{
/*!
 * Parses functor to objective function interface.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to objective function interface.
 **/
mxArray* parseObjectiveFunction(const mxArray* input_);
/*!
 * Parses functor to equality constraint interface.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to equality constraint interface.
 **/
mxArray* parseEqualityConstraint(const mxArray* input_);
/*!
 * Parses functor to inequality constraint interface.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to inequality constraint interface.
 **/
mxArray* parseInequalityConstraint(const mxArray* input_);
/*!
 * Parses lower bounds on dual variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to dual lower bounds.
 **/
mxArray* parseDualLowerBound(const mxArray* input_);
/*!
 * Parses upper bounds on dual variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to dual upper bounds.
 **/
mxArray* parseDualUpperBound(const mxArray* input_);
/*!
 * Parses lower bounds on state variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to state lower bounds.
 **/
mxArray* parseStateLowerBound(const mxArray* input_);
/*!
 * Parses upper bounds on state variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to state upper bounds.
 **/
mxArray* parseStateUpperBound(const mxArray* input_);
/*!
 * Parses initial control variables (i.e. initial guess).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to control variables.
 **/
mxArray* parseInitialControl(const mxArray* input_);
/*!
 * Parses lower bounds on control variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to control lower bounds.
 **/
mxArray* parseControlLowerBound(const mxArray* input_);
/*!
 * Parses upper bounds on control variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to control upper bounds.
 **/
mxArray* parseControlUpperBound(const mxArray* input_);
/*!
 * Parses perturbation vector for finite difference diagnostics.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 * \return MEX array pointer to perturbation vector.
 **/
mxArray* parseFiniteDifferencePerturbation(const mxArray* input_);
//@}

}

}

#endif /* DOTK_MEXALGORITHMPARSER_HPP_ */
