/*
 * TRROM_MxParsingUtilities.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MXPARSINGUTILITIES_HPP_
#define TRROM_MXPARSINGUTILITIES_HPP_

#include <mex.h>
#include <tr1/memory>

namespace trrom
{

class MxVector;
class ReducedBasisPDE;
class ReducedBasisObjective;

namespace mx
{

//! @name Integer free functions
//@{
/*!
 * Parses the length of the array of dual variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of dual variables.
 **/
int parseNumberDuals(const mxArray* input_);
/*!
 * Parses the length of the array of the state variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of state variables.
 **/
int parseNumberStates(const mxArray* input_);
/*!
 * Parses the length of the array of slack variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of slack variables.
 **/
int parseNumberSlacks(const mxArray* input_);
/*!
 * Parses the length of the array of control variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return number of control variables.
 **/
int parseNumberControls(const mxArray* input_);
/*!
 * Parses the maximum number of trust region sub-problem iterations.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of iterations.
 **/
int parseMaxNumberSubProblemIterations(const mxArray* input_);
/*!
 * Parses the maximum number of outer iterations (i.e. optimization algorithm iterations).
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return maximum number of iterations.
 **/
int parseMaxNumberOuterIterations(const mxArray* input_);
//@}

//! @name Scalar free functions
//@{
/*!
 * Parses the lower bound on the trust region radius.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseMinTrustRegionRadius(const mxArray* input_);
/*!
 * Parses the upper bound on the trust region radius.
  * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseMaxTrustRegionRadius(const mxArray* input_);
/*!
 * Parses trust region radius contraction scalar.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return contraction scalar.
 **/
double parseTrustRegionContractionScalar(const mxArray* input_);
/*!
 * Parses trust region radius expansion scalar.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return expansion scalar.
 **/
double parseTrustRegionExpansionScalar(const mxArray* input_);
/*!
 * Parses mid bound on the actual over predicted reduction ration.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseActualOverPredictedReductionMidBound(const mxArray* input_);
/*!
 * Parses lower bound on the actual over predicted reduction ration.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseActualOverPredictedReductionLowerBound(const mxArray* input_);
/*!
 * Parses upper bound on the actual over predicted reduction ration.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return bound.
 **/
double parseActualOverPredictedReductionUpperBound(const mxArray* input_);
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
 * Parses tolerance on the norm of the gradient.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseGradientTolerance(const mxArray* input_);
/*!
 * Parses tolerance on the objective function value.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseObjectiveTolerance(const mxArray* input_);
/*!
 * Parses tolerance on the misfit between two subsequent objective
 * function evaluations, i.e. \f_i(x)-f_{i-1}(x), where f denotes
 * the objective function, i is the current outer iteration count,
 * and x is the primal variable.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *
 * \return tolerance.
 **/
double parseStagnationTolerance(const mxArray* input_);
//@}

//! @name Void free functions
//@{
/*!
 * Parses the lower bounds on the control variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *    \param Out
 *          output_: MxVector of upper bounds
 **/
void parseControlLowerBound(const mxArray* input_, trrom::MxVector & output_);
/*!
 * Parses the upper bounds on the control variables.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *    \param Out
 *          output_: MxVector of upper bounds
 **/
void parseControlUpperBound(const mxArray* input_, trrom::MxVector & output_);
/*!
 * Parses functor to objective function operators' interface.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *    \param Out
 *          output_: functor to operators' interface
 **/
void parseReducedObjectiveFunction(const mxArray* input_, std::tr1::shared_ptr<trrom::ReducedBasisObjective> & output_);
/*!
 * Parses functor to reduced basis partial differential equation (PDE) operators' interface.
 * Parameters:
 *    \param In
 *          input_: const MEX array pointer
 *    \param Out
 *          output_: functor to operators' interface
 **/
void parseReducedBasisPartialDifferentialEquation(const mxArray* input_, std::tr1::shared_ptr<trrom::ReducedBasisPDE> & output_);

}

}

#endif /* TRROM_MXPARSINGUTILITIES_HPP_ */
