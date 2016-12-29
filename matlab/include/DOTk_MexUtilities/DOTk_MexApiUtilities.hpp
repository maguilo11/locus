/*
 * DOTk_MexApiUtilities.hpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXAPIUTILITIES_HPP_
#define DOTK_MEXAPIUTILITIES_HPP_

#include <mex.h>
#include <string>
#include "DOTk_Types.hpp"

namespace dotk
{

namespace mex
{

//! @name Void free functions
//@{
/*!
 * Destroys MEX array pointer.
 * Parameters:
 *    \param In/Out
 *          input_: MEX array pointer
 **/
void destroy(mxArray* input_);
/*!
 * Handles MATLAB exception and removes unnecessary jargon.
 * Parameters:
 *    \param In
 *          input_: MEX array pointer
 *    \param Out
 *          output_: string with error message
 **/
void handleException(mxArray* input_, std::string output_);
//@}

dotk::types::problem_t getProblemType(const mxArray* input_);
dotk::types::line_search_t getLineSearchMethod(const mxArray* input_);
dotk::types::trustregion_t getTrustRegionMethod(const mxArray* input_);
dotk::types::nonlinearcg_t getNonlinearCgMethod(const mxArray* input_);
dotk::types::display_t getDiagnosticsDisplayOption(const mxArray* input_);
dotk::types::hessian_t getHessianComputationMethod(const mxArray* input_);
dotk::types::gradient_t getGradientComputationMethod(const mxArray* input_);
dotk::types::constraint_method_t getBoundConstraintMethod(const mxArray* input_);

}

}

#endif /* DOTK_MEXAPIUTILITIES_HPP_ */
