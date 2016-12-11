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
#include "DOTk_VectorTypes.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;

template<typename ScalarType>
class Vector;

namespace mex
{

void handleException(mxArray* err_, std::string msg_);
size_t getMexArrayDim(const dotk::DOTk_MexArrayPtr & ptr_);
void setDOTkData(const dotk::DOTk_MexArrayPtr & ptr_, dotk::Vector<double> & data_);
void copyData(size_t input_dim_, double* input_, dotk::Vector<double> & output_);

dotk::types::problem_t getProblemType(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::container_t getContainerType(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::line_search_t getLineSearchMethod(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::trustregion_t getTrustRegionMethod(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::nonlinearcg_t getNonlinearCgMethod(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::display_t getDiagnosticsDisplayOption(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::hessian_t getHessianComputationMethod(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::gradient_t getGradientComputationMethod(const dotk::DOTk_MexArrayPtr & ptr_);
dotk::types::constraint_method_t getBoundConstraintMethod(const dotk::DOTk_MexArrayPtr & ptr_);

}

}

#endif /* DOTK_MEXAPIUTILITIES_HPP_ */
