/*
 * DOTk_GradBasedIoUtils.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_GRADBASEDIOUTILS_HPP_
#define DOTK_GRADBASEDIOUTILS_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

namespace ioUtils
{

bool printMessage(std::ostringstream & msg_);
void getLicenseMessage(std::ostringstream & msg_);
void checkType(dotk::types::variable_t input_type_, dotk::types::variable_t primal_type_);
void getSolverExitCriterion(dotk::types::solver_stop_criterion_t type_, std::ostringstream & criterion_);
void checkDataPtr(const std::shared_ptr<dotk::Vector<Real> > & data_, std::ostringstream & data_type_);

}

}

#endif /* DOTK_GRADBASEDIOUTILS_HPP_ */
