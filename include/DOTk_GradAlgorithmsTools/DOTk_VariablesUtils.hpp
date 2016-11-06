/*
 * DOTk_VariablesUtils.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_VARIABLESUTILS_HPP_
#define DOTK_VARIABLESUTILS_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename Type>
class vector;

void printDual(const std::tr1::shared_ptr<dotk::vector<Real> > & dual_);
void printState(const std::tr1::shared_ptr<dotk::vector<Real> > & state_);
void printControl(const std::tr1::shared_ptr<dotk::vector<Real> > & control_);
void printSolution(const std::tr1::shared_ptr<dotk::vector<Real> > & solution_);

}

#endif /* DOTK_VARIABLESUTILS_HPP_ */
