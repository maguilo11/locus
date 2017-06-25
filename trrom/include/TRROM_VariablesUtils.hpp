/*
 * TRROM_VariablesUtils.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_VARIABLESUTILS_HPP_
#define TRROM_VARIABLESUTILS_HPP_

#include <memory>

namespace trrom
{

template<typename ScalarType>
class Vector;

void printDual(const std::shared_ptr<trrom::Vector<double> > & dual_);
void printState(const std::shared_ptr<trrom::Vector<double> > & state_);
void printControl(const std::shared_ptr<trrom::Vector<double> > & control_);
void printSolution(const std::shared_ptr<trrom::Vector<double> > & solution_);

}

#endif /* TRROM_VARIABLESUTILS_HPP_ */
