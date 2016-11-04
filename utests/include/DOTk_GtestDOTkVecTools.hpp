/*
 * DOTk_GtestDOTkVecTools.hpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GTESTDOTKVECTOOLS_HPP_
#define DOTK_GTESTDOTKVECTOOLS_HPP_

#include "DOTK_Types.hpp"

namespace dotk
{

class DOTk_Variable;

template<class Type>
class vector;

namespace gtest
{

std::tr1::shared_ptr<dotk::vector<Real> > allocateControl();
std::tr1::shared_ptr<dotk::vector<Real> > allocateData(size_t num_entries_, Real value_ = 0);

void checkResults(const std::vector<Real> & results_, const std::vector<Real> & gold_, Real tol_ = 1e-8);
void checkResults(const dotk::vector<Real> & results_, const dotk::vector<Real> & gold_, Real tol_ = 1e-8);
void checkResults(const dotk::vector<Real> & gold_, const dotk::vector<Real> & results_, int thread_count_, Real tol_ = 1e-8);
void checkResults(const size_t & num_gold_values,
                  const Real* gold_,
                  const dotk::vector<Real> & results_,
                  int thread_count_ = 1,
                  Real tol_ = 1e-8);
void checkResults(const size_t & num_gold_values,
                  const Real* gold_,
                  const size_t & num_result_values,
                  const Real* results_,
                  int thread_count_ = 1,
                  Real tol_ = 1e-8);

}

}

#endif /* DOTK_GTESTDOTKVECTOOLS_HPP_ */
