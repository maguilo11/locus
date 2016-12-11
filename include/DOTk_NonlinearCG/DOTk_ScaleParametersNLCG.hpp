/*
 * DOTk_ScaleParametersNLCG.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SCALEPARAMETERSNLCG_HPP_
#define DOTK_SCALEPARAMETERSNLCG_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace nlcg
{

Real fletcherReeves(const dotk::Vector<Real> & new_steepest_descent_, const dotk::Vector<Real> & old_steepest_descent_);
Real polakRibiere(const dotk::Vector<Real> & new_steepest_descent_, const dotk::Vector<Real> & old_steepest_descent_);
Real hestenesStiefel(const dotk::Vector<Real> & new_steepest_descent_,
                     const dotk::Vector<Real> & old_steepest_descent_,
                     const dotk::Vector<Real> & old_trial_step_);
Real daiYuan(const dotk::Vector<Real> & new_steepest_descent_,
             const dotk::Vector<Real> & old_steepest_descent_,
             const dotk::Vector<Real> & old_trial_step_);
Real conjugateDescent(const dotk::Vector<Real> & new_steepest_descent_,
                      const dotk::Vector<Real> & old_steepest_descent_,
                      const dotk::Vector<Real> & old_trial_step_);
Real liuStorey(const dotk::Vector<Real> & new_steepest_descent_,
               const dotk::Vector<Real> & old_steepest_descent_,
               const dotk::Vector<Real> & old_trial_step_);

}

}

#endif /* DOTK_SCALEPARAMETERSNLCG_HPP_ */
