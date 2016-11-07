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

template<typename Type>
class vector;

namespace nlcg
{

Real fletcherReeves(const dotk::vector<Real> & new_steepest_descent_, const dotk::vector<Real> & old_steepest_descent_);
Real polakRibiere(const dotk::vector<Real> & new_steepest_descent_, const dotk::vector<Real> & old_steepest_descent_);
Real hestenesStiefel(const dotk::vector<Real> & new_steepest_descent_,
                     const dotk::vector<Real> & old_steepest_descent_,
                     const dotk::vector<Real> & old_trial_step_);
Real daiYuan(const dotk::vector<Real> & new_steepest_descent_,
             const dotk::vector<Real> & old_steepest_descent_,
             const dotk::vector<Real> & old_trial_step_);
Real conjugateDescent(const dotk::vector<Real> & new_steepest_descent_,
                      const dotk::vector<Real> & old_steepest_descent_,
                      const dotk::vector<Real> & old_trial_step_);
Real liuStorey(const dotk::vector<Real> & new_steepest_descent_,
               const dotk::vector<Real> & old_steepest_descent_,
               const dotk::vector<Real> & old_trial_step_);

}

}

#endif /* DOTK_SCALEPARAMETERSNLCG_HPP_ */
