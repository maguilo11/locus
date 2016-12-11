/*
 * DOTk_ScaleParametersNLCG.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_ScaleParametersNLCG.hpp"

namespace dotk
{

namespace nlcg
{

Real fletcherReeves(const dotk::Vector<Real> & new_steepest_descent_, const dotk::Vector<Real> & old_steepest_descent_)
{
    Real scaling = (new_steepest_descent_.dot(new_steepest_descent_))
            / (old_steepest_descent_.dot(old_steepest_descent_));
    return (scaling);
}

Real polakRibiere(const dotk::Vector<Real> & new_steepest_descent_, const dotk::Vector<Real> & old_steepest_descent_)
{
    Real numerator = new_steepest_descent_.dot(new_steepest_descent_)
            - new_steepest_descent_.dot(old_steepest_descent_);
    Real denominator = old_steepest_descent_.dot(old_steepest_descent_);
    Real scaling = numerator / denominator;
    return (scaling);
}

Real hestenesStiefel(const dotk::Vector<Real> & new_steepest_descent_,
                     const dotk::Vector<Real> & old_steepest_descent_,
                     const dotk::Vector<Real> & old_trial_step_)
{
    Real numerator = new_steepest_descent_.dot(new_steepest_descent_)
            - new_steepest_descent_.dot(old_steepest_descent_);
    Real denominator = old_trial_step_.dot(new_steepest_descent_) - old_trial_step_.dot(old_steepest_descent_);
    Real scaling = numerator / denominator;
    return (scaling);
}

Real daiYuan(const dotk::Vector<Real> & new_steepest_descent_,
             const dotk::Vector<Real> & old_steepest_descent_,
             const dotk::Vector<Real> & old_trial_step_)
{
    Real numerator = new_steepest_descent_.dot(new_steepest_descent_);
    Real denominator = old_trial_step_.dot(new_steepest_descent_) - old_trial_step_.dot(old_steepest_descent_);
    Real scaling = numerator / denominator;
    return (scaling);
}

Real conjugateDescent(const dotk::Vector<Real> & new_steepest_descent_,
                      const dotk::Vector<Real> & old_steepest_descent_,
                      const dotk::Vector<Real> & old_trial_step_)
{
    Real numerator = new_steepest_descent_.norm();
    Real denominator = old_trial_step_.dot(old_steepest_descent_);
    Real scaling = numerator / denominator;
    return (scaling);
}

Real liuStorey(const dotk::Vector<Real> & new_steepest_descent_,
               const dotk::Vector<Real> & old_steepest_descent_,
               const dotk::Vector<Real> & old_trial_step_)
{
    Real numerator = new_steepest_descent_.dot(new_steepest_descent_)
            - new_steepest_descent_.dot(old_steepest_descent_);
    Real denominator = old_trial_step_.dot(old_steepest_descent_);
    Real scaling = numerator / denominator;
    return (scaling);
}

}

}
