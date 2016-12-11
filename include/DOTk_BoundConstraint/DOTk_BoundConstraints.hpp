/*
 * DOTk_BoundConstraints.hpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BOUNDCONSTRAINTS_HPP_
#define DOTK_BOUNDCONSTRAINTS_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<class ScalarType>
class Vector;

class DOTk_BoundConstraints
{
public:
    DOTk_BoundConstraints();
    virtual ~DOTk_BoundConstraints();

    bool active();
    void deactivate();
    Real getContractionFactor() const;
    void setContractionFactor(Real input_);
    size_t getMaxNumFeasibleIterations() const;
    void setMaxNumFeasibleIterations(size_t input_);


    virtual bool isDirectionFeasible(const dotk::Vector<Real> & lower_bound_,
                                     const dotk::Vector<Real> & upper_bound_,
                                     const dotk::Vector<Real> & data_);
    virtual void computeFeasibleDirection(const dotk::Vector<Real> & lower_bound_,
                                          const dotk::Vector<Real> & upper_bound_,
                                          const dotk::Vector<Real> & current_variable_,
                                          const dotk::Vector<Real> & current_trial_step_,
                                          dotk::Vector<Real> & trial_variable_,
                                          dotk::Vector<Real> & feasible_direction_);
    virtual void project(const dotk::Vector<Real> & lower_bound_,
                         const dotk::Vector<Real> & upper_bound_,
                         dotk::Vector<Real> & variable_);
    virtual void pruneActive(const dotk::Vector<Real> & active_set_,
                             dotk::Vector<Real> & direction_,
                             bool prune_ = true);
    virtual void computeProjectedStep(const dotk::Vector<Real> & trial_variables_,
                                      const dotk::Vector<Real> & current_variables_,
                                      dotk::Vector<Real> & projected_step_);
    virtual void computeProjectedGradient(const dotk::Vector<Real> & trial_variable_,
                                          const dotk::Vector<Real> & lower_bound_,
                                          const dotk::Vector<Real> & upper_bound_,
                                          dotk::Vector<Real> & gradient_);
    virtual void projectActive(const dotk::Vector<Real> & lower_bound_,
                               const dotk::Vector<Real> & upper_bound_,
                               dotk::Vector<Real> & variable_,
                               dotk::Vector<Real> & active_set_);
    virtual void computeActiveAndInactiveSets(const dotk::Vector<Real> & input_,
                                              const dotk::Vector<Real> & lower_bound_,
                                              const dotk::Vector<Real> & upper_bound_,
                                              dotk::Vector<Real> & active_,
                                              dotk::Vector<Real> & inactive_);

private:
    bool m_Active;
    Real m_ContractionFactor;
    size_t m_MaxNumFeasibleIterations;

private:
    DOTk_BoundConstraints(const dotk::DOTk_BoundConstraints &);
    dotk::DOTk_BoundConstraints & operator=(const dotk::DOTk_BoundConstraints &);
};

}

#endif /* DOTK_BOUNDCONSTRAINTS_HPP_ */
