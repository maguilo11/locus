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

template<class Type>
class vector;

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


    virtual bool isDirectionFeasible(const dotk::vector<Real> & lower_bound_,
                                     const dotk::vector<Real> & upper_bound_,
                                     const dotk::vector<Real> & data_);
    virtual void computeFeasibleDirection(const dotk::vector<Real> & lower_bound_,
                                          const dotk::vector<Real> & upper_bound_,
                                          const dotk::vector<Real> & current_variable_,
                                          const dotk::vector<Real> & current_trial_step_,
                                          dotk::vector<Real> & trial_variable_,
                                          dotk::vector<Real> & feasible_direction_);
    virtual void project(const dotk::vector<Real> & lower_bound_,
                         const dotk::vector<Real> & upper_bound_,
                         dotk::vector<Real> & variable_);
    virtual void pruneActive(const dotk::vector<Real> & active_set_,
                             dotk::vector<Real> & direction_,
                             bool prune_ = true);
    virtual void computeProjectedStep(const dotk::vector<Real> & trial_variables_,
                                      const dotk::vector<Real> & current_variables_,
                                      dotk::vector<Real> & projected_step_);
    virtual void computeProjectedGradient(const dotk::vector<Real> & trial_variable_,
                                          const dotk::vector<Real> & lower_bound_,
                                          const dotk::vector<Real> & upper_bound_,
                                          dotk::vector<Real> & gradient_);
    virtual void projectActive(const dotk::vector<Real> & lower_bound_,
                               const dotk::vector<Real> & upper_bound_,
                               dotk::vector<Real> & variable_,
                               dotk::vector<Real> & active_set_);
    virtual void computeActiveAndInactiveSets(const dotk::vector<Real> & input_,
                                              const dotk::vector<Real> & lower_bound_,
                                              const dotk::vector<Real> & upper_bound_,
                                              dotk::vector<Real> & active_,
                                              dotk::vector<Real> & inactive_);

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
