/*
 * TRROM_BoundConstraints.hpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_BOUNDCONSTRAINTS_HPP_
#define TRROM_BOUNDCONSTRAINTS_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class BoundConstraints
{
public:
    BoundConstraints();
    virtual ~BoundConstraints();

    bool active();
    void inactivate();
    double getContractionFactor() const;
    void setContractionFactor(double input_);
    int getMaxNumFeasibleIterations() const;
    void setMaxNumFeasibleIterations(int input_);

    virtual bool isDirectionFeasible(const trrom::Vector<double> & lower_bound_,
                                     const trrom::Vector<double> & upper_bound_,
                                     const trrom::Vector<double> & data_);
    virtual void computeFeasibleDirection(const trrom::Vector<double> & lower_bound_,
                                          const trrom::Vector<double> & upper_bound_,
                                          const trrom::Vector<double> & current_variable_,
                                          const trrom::Vector<double> & current_trial_step_,
                                          trrom::Vector<double> & trial_variable_,
                                          trrom::Vector<double> & feasible_direction_);
    virtual void project(const trrom::Vector<double> & lower_bound_,
                         const trrom::Vector<double> & upper_bound_,
                         trrom::Vector<double> & variable_);
    virtual void pruneActive(const trrom::Vector<double> & active_set_,
                             trrom::Vector<double> & direction_,
                             bool prune_ = true);
    virtual void computeProjectedStep(const trrom::Vector<double> & trial_variables_,
                                      const trrom::Vector<double> & current_variables_,
                                      trrom::Vector<double> & projected_step_);
    virtual void computeProjectedGradient(const trrom::Vector<double> & trial_variable_,
                                          const trrom::Vector<double> & lower_bound_,
                                          const trrom::Vector<double> & upper_bound_,
                                          trrom::Vector<double> & gradient_);
    virtual void projectActive(const trrom::Vector<double> & lower_bound_,
                               const trrom::Vector<double> & upper_bound_,
                               trrom::Vector<double> & variable_,
                               trrom::Vector<double> & active_set_);
    virtual void computeActiveAndInactiveSets(const trrom::Vector<double> & input_,
                                              const trrom::Vector<double> & lower_bound_,
                                              const trrom::Vector<double> & upper_bound_,
                                              trrom::Vector<double> & active_,
                                              trrom::Vector<double> & inactive_);

private:
    bool m_Active;
    double m_ContractionFactor;
    int m_MaxNumFeasibleIterations;

private:
    BoundConstraints(const trrom::BoundConstraints &);
    trrom::BoundConstraints & operator=(const trrom::BoundConstraints &);
};

}

#endif /* TRROM_BOUNDCONSTRAINTS_HPP_ */
