/*
 * DOTk_BoundConstraints.cpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "vector.hpp"
#include "DOTk_BoundConstraints.hpp"

namespace dotk
{

DOTk_BoundConstraints::DOTk_BoundConstraints() :
        m_Active(true),
        m_ContractionFactor(0.5),
        m_MaxNumFeasibleIterations(10)
{
}

DOTk_BoundConstraints::~DOTk_BoundConstraints()
{
}

bool DOTk_BoundConstraints::active()
{
    return (m_Active);
}

void DOTk_BoundConstraints::deactivate()
{
    m_Active = false;
}

Real DOTk_BoundConstraints::getContractionFactor() const
{
    return (m_ContractionFactor);
}

void DOTk_BoundConstraints::setContractionFactor(Real input_)
{
    m_ContractionFactor = input_;
}

size_t DOTk_BoundConstraints::getMaxNumFeasibleIterations() const
{
    return (m_MaxNumFeasibleIterations);
}

void DOTk_BoundConstraints::setMaxNumFeasibleIterations(size_t input_)
{
    m_MaxNumFeasibleIterations = input_;
}

bool DOTk_BoundConstraints::isDirectionFeasible(const dotk::Vector<Real> & lower_bound_,
                                                const dotk::Vector<Real> & upper_bound_,
                                                const dotk::Vector<Real> & data_)
{
    size_t index = 0;
    bool feasible = true;

    while(index < data_.size())
    {
        bool element_less_than_lower_bound = data_[index] < lower_bound_[index] ? true: false;
        bool element_greater_than_upper_bound = data_[index] > upper_bound_[index] ? true: false;
        if(element_less_than_lower_bound || element_greater_than_upper_bound)
        {
            feasible = false;
            break;
        }
        ++ index;
    }

    return (feasible);
}

void DOTk_BoundConstraints::computeFeasibleDirection(const dotk::Vector<Real> & lower_bound_,
                                                     const dotk::Vector<Real> & upper_bound_,
                                                     const dotk::Vector<Real> & current_variable_,
                                                     const dotk::Vector<Real> & current_trial_step_,
                                                     dotk::Vector<Real> & trial_variable_,
                                                     dotk::Vector<Real> & feasible_direction_)
{
    Real step = 1.0;
    feasible_direction_.copy(current_trial_step_);
    trial_variable_.copy(current_variable_);
    trial_variable_.axpy(step, feasible_direction_);

    size_t itr = 1;
    while(this->isDirectionFeasible(lower_bound_, upper_bound_, trial_variable_) == false)
    {
        step *= m_ContractionFactor;
        trial_variable_.copy(current_variable_);
        trial_variable_.axpy(step, feasible_direction_);
        if(itr >= m_MaxNumFeasibleIterations)
        {
            feasible_direction_.scale(step);
            this->project(lower_bound_, upper_bound_, trial_variable_);
            this->computeProjectedStep(trial_variable_, current_variable_, feasible_direction_);
            break;
        }
        ++ itr;
    }

    if(itr < m_MaxNumFeasibleIterations)
    {
        feasible_direction_.scale(step);
    }
}

void DOTk_BoundConstraints::project(const dotk::Vector<Real> & lower_bound_,
                                    const dotk::Vector<Real> & upper_bound_,
                                    dotk::Vector<Real> & variable_)
{
    size_t number_variables = variable_.size();
    for(size_t index = 0; index < number_variables; ++ index)
    {
        variable_[index] = std::max(variable_[index], lower_bound_[index]);
        variable_[index] = std::min(variable_[index], upper_bound_[index]);
    }
}

void DOTk_BoundConstraints::pruneActive(const dotk::Vector<Real> & active_set_,
                                        dotk::Vector<Real> & direction_,
                                        bool prune_)
{
    if(prune_ == true)
    {
        size_t number_variables = active_set_.size();
        for(size_t index = 0; index < number_variables; ++ index)
        {
            direction_[index] = active_set_[index] > static_cast<Real>(0.) ? static_cast<Real>(0.): direction_[index];
        }
    }
}

void DOTk_BoundConstraints::computeProjectedStep(const dotk::Vector<Real> & trial_variables_,
                                                 const dotk::Vector<Real> & current_variables_,
                                                 dotk::Vector<Real> & projected_step_)
{
    projected_step_.copy(trial_variables_);
    projected_step_.axpy(static_cast<Real>(-1), current_variables_);
}

void DOTk_BoundConstraints::computeProjectedGradient(const dotk::Vector<Real> & trial_variable_,
                                                     const dotk::Vector<Real> & lower_bound_,
                                                     const dotk::Vector<Real> & upper_bound_,
                                                     dotk::Vector<Real> & gradient_)
{
    size_t number_variables = trial_variable_.size();
    for(size_t index = 0; index < number_variables; ++ index)
    {
        gradient_[index] = trial_variable_[index] < lower_bound_[index] ? static_cast<Real>(0.): gradient_[index];
        gradient_[index] = trial_variable_[index] > upper_bound_[index] ? static_cast<Real>(0.): gradient_[index];
    }
}

void DOTk_BoundConstraints::projectActive(const dotk::Vector<Real> & lower_bound_,
                                          const dotk::Vector<Real> & upper_bound_,
                                          dotk::Vector<Real> & variable_,
                                          dotk::Vector<Real> & active_set_)
{
    active_set_.fill(0.);
    size_t number_variables = variable_.size();
    for(size_t index = 0; index < number_variables; ++ index)
    {
        variable_[index] = std::max(variable_[index], lower_bound_[index]);
        variable_[index] = std::min(variable_[index], upper_bound_[index]);
        active_set_[index] = static_cast<size_t>((variable_[index] == lower_bound_[index])
                || (variable_[index] == upper_bound_[index]));
    }
}

void DOTk_BoundConstraints::computeActiveAndInactiveSets(const dotk::Vector<Real> & input_,
                                                         const dotk::Vector<Real> & lower_bound_,
                                                         const dotk::Vector<Real> & upper_bound_,
                                                         dotk::Vector<Real> & active_,
                                                         dotk::Vector<Real> & inactive_)
{
    assert(input_.size() == inactive_.size());
    assert(active_.size() == inactive_.size());
    assert(input_.size() == lower_bound_.size());
    assert(input_.size() == upper_bound_.size());

    active_.fill(0.);
    inactive_.fill(0.);
    size_t number_elements = input_.size();
    for(size_t index = 0; index < number_elements; ++ index)
    {
        active_[index] = static_cast<size_t>((input_[index] >= upper_bound_[index])
                || (input_[index] <= lower_bound_[index]));
        inactive_[index] = static_cast<size_t>((input_[index] < upper_bound_[index])
                && (input_[index] > lower_bound_[index]));
    }
}

}
