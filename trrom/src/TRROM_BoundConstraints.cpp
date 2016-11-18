/*
 * TRROM_BoundConstraints.cpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>

#include "TRROM_Vector.hpp"
#include "TRROM_BoundConstraints.hpp"

namespace trrom
{

BoundConstraints::BoundConstraints() :
        m_Active(true),
        m_ContractionFactor(0.5),
        m_MaxNumFeasibleIterations(10)
{
}

BoundConstraints::~BoundConstraints()
{
}

bool BoundConstraints::active()
{
    return (m_Active);
}

void BoundConstraints::inactivate()
{
    m_Active = false;
}

double BoundConstraints::getContractionFactor() const
{
    return (m_ContractionFactor);
}

void BoundConstraints::setContractionFactor(double input_)
{
    m_ContractionFactor = input_;
}

int BoundConstraints::getMaxNumFeasibleIterations() const
{
    return (m_MaxNumFeasibleIterations);
}

void BoundConstraints::setMaxNumFeasibleIterations(int input_)
{
    m_MaxNumFeasibleIterations = input_;
}

bool BoundConstraints::isDirectionFeasible(const trrom::Vector<double> & lower_bound_,
                                           const trrom::Vector<double> & upper_bound_,
                                           const trrom::Vector<double> & data_)
{
    int index = 0;
    bool feasible = true;

    while(index < data_.size())
    {
        bool element_less_than_lower_bound = data_[index] < lower_bound_[index] ? true : false;
        bool element_greater_than_upper_bound = data_[index] > upper_bound_[index] ? true : false;
        if(element_less_than_lower_bound || element_greater_than_upper_bound)
        {
            feasible = false;
            break;
        }
        ++ index;
    }

    return (feasible);
}

void BoundConstraints::computeFeasibleDirection(const trrom::Vector<double> & lower_bound_,
                                                const trrom::Vector<double> & upper_bound_,
                                                const trrom::Vector<double> & current_variable_,
                                                const trrom::Vector<double> & current_trial_step_,
                                                trrom::Vector<double> & trial_variable_,
                                                trrom::Vector<double> & feasible_direction_)
{
    double step = 1.0;
    feasible_direction_.update(1., current_trial_step_, 0.);
    trial_variable_.update(1., current_variable_, 0.);
    trial_variable_.update(step, feasible_direction_, 1.);

    int itr = 1;
    while(this->isDirectionFeasible(lower_bound_, upper_bound_, trial_variable_) == false)
    {
        step *= m_ContractionFactor;
        trial_variable_.update(1., current_variable_, 0.);
        trial_variable_.update(step, feasible_direction_, 1.);
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

void BoundConstraints::project(const trrom::Vector<double> & lower_bound_,
                               const trrom::Vector<double> & upper_bound_,
                               trrom::Vector<double> & variable_)
{
    int number_variables = variable_.size();
    for(int index = 0; index < number_variables; ++ index)
    {
        variable_[index] = std::max(variable_[index], lower_bound_[index]);
        variable_[index] = std::min(variable_[index], upper_bound_[index]);
    }
}

void BoundConstraints::pruneActive(const trrom::Vector<double> & active_set_,
                                   trrom::Vector<double> & direction_,
                                   bool prune_)
{
    if(prune_ == true)
    {
        int number_variables = active_set_.size();
        for(int index = 0; index < number_variables; ++ index)
        {
            direction_[index] =
                    active_set_[index] > static_cast<double>(0.) ? static_cast<double>(0.) : direction_[index];
        }
    }
}

void BoundConstraints::computeProjectedStep(const trrom::Vector<double> & trial_variables_,
                                            const trrom::Vector<double> & current_variables_,
                                            trrom::Vector<double> & projected_step_)
{
    projected_step_.update(1., trial_variables_, 0.);
    projected_step_.update(-1., current_variables_, 1.);
}

void BoundConstraints::computeProjectedGradient(const trrom::Vector<double> & trial_variable_,
                                                const trrom::Vector<double> & lower_bound_,
                                                const trrom::Vector<double> & upper_bound_,
                                                trrom::Vector<double> & gradient_)
{
    int number_variables = trial_variable_.size();
    for(int index = 0; index < number_variables; ++ index)
    {
        gradient_[index] = trial_variable_[index] < lower_bound_[index] ? static_cast<double>(0.) : gradient_[index];
        gradient_[index] = trial_variable_[index] > upper_bound_[index] ? static_cast<double>(0.) : gradient_[index];
    }
}

void BoundConstraints::projectActive(const trrom::Vector<double> & lower_bound_,
                                     const trrom::Vector<double> & upper_bound_,
                                     trrom::Vector<double> & variable_,
                                     trrom::Vector<double> & active_set_)
{
    active_set_.fill(0.);
    int number_variables = variable_.size();
    for(int index = 0; index < number_variables; ++ index)
    {
        variable_[index] = std::max(variable_[index], lower_bound_[index]);
        variable_[index] = std::min(variable_[index], upper_bound_[index]);
        active_set_[index] = static_cast<int>((variable_[index] == lower_bound_[index])
                || (variable_[index] == upper_bound_[index]));
    }
}

void BoundConstraints::computeActiveAndInactiveSets(const trrom::Vector<double> & input_,
                                                    const trrom::Vector<double> & lower_bound_,
                                                    const trrom::Vector<double> & upper_bound_,
                                                    trrom::Vector<double> & active_,
                                                    trrom::Vector<double> & inactive_)
{
    assert(input_.size() == inactive_.size());
    assert(active_.size() == inactive_.size());
    assert(input_.size() == lower_bound_.size());
    assert(input_.size() == upper_bound_.size());

    active_.fill(0.);
    inactive_.fill(0.);
    int number_elements = input_.size();
    for(int index = 0; index < number_elements; ++ index)
    {
        active_[index] = static_cast<int>((input_[index] >= upper_bound_[index])
                || (input_[index] <= lower_bound_[index]));
        inactive_[index] = static_cast<int>((input_[index] < upper_bound_[index])
                && (input_[index] > lower_bound_[index]));
    }
}

}
