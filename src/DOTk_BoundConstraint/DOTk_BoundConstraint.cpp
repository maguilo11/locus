/*
 * DOTk_BoundConstraint.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_BoundConstraint.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_BoundConstraint::DOTk_BoundConstraint(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                           dotk::types::constraint_method_t aType) :
        m_Active(true),
        m_StepSize(1.0),
        m_ContractionStep(0.5),
        m_StagnationTolerance(1e-8),
        m_NewObjectiveFunctionValue(0.),
        m_NumFeasibleItr(0),
        m_MaxNumFeasibleItr(10),
        m_StepType(dotk::types::bound_step_t::ARMIJO_STEP),
        m_Type(aType),
        m_ActiveSet(aPrimal->control()->clone())
{
}

DOTk_BoundConstraint::~DOTk_BoundConstraint()
{
}

void DOTk_BoundConstraint::setStepSize(Real aInput)
{
    m_StepSize = aInput;
}

Real DOTk_BoundConstraint::getStepSize() const
{
    return (m_StepSize);
}

void DOTk_BoundConstraint::setStagnationTolerance(Real aInput)
{
    m_StagnationTolerance = aInput;
}

Real DOTk_BoundConstraint::getStagnationTolerance() const
{
    return (m_StagnationTolerance);
}

void DOTk_BoundConstraint::setContractionStep(Real aInput)
{
    m_ContractionStep = aInput;
}

Real DOTk_BoundConstraint::getContractionStep() const
{
    return (m_ContractionStep);
}

void DOTk_BoundConstraint::setNewObjectiveFunctionValue(Real aInput)
{
    m_NewObjectiveFunctionValue = aInput;
}

Real DOTk_BoundConstraint::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunctionValue);
}

void DOTk_BoundConstraint::setMaxNumFeasibleItr(size_t aInput)
{
    m_MaxNumFeasibleItr = aInput;
}

size_t DOTk_BoundConstraint::getMaxNumFeasibleItr() const
{
    return (m_MaxNumFeasibleItr);
}

void DOTk_BoundConstraint::setNumFeasibleItr(size_t aInput)
{
    m_NumFeasibleItr = aInput;
}

size_t DOTk_BoundConstraint::getNumFeasibleItr() const
{
    return (m_NumFeasibleItr);
}

void DOTk_BoundConstraint::activate(bool aInput)
{
    m_Active = aInput;
}

bool DOTk_BoundConstraint::active() const
{
    return (m_Active);
}

void DOTk_BoundConstraint::setStepType(dotk::types::bound_step_t aType)
{
    m_StepType = aType;
}

dotk::types::bound_step_t DOTk_BoundConstraint::getStepType() const
{
    return (m_StepType);
}

dotk::types::constraint_method_t DOTk_BoundConstraint::type() const
{
    return (m_Type);
}

Real DOTk_BoundConstraint::getArmijoStep(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                                         const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    aStep->step(aMng);
    Real new_objective_function_value = aStep->getNewObjectiveFunctionValue();
    this->setNewObjectiveFunctionValue(new_objective_function_value);
    Real step = aStep->getStepSize();

    return (step);
}

Real DOTk_BoundConstraint::getMinReductionStep(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    size_t itr = 1;
    Real step = 1.;
    Real new_objective_function_value = 0.;
    Real current_objective_func_val = aMng->getOldObjectiveFunctionValue();
    while(1)
    {
        aMng->getNewPrimal()->update(step, *aMng->getTrialStep(), 1.);
        new_objective_function_value = aMng->evaluateObjective(aMng->getNewPrimal());
        bool is_new_fval_less_than_current_fval =
                new_objective_function_value < current_objective_func_val ? true: false;
        if((is_new_fval_less_than_current_fval == true) || (itr == this->getMaxNumFeasibleItr()))
        {
            break;
        }
        aMng->getNewPrimal()->update(1., *aMng->getOldPrimal(), 0.);
        step = step * this->getContractionStep();
        ++ itr;
    }
    this->setNumFeasibleItr(itr);
    this->setNewObjectiveFunctionValue(new_objective_function_value);

    return (step);
}

Real DOTk_BoundConstraint::getStep(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                                   const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    Real step = 0.;
    switch(this->getStepType())
    {
        case dotk::types::bound_step_t::ARMIJO_STEP:
        {
            step = this->getArmijoStep(aStep,aMng);
            break;
        }
        case dotk::types::bound_step_t::MIN_REDUCTION_STEP:
        {
            step = this->getMinReductionStep(aMng);
            break;
        }
        case dotk::types::bound_step_t::CONSTANT_STEP:
        {
            step = this->getStepSize();
            break;
        }
        case dotk::types::bound_step_t::TRUST_REGION_STEP:
        {
            break;
        }
    }
    return (step);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_BoundConstraint::activeSet() const
{
    return (m_ActiveSet);
}

void DOTk_BoundConstraint::computeScaledTrialStep(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                                                  const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                                  const std::shared_ptr<dotk::Vector<Real> > & aCurrentPrimal)
{
    dotk::types::bound_step_t type = this->getStepType();
    switch(type)
    {
        case dotk::types::bound_step_t::ARMIJO_STEP:
        case dotk::types::bound_step_t::MIN_REDUCTION_STEP:
        {
            Real current_objective_function_value = aMng->getNewObjectiveFunctionValue();
            Real step = this->getStep(aStep, aMng);
            aMng->getTrialStep()->scale(step);
            aMng->getNewPrimal()->update(1., *aCurrentPrimal, 0.);
            aMng->setNewObjectiveFunctionValue(current_objective_function_value);
            break;
        }
        case dotk::types::bound_step_t::CONSTANT_STEP:
        {
            Real step = this->getStep(aStep, aMng);
            aMng->getTrialStep()->scale(step);
            break;
        }
        case dotk::types::bound_step_t::TRUST_REGION_STEP:
        {
            break;
        }
    }
}

void DOTk_BoundConstraint::project(const std::shared_ptr<dotk::Vector<Real> > & aLowerBound,
                                   const std::shared_ptr<dotk::Vector<Real> > & aUpperBound,
                                   const std::shared_ptr<dotk::Vector<Real> > & aPrimal)
{
    m_ActiveSet->fill(0);
    for(size_t index = 0; index < aPrimal->size(); ++index)
    {
        (*aPrimal)[index] = std::min((*aUpperBound)[index], (*aPrimal)[index]);
        (*aPrimal)[index] = std::max((*aLowerBound)[index], (*aPrimal)[index]);
        (*m_ActiveSet)[index] = static_cast<size_t>(((*aPrimal)[index] == (*aLowerBound)[index])
                || ((*aPrimal)[index] == (*aUpperBound)[index]));
    }
}

void DOTk_BoundConstraint::computeActiveSet(const std::shared_ptr<dotk::Vector<Real> > & aLowerBound,
                                            const std::shared_ptr<dotk::Vector<Real> > & aUpperBound,
                                            const std::shared_ptr<dotk::Vector<Real> > & aPrimal)
{
    assert(m_ActiveSet->size() == aPrimal->size());

    m_ActiveSet->fill(0);
    for(size_t index = 0; index < m_ActiveSet->size(); ++index)
    {
        (*m_ActiveSet)[index] = static_cast<size_t>(((*aPrimal)[index] <= (*aLowerBound)[index])
                || ((*aPrimal)[index] >= (*aUpperBound)[index]));
    }
}

void DOTk_BoundConstraint::pruneActive(const std::shared_ptr<dotk::Vector<Real> > & aDirection)
{
    for(size_t index = 0; index < m_ActiveSet->size(); ++ index)
    {
        (*aDirection)[index] =
                (*m_ActiveSet)[index] > static_cast<Real>(0.) ? static_cast<Real>(0.) : (*aDirection)[index];
    }
}

bool DOTk_BoundConstraint::isFeasible(const std::shared_ptr<dotk::Vector<Real> > & aLowerBound,
                                      const std::shared_ptr<dotk::Vector<Real> > & aUpperBound,
                                      const std::shared_ptr<dotk::Vector<Real> > & aPrimal)
{
    size_t index = 0;
    bool is_stationary = true;

    while(index < aPrimal->size())
    {
        bool entry_lesser_than_lower_bound = (*aPrimal)[index] < (*aLowerBound)[index] ? true: false;
        bool entry_greater_than_upper_bound = (*aPrimal)[index] > (*aUpperBound)[index] ? true: false;
        if(entry_lesser_than_lower_bound || entry_greater_than_upper_bound)
        {
            is_stationary = false;
            break;
        }
        ++index;
    }

    return (is_stationary);
}

}
