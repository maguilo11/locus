/*
 * DOTk_BoundConstraint.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_BoundConstraint.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_BoundConstraint::DOTk_BoundConstraint(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           dotk::types::constraint_method_t type_) :
        m_Active(true),
        m_StepSize(1.0),
        m_ContractionStep(0.5),
        m_StagnationTolerance(1e-8),
        m_NewObjectiveFunctionValue(0.),
        m_NumFeasibleItr(0),
        m_MaxNumFeasibleItr(10),
        m_StepType(dotk::types::bound_step_t::ARMIJO_STEP),
        m_Type(type_),
        m_ActiveSet(primal_->control()->clone())
{
}

DOTk_BoundConstraint::~DOTk_BoundConstraint()
{
}

void DOTk_BoundConstraint::setStepSize(Real value_)
{
    m_StepSize = value_;
}

Real DOTk_BoundConstraint::getStepSize() const
{
    return (m_StepSize);
}

void DOTk_BoundConstraint::setStagnationTolerance(Real tol_)
{
    m_StagnationTolerance = tol_;
}

Real DOTk_BoundConstraint::getStagnationTolerance() const
{
    return (m_StagnationTolerance);
}

void DOTk_BoundConstraint::setContractionStep(Real value_)
{
    m_ContractionStep = value_;
}

Real DOTk_BoundConstraint::getContractionStep() const
{
    return (m_ContractionStep);
}

void DOTk_BoundConstraint::setNewObjectiveFunctionValue(Real value_)
{
    m_NewObjectiveFunctionValue = value_;
}

Real DOTk_BoundConstraint::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunctionValue);
}

void DOTk_BoundConstraint::setMaxNumFeasibleItr(size_t itr_)
{
    m_MaxNumFeasibleItr = itr_;
}

size_t DOTk_BoundConstraint::getMaxNumFeasibleItr() const
{
    return (m_MaxNumFeasibleItr);
}

void DOTk_BoundConstraint::setNumFeasibleItr(size_t itr_)
{
    m_NumFeasibleItr = itr_;
}

size_t DOTk_BoundConstraint::getNumFeasibleItr() const
{
    return (m_NumFeasibleItr);
}

void DOTk_BoundConstraint::activate(bool enable_)
{
    m_Active = enable_;
}

bool DOTk_BoundConstraint::active() const
{
    return (m_Active);
}

void DOTk_BoundConstraint::setStepType(dotk::types::bound_step_t type_)
{
    m_StepType = type_;
}

dotk::types::bound_step_t DOTk_BoundConstraint::getStepType() const
{
    return (m_StepType);
}

dotk::types::constraint_method_t DOTk_BoundConstraint::type() const
{
    return (m_Type);
}

Real DOTk_BoundConstraint::getArmijoStep(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                         const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    step_->step(mng_);
    Real new_objective_function_value = step_->getNewObjectiveFunctionValue();
    this->setNewObjectiveFunctionValue(new_objective_function_value);
    Real step = step_->getStepSize();

    return (step);
}

Real DOTk_BoundConstraint::getMinReductionStep(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    size_t itr = 1;
    Real step = 1.;
    Real new_objective_function_value = 0.;
    Real current_objective_func_val = mng_->getOldObjectiveFunctionValue();
    while(1)
    {
        mng_->getNewPrimal()->axpy(step, *mng_->getTrialStep());
        new_objective_function_value = mng_->evaluateObjective(mng_->getNewPrimal());
        bool is_new_fval_less_than_current_fval =
                new_objective_function_value < current_objective_func_val ? true: false;
        if((is_new_fval_less_than_current_fval == true) || (itr == this->getMaxNumFeasibleItr()))
        {
            break;
        }
        mng_->getNewPrimal()->copy(*mng_->getOldPrimal());
        step = step * this->getContractionStep();
        ++ itr;
    }
    this->setNumFeasibleItr(itr);
    this->setNewObjectiveFunctionValue(new_objective_function_value);

    return (step);
}

Real DOTk_BoundConstraint::getStep(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                   const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real step = 0.;
    switch(this->getStepType())
    {
        case dotk::types::bound_step_t::ARMIJO_STEP:
        {
            step = this->getArmijoStep(step_,mng_);
            break;
        }
        case dotk::types::bound_step_t::MIN_REDUCTION_STEP:
        {
            step = this->getMinReductionStep(mng_);
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

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_BoundConstraint::activeSet() const
{
    return (m_ActiveSet);
}

void DOTk_BoundConstraint::computeScaledTrialStep(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                                  const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                  const std::tr1::shared_ptr<dotk::vector<Real> > & current_primal_)
{
    dotk::types::bound_step_t type = this->getStepType();
    switch(type)
    {
        case dotk::types::bound_step_t::ARMIJO_STEP:
        case dotk::types::bound_step_t::MIN_REDUCTION_STEP:
        {
            Real current_objective_function_value = mng_->getNewObjectiveFunctionValue();
            Real step = this->getStep(step_, mng_);
            mng_->getTrialStep()->scale(step);
            mng_->getNewPrimal()->copy(*current_primal_);
            mng_->setNewObjectiveFunctionValue(current_objective_function_value);
            break;
        }
        case dotk::types::bound_step_t::CONSTANT_STEP:
        {
            Real step = this->getStep(step_, mng_);
            mng_->getTrialStep()->scale(step);
            break;
        }
        case dotk::types::bound_step_t::TRUST_REGION_STEP:
        {
            break;
        }
    }
}

void DOTk_BoundConstraint::project(const std::tr1::shared_ptr<dotk::vector<Real> > & lower_bound_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & upper_bound_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & primal_)
{
    m_ActiveSet->fill(0);
    for(size_t index = 0; index < primal_->size(); ++index)
    {
        (*primal_)[index] = std::min((*upper_bound_)[index], (*primal_)[index]);
        (*primal_)[index] = std::max((*lower_bound_)[index], (*primal_)[index]);
        (*m_ActiveSet)[index] = static_cast<size_t>(((*primal_)[index] == (*lower_bound_)[index])
                || ((*primal_)[index] == (*upper_bound_)[index]));
    }
}

void DOTk_BoundConstraint::computeActiveSet(const std::tr1::shared_ptr<dotk::vector<Real> > & lower_bound_,
                                            const std::tr1::shared_ptr<dotk::vector<Real> > & upper_bound_,
                                            const std::tr1::shared_ptr<dotk::vector<Real> > & primal_)
{
    assert(m_ActiveSet->size() == primal_->size());

    m_ActiveSet->fill(0);
    for(size_t index = 0; index < m_ActiveSet->size(); ++index)
    {
        (*m_ActiveSet)[index] = static_cast<size_t>(((*primal_)[index] <= (*lower_bound_)[index])
                || ((*primal_)[index] >= (*upper_bound_)[index]));
    }
}

void DOTk_BoundConstraint::pruneActive(const std::tr1::shared_ptr<dotk::vector<Real> > & direction_)
{
    for(size_t index = 0; index < m_ActiveSet->size(); ++ index)
    {
        (*direction_)[index] =
                (*m_ActiveSet)[index] > static_cast<Real>(0.) ? static_cast<Real>(0.) : (*direction_)[index];
    }
}

bool DOTk_BoundConstraint::isFeasible(const std::tr1::shared_ptr<dotk::vector<Real> > & lower_bound_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & upper_bound_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & primal_)
{
    size_t index = 0;
    bool is_stationary = true;

    while(index < primal_->size())
    {
        bool entry_lesser_than_lower_bound = (*primal_)[index] < (*lower_bound_)[index] ? true: false;
        bool entry_greater_than_upper_bound = (*primal_)[index] > (*upper_bound_)[index] ? true: false;
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
