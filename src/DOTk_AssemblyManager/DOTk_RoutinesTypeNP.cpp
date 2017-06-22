/*
 * DOTk_RoutinesTypeNP.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */


#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RoutinesTypeNP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

DOTk_RoutinesTypeNP::DOTk_RoutinesTypeNP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                         const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                         const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                                         const std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_) :
        m_State(),
        m_StateWorkVec(),
        m_ControlWorkVec(),
        m_EqualityConstraintDual(),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(equality_),
        m_InequalityConstraint(inequality_.begin(), inequality_.end())
{
    this->initialize(primal_);
}

DOTk_RoutinesTypeNP::~DOTk_RoutinesTypeNP()
{
}

Real DOTk_RoutinesTypeNP::objective(const std::shared_ptr<dotk::Vector<Real> > & control_)
{
    m_State->fill(0.);
    m_StateWorkVec->fill(0.);
    m_EqualityConstraint->solve(*control_, (*m_State));
    DOTk_AssemblyManager::updateEqualityConstraintEvaluationCounter();

    Real objective_function_value = m_ObjectiveFunction->value(*m_State, *control_);
    DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();

    return (objective_function_value);
}

void DOTk_RoutinesTypeNP::gradient(const std::shared_ptr<dotk::Vector<Real> > & control_,
                                   const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    m_StateWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeState(*m_State, *control_, *m_StateWorkVec);
    m_StateWorkVec->scale(static_cast<Real>(-1.0));

    // solve adjoint problem
    m_EqualityConstraintDual->fill(0.);
    m_EqualityConstraint->applyAdjointInverseJacobianState(*m_State,
                                                           *control_,
                                                           *m_StateWorkVec,
                                                           *m_EqualityConstraintDual);
    DOTk_AssemblyManager::updateAdjointInverseJacobianStateCounter();

    // get equality constraint contribution to the gradient operator
    m_ControlWorkVec->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeControl(*m_State,
                                                          *control_,
                                                          *m_EqualityConstraintDual,
                                                          *m_ControlWorkVec);

    // assemble gradient operator
    gradient_->update(1., *m_ControlWorkVec, 0.);
    m_ControlWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeControl(*m_State, *control_, *m_ControlWorkVec);
    gradient_->update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);
    DOTk_AssemblyManager::updateGradientEvaluationCounter();
}

Real DOTk_RoutinesTypeNP::inequalityBound(const size_t index_)
{
    Real value = m_InequalityConstraint[index_]->bound();
    return (value);
}

Real DOTk_RoutinesTypeNP::inequalityValue(const size_t index_,
                                             const std::shared_ptr<dotk::Vector<Real> > & control_)
{
    Real value = m_InequalityConstraint[index_]->value(*m_State, *control_);
    return (value);
}

void DOTk_RoutinesTypeNP::inequalityGradient(const size_t index_,
                                             const std::shared_ptr<dotk::Vector<Real> > & control_,
                                             const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    gradient_->fill(0.);
    m_InequalityConstraint[index_]->partialDerivativeControl(*m_State, *control_, *gradient_);
    DOTk_AssemblyManager::updateInequalityConstraintGradientCounter();
}

void DOTk_RoutinesTypeNP::initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->state().use_count() > 0)
    {
        m_State = primal_->state()->clone();
        m_StateWorkVec = primal_->state()->clone();
        m_EqualityConstraintDual = primal_->state()->clone();
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> STATE vector is NULL. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }

    if(primal_->control().use_count() > 0)
    {
        m_ControlWorkVec = primal_->control()->clone();
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> CONTROL vector is NULL. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }
}

}
