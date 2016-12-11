/*
 * DOTk_RoutinesTypeLP.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */


#include "vector.hpp"
#include "DOTk_RoutinesTypeLP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

DOTk_RoutinesTypeLP::DOTk_RoutinesTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                         const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_) :
        m_ObjectiveFunction(objective_),
        m_InequalityConstraint(inequality_.begin(), inequality_.end())
{
}
DOTk_RoutinesTypeLP::~DOTk_RoutinesTypeLP()
{
}

Real DOTk_RoutinesTypeLP::objective(const std::tr1::shared_ptr<dotk::Vector<Real> > & control_)
{
    Real objective_function_value = m_ObjectiveFunction->value(*control_);
    DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();

    return (objective_function_value);
}

void DOTk_RoutinesTypeLP::gradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    m_ObjectiveFunction->gradient(*control_, *gradient_);
    dotk::DOTk_AssemblyManager::updateGradientEvaluationCounter();
}

Real DOTk_RoutinesTypeLP::inequalityBound(const size_t index_)
{
    Real value = m_InequalityConstraint[index_]->bound();
    return (value);
}

Real DOTk_RoutinesTypeLP::inequalityValue(const size_t index_,
                                          const std::tr1::shared_ptr<dotk::Vector<Real> > & control_)
{
    Real value = m_InequalityConstraint[index_]->value(*control_);
    return (value);
}

void DOTk_RoutinesTypeLP::inequalityGradient(const size_t index_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    m_InequalityConstraint[index_]->gradient(*control_, *gradient_);
    DOTk_AssemblyManager::updateInequalityConstraintGradientCounter();
}

}
