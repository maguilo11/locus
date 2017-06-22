/*
 * DOTk_LineSearchAlgorithmsDataMng.cpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_LineSearchAlgorithmsDataMng::DOTk_LineSearchAlgorithmsDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_OptimizationDataMng(primal_),
        m_RoutinesMng(),
        m_FirstOrderOperator()
{
    this->setUserDefinedGradient();
}

DOTk_LineSearchAlgorithmsDataMng::~DOTk_LineSearchAlgorithmsDataMng()
{
}

size_t DOTk_LineSearchAlgorithmsDataMng::getObjectiveFuncEvalCounter() const
{
    return (m_RoutinesMng->getObjectiveFunctionEvaluationCounter());
}

size_t DOTk_LineSearchAlgorithmsDataMng::getAdjointInverseJacobianWrtStateCounter() const
{
    return (m_RoutinesMng->getAdjointInverseJacobianStateCounter());
}

size_t DOTk_LineSearchAlgorithmsDataMng::getHessianEvaluationCounter() const
{
    return (m_RoutinesMng->getHessianEvaluationCounter());
}

size_t DOTk_LineSearchAlgorithmsDataMng::getGradientEvaluationCounter() const
{
    return (m_RoutinesMng->getGradientEvaluationCounter());
}

size_t DOTk_LineSearchAlgorithmsDataMng::getEqualityConstraintEvaluationCounter() const
{
    return (m_RoutinesMng->getEqualityConstraintEvaluationCounter());
}

void DOTk_LineSearchAlgorithmsDataMng::setUserDefinedGradient()
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildUserDefinedGradient(m_FirstOrderOperator);
}

void DOTk_LineSearchAlgorithmsDataMng::computeGradient()
{
    m_FirstOrderOperator->gradient(this);
}

Real DOTk_LineSearchAlgorithmsDataMng::evaluateObjective()
{
    Real value = m_RoutinesMng->objective(this->getNewPrimal());
    return (value);
}

Real DOTk_LineSearchAlgorithmsDataMng::evaluateObjective(const std::shared_ptr<dotk::Vector<Real> > & input_)
{
    Real value = m_RoutinesMng->objective(input_);
    return (value);
}

void DOTk_LineSearchAlgorithmsDataMng::computeGradient(const std::shared_ptr<dotk::Vector<Real> > & input_,
                                                       const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    m_RoutinesMng->gradient(input_, gradient_);
}

const std::shared_ptr<dotk::DOTk_AssemblyManager> & DOTk_LineSearchAlgorithmsDataMng::getRoutinesMng() const
{
    return (m_RoutinesMng);
}

}
