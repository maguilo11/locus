/*
 * DOTk_DualSolverCCSA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_DualSolverCCSA.hpp"

namespace dotk
{

DOTk_DualSolverCCSA::DOTk_DualSolverCCSA(dotk::ccsa::dual_solver_t type_) :
        m_DualSolverType(type_),
        m_StoppingCriterion(dotk::ccsa::stopping_criterion_t::NOT_CONVERGED),
        m_IterationCount(0),
        m_MaxNumIterations(10),
        m_LineSearchIterationCount(0),
        m_MaxNumLineSearchIterations(5),
        m_GradientTolerance(1e-8),
        m_TrialStepTolerance(1e-8),
        m_LineSearchStepLowerBound(1e-3),
        m_LineSearchStepUpperBound(0.5),
        m_ObjectiveStagnationTolerance(1e-8)
{
}

DOTk_DualSolverCCSA::~DOTk_DualSolverCCSA()
{
}

dotk::ccsa::dual_solver_t DOTk_DualSolverCCSA::getDualSolverType() const
{
    return (m_DualSolverType);
}

dotk::ccsa::stopping_criterion_t DOTk_DualSolverCCSA::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

void DOTk_DualSolverCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t type_)
{
    m_StoppingCriterion = type_;
}

size_t DOTk_DualSolverCCSA::getIterationCount() const
{
    return (m_IterationCount);
}

void DOTk_DualSolverCCSA::resetIterationCount()
{
    m_IterationCount = 0;
}

void DOTk_DualSolverCCSA::updateIterationCount()
{
    m_IterationCount++;
}

size_t DOTk_DualSolverCCSA::getMaxNumIterations() const
{
    return (m_MaxNumIterations);
}

void DOTk_DualSolverCCSA::setMaxNumIterations(size_t input_)
{
    m_MaxNumIterations = input_;
}

Real DOTk_DualSolverCCSA::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

void DOTk_DualSolverCCSA::setGradientTolerance(Real tolerance_)
{
    m_GradientTolerance = tolerance_;
}

Real DOTk_DualSolverCCSA::getObjectiveStagnationTolerance() const
{
    return (m_ObjectiveStagnationTolerance);
}

void DOTk_DualSolverCCSA::setObjectiveStagnationTolerance(Real tolerance_)
{
    m_ObjectiveStagnationTolerance = tolerance_;
}

Real DOTk_DualSolverCCSA::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

void DOTk_DualSolverCCSA::setTrialStepTolerance(Real tolerance_)
{
    m_TrialStepTolerance = tolerance_;
}

size_t DOTk_DualSolverCCSA::getLineSearchIterationCount() const
{
    return (m_LineSearchIterationCount);
}

void DOTk_DualSolverCCSA::resetLineSearchIterationCount()
{
    m_LineSearchIterationCount = 0;
}

void DOTk_DualSolverCCSA::updateLineSearchIterationCount()
{
    m_LineSearchIterationCount++;
}

size_t DOTk_DualSolverCCSA::getMaxNumLineSearchIterations() const
{
    return (m_MaxNumLineSearchIterations);
}

void DOTk_DualSolverCCSA::setMaxNumLineSearchIterations(size_t iterations_)
{
    m_MaxNumLineSearchIterations = iterations_;
}

Real DOTk_DualSolverCCSA::getLineSearchStepLowerBound() const
{
    return (m_LineSearchStepLowerBound);
}

void DOTk_DualSolverCCSA::setLineSearchStepLowerBound(Real input_)
{
    m_LineSearchStepLowerBound = input_;
}

Real DOTk_DualSolverCCSA::getLineSearchStepUpperBound() const
{
    return (m_LineSearchStepUpperBound);
}
void DOTk_DualSolverCCSA::setLineSearchStepUpperBound(Real input_)
{
    m_LineSearchStepUpperBound = input_;
}

}
