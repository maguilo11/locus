/*
 * DOTk_SubProblemCCSA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_SubProblemCCSA.hpp"

namespace dotk
{

DOTk_SubProblemCCSA::DOTk_SubProblemCCSA(dotk::ccsa::subproblem_t type_) :
        m_Type(type_),
        m_StoppingCriterion(dotk::ccsa::stopping_criterion_t::NOT_CONVERGED),
        m_IterationCount(0),
        m_MaxNumIterations(10),
        m_ResidualTolerance(1e-6),
        m_StagnationTolerance(1e-6),
        m_DualObjectiveTrialControlBoundScaling(0.5)
{
}

DOTk_SubProblemCCSA::~DOTk_SubProblemCCSA()
{
}

dotk::ccsa::subproblem_t DOTk_SubProblemCCSA::type() const
{
    return (m_Type);
}

dotk::ccsa::stopping_criterion_t DOTk_SubProblemCCSA::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

void DOTk_SubProblemCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t type_)
{
    m_StoppingCriterion = type_;
}

size_t DOTk_SubProblemCCSA::getIterationCount() const
{
    return (m_IterationCount);
}

void DOTk_SubProblemCCSA::resetIterationCount()
{
    m_IterationCount = 0;
}

void DOTk_SubProblemCCSA::updateIterationCount()
{
    m_IterationCount++;
}

size_t DOTk_SubProblemCCSA::getMaxNumIterations() const
{
    return (m_MaxNumIterations);
}

void DOTk_SubProblemCCSA::setMaxNumIterations(size_t input_)
{
    m_MaxNumIterations = input_;
}

Real DOTk_SubProblemCCSA::getResidualTolerance() const
{
    return (m_ResidualTolerance);
}

void DOTk_SubProblemCCSA::setResidualTolerance(Real tolerance_)
{
    m_ResidualTolerance = tolerance_;
}

Real DOTk_SubProblemCCSA::getStagnationTolerance() const
{
    return (m_StagnationTolerance);
}

void DOTk_SubProblemCCSA::setStagnationTolerance(Real tolerance_)
{
    m_StagnationTolerance = tolerance_;
}

Real DOTk_SubProblemCCSA::getDualObjectiveTrialControlBoundScaling() const
{
    return (m_DualObjectiveTrialControlBoundScaling);
}

void DOTk_SubProblemCCSA::setDualObjectiveTrialControlBoundScaling(Real input_)
{
    m_DualObjectiveTrialControlBoundScaling = input_;
}

}
