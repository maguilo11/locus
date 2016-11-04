/*
 * DOTk_SteihaugTointNewton.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_SteihaugTointNewton.hpp"

namespace dotk
{

DOTk_SteihaugTointNewton::DOTk_SteihaugTointNewton() :
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_ObjectiveTolerance(1e-10),
        m_ActualReductionTolerance(1e-10),
        m_MaxNumOptimizationItr(100),
        m_NumOptimizationItrDone(0),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED)

{
}

DOTk_SteihaugTointNewton::~DOTk_SteihaugTointNewton()
{
}

void DOTk_SteihaugTointNewton::setGradientTolerance(Real input_)
{
    m_GradientTolerance = input_;
}

Real DOTk_SteihaugTointNewton::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

void DOTk_SteihaugTointNewton::setTrialStepTolerance(Real input_)
{
    m_TrialStepTolerance = input_;
}

Real DOTk_SteihaugTointNewton::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

void DOTk_SteihaugTointNewton::setObjectiveTolerance(Real input_)
{
    m_ObjectiveTolerance = input_;
}

Real DOTk_SteihaugTointNewton::getObjectiveTolerance() const
{
    return (m_ObjectiveTolerance);
}

void DOTk_SteihaugTointNewton::setActualReductionTolerance(Real input_)
{
    m_ActualReductionTolerance = input_;
}

Real DOTk_SteihaugTointNewton::getActualReductionTolerance() const
{
    return (m_ActualReductionTolerance);
}

void DOTk_SteihaugTointNewton::setNumOptimizationItrDone(size_t input_)
{
    m_NumOptimizationItrDone = input_;
}

size_t DOTk_SteihaugTointNewton::getNumOptimizationItrDone() const
{
    return (m_NumOptimizationItrDone);
}

void DOTk_SteihaugTointNewton::setMaxNumOptimizationItr(size_t input_)
{
    m_MaxNumOptimizationItr = input_;
}

size_t DOTk_SteihaugTointNewton::getMaxNumOptimizationItr() const
{
    return (m_MaxNumOptimizationItr);
}

void DOTk_SteihaugTointNewton::setStoppingCriterion(dotk::types::stop_criterion_t input_)
{
    m_StoppingCriterion = input_;
}

dotk::types::stop_criterion_t DOTk_SteihaugTointNewton::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

}
