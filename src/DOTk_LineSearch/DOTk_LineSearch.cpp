/*
 * DOTk_LineSearch.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Types.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LineSearch::DOTk_LineSearch(dotk::types::line_search_t type_) :
        m_MaxNumLineSearchItr(50),
        m_NumLineSearchItrDone(0),
        m_StepSize(1.0),
        m_ContractionFactor(0.5),
        m_StepStagnationTol(1e-8),
        m_NewObjectiveFunctionValue(0.),
        m_OldObjectiveFunctionValue(0.),
        m_Type(type_)
{
}

DOTk_LineSearch::~DOTk_LineSearch()
{
}

size_t DOTk_LineSearch::getMaxNumLineSearchItr() const
{
    return (m_MaxNumLineSearchItr);
}

void DOTk_LineSearch::setMaxNumLineSearchItr(size_t itr_)
{
    m_MaxNumLineSearchItr = itr_;
}

size_t DOTk_LineSearch::getNumLineSearchItrDone() const
{
    return (m_NumLineSearchItrDone);
}

void DOTk_LineSearch::setNumLineSearchItrDone(size_t itr_)
{
    m_NumLineSearchItrDone = itr_;
}

Real DOTk_LineSearch::getStepSize() const
{
    return (m_StepSize);
}

void DOTk_LineSearch::setStepSize(Real value_)
{
    m_StepSize = value_;
}

Real DOTk_LineSearch::getConstant() const
{
    return (0);
}

void DOTk_LineSearch::setConstant(Real value_)
{
}

Real DOTk_LineSearch::getStepStagnationTol() const
{
    return (m_StepStagnationTol);
}

void DOTk_LineSearch::setStepStagnationTol(Real tol_)
{
    m_StepStagnationTol = tol_;
}

Real DOTk_LineSearch::getContractionFactor() const
{
    return (m_ContractionFactor);
}

void DOTk_LineSearch::setContractionFactor(Real value_)
{
    m_ContractionFactor = value_;
}

void DOTk_LineSearch::setOldObjectiveFunctionValue(Real value_)
{
    m_OldObjectiveFunctionValue = value_;
}

Real DOTk_LineSearch::getOldObjectiveFunctionValue() const
{
    return (m_OldObjectiveFunctionValue);
}

void DOTk_LineSearch::setNewObjectiveFunctionValue(Real value_)
{
    m_NewObjectiveFunctionValue = value_;
}

Real DOTk_LineSearch::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunctionValue);
}

dotk::types::line_search_t DOTk_LineSearch::type() const
{
    return (m_Type);
}

}
