/*
 * DOTk_OptimalityCriteriaDataMng.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>
#include <cassert>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_OptimalityCriteriaDataMng.hpp"

namespace dotk
{

DOTk_OptimalityCriteriaDataMng::DOTk_OptimalityCriteriaDataMng
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        m_MoveLimit(0.01),
        m_InequalityDual(0),
        m_DampingParameter(0.5),
        m_GradientTolerance(1e-8),
        m_BisectionTolerance(1e-4),
        m_FeasibilityTolerance(1e-8),
        m_NewObjectiveFunctionValue(0),
        m_OldObjectiveFunctionValue(0),
        m_ControlStagnationTolerance(1e-3),
        m_InequalityConstraintResidual(std::numeric_limits<Real>::max()),
        m_MaxControlRelativeDifference(std::numeric_limits<Real>::max()),
        m_NormObjectiveFunctionGradient(std::numeric_limits<Real>::max()),
        m_InequalityConstraintDualLowerBound(0),
        m_InequalityConstraintDualUpperBound(1e4),
        m_MaxNumOptimizationItr(100),
        m_State(primal_->state()->clone()),
        m_OldControl(primal_->control()->clone()),
        m_NewControl(primal_->control()->clone()),
        m_ControlLowerBound(primal_->control()->clone()),
        m_ControlUpperBound(primal_->control()->clone()),
        m_ObjectiveGradient(primal_->control()->clone()),
        m_InequalityGradient(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_OptimalityCriteriaDataMng::~DOTk_OptimalityCriteriaDataMng()
{
}

Real DOTk_OptimalityCriteriaDataMng::getMoveLimit() const
{
    return (m_MoveLimit);
}

void DOTk_OptimalityCriteriaDataMng::setMoveLimit(Real value_)
{
    m_MoveLimit = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getInequalityDual() const
{
    return (m_InequalityDual);
}

void DOTk_OptimalityCriteriaDataMng::setInequalityDual(Real value_)
{
    m_InequalityDual = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getDampingParameter() const
{
    return (m_DampingParameter);
}

void DOTk_OptimalityCriteriaDataMng::setDampingParameter(Real value_)
{
    m_DampingParameter = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

void DOTk_OptimalityCriteriaDataMng::setGradientTolerance(Real value_)
{
    m_GradientTolerance = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getBisectionTolerance() const
{
    return (m_BisectionTolerance);
}

void DOTk_OptimalityCriteriaDataMng::setBisectionTolerance(Real value_)
{
    m_BisectionTolerance = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getFeasibilityTolerance() const
{
    return (m_FeasibilityTolerance);
}

void DOTk_OptimalityCriteriaDataMng::setFeasibilityTolerance(Real value_)
{
    m_FeasibilityTolerance = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getOldObjectiveFunctionValue() const
{
    return (m_OldObjectiveFunctionValue);
}

void DOTk_OptimalityCriteriaDataMng::setOldObjectiveFunctionValue(Real value_)
{
    m_OldObjectiveFunctionValue = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunctionValue);
}

void DOTk_OptimalityCriteriaDataMng::setNewObjectiveFunctionValue(Real value_)
{
    m_NewObjectiveFunctionValue = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getInequalityConstraintResidual() const
{
    return (m_InequalityConstraintResidual);
}

void DOTk_OptimalityCriteriaDataMng::setInequalityConstraintResidual(Real value_)
{
    m_InequalityConstraintResidual = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getMaxControlRelativeDifference() const
{
    return (m_MaxControlRelativeDifference);
}

void DOTk_OptimalityCriteriaDataMng::setMaxControlRelativeDifference(Real value_)
{
    m_MaxControlRelativeDifference = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getNormObjectiveFunctionGradient() const
{
    return (m_NormObjectiveFunctionGradient);
}

void DOTk_OptimalityCriteriaDataMng::setNormObjectiveFunctionGradient(Real value_)
{
    m_NormObjectiveFunctionGradient = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getInequalityConstraintDualLowerBound() const
{
    return (m_InequalityConstraintDualLowerBound);
}

void DOTk_OptimalityCriteriaDataMng::setInequalityConstraintDualLowerBound(Real value_)
{
    m_InequalityConstraintDualLowerBound = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getInequalityConstraintDualUpperBound() const
{
    return (m_InequalityConstraintDualUpperBound);
}

void DOTk_OptimalityCriteriaDataMng::setInequalityConstraintDualUpperBound(Real value_)
{
    m_InequalityConstraintDualUpperBound = value_;
}

Real DOTk_OptimalityCriteriaDataMng::getControlStagnationTolerance() const
{
    return (m_ControlStagnationTolerance);
}

void DOTk_OptimalityCriteriaDataMng::setControlStagnationTolerance(Real value_)
{
    m_ControlStagnationTolerance = value_;
}

size_t DOTk_OptimalityCriteriaDataMng::getMaxNumOptimizationItr() const
{
    return (m_MaxNumOptimizationItr);
}

void DOTk_OptimalityCriteriaDataMng::setMaxNumOptimizationItr(size_t value_)
{
    m_MaxNumOptimizationItr = value_;
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getState() const
{
    return (*m_State);
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getOldControl() const
{
    return (*m_OldControl);
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getNewControl() const
{
    return (*m_NewControl);
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getControlLowerBound() const
{
    return (*m_ControlLowerBound);
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getControlUpperBound() const
{
    return (*m_ControlUpperBound);
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getObjectiveGradient() const
{
    return (*m_ObjectiveGradient);
}

dotk::Vector<Real> & DOTk_OptimalityCriteriaDataMng::getInequalityGradient() const
{
    return (*m_InequalityGradient);
}

void DOTk_OptimalityCriteriaDataMng::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    assert(primal_->state().use_count() > 0);
    assert(primal_->control().use_count() > 0);
    m_State->copy(*primal_->state());
    m_NewControl->copy(*primal_->control());

    assert(primal_->getControlLowerBound().use_count() > 0);
    assert(primal_->getControlUpperBound().use_count() > 0);
    m_ControlLowerBound->copy(*primal_->getControlLowerBound());
    m_ControlUpperBound->copy(*primal_->getControlUpperBound());

    m_OldControl->fill(0.);
    m_ObjectiveGradient->fill(0.);
    m_InequalityGradient->fill(0.);
}

}
