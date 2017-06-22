/*
 * DOTk_Primal.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>

#include "vector.hpp"
#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Control.hpp"

namespace dotk
{

DOTk_Primal::DOTk_Primal() :
        m_Dual(new dotk::DOTk_Dual),
        m_State(new dotk::DOTk_State),
        m_Control(new dotk::DOTk_Control)
{
}

DOTk_Primal::~DOTk_Primal()
{
}

dotk::types::variable_t DOTk_Primal::type() const
{
    return (dotk::types::PRIMAL);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::dual() const
{
    return (m_Dual->data());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::state() const
{
    return (m_State->data());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::control() const
{
    return (m_Control->data());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::getDualLowerBound() const
{
    return (m_Dual->lowerBound());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::getDualUpperBound() const
{
    return (m_Dual->upperBound());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::getStateLowerBound() const
{
    return (m_State->lowerBound());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::getStateUpperBound() const
{
    return (m_State->upperBound());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::getControlLowerBound() const
{
    return (m_Control->lowerBound());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_Primal::getControlUpperBound() const
{
    return (m_Control->upperBound());
}

void DOTk_Primal::setDualBasisSize(size_t size_)
{
    m_Dual->setDualBasisSize(size_);
}

void DOTk_Primal::setDualLowerBound(Real value_)
{
    m_Dual->setLowerBound(value_);
}

void DOTk_Primal::setDualUpperBound(Real value_)
{
    m_Dual->setUpperBound(value_);
}

void DOTk_Primal::setDualLowerBound(const dotk::Vector<Real> & lower_bound_)
{
    m_Dual->setLowerBound(lower_bound_);
}

void DOTk_Primal::setDualUpperBound(const dotk::Vector<Real> & upper_bound_)
{
    m_Dual->setUpperBound(upper_bound_);
}

void DOTk_Primal::setStateBasisSize(size_t size_)
{
    m_State->setStateBasisSize(size_);
}

void DOTk_Primal::setStateLowerBound(Real value_)
{
    m_State->setLowerBound(value_);
}

void DOTk_Primal::setStateUpperBound(Real value_)
{
    m_State->setUpperBound(value_);
}

void DOTk_Primal::setStateLowerBound(const dotk::Vector<Real> & lower_bound_)
{
    m_State->setLowerBound(lower_bound_);
}

void DOTk_Primal::setStateUpperBound(const dotk::Vector<Real> & upper_bound_)
{
    m_State->setUpperBound(upper_bound_);
}

void DOTk_Primal::setControlBasisSize(size_t size_)
{
    m_Control->setControlBasisSize(size_);
}

void DOTk_Primal::setControlLowerBound(Real value_)
{
    m_Control->setLowerBound(value_);
}

void DOTk_Primal::setControlUpperBound(Real value_)
{
    m_Control->setUpperBound(value_);
}

void DOTk_Primal::setControlLowerBound(const dotk::Vector<Real> & lower_bound_)
{
    m_Control->setLowerBound(lower_bound_);
}

void DOTk_Primal::setControlUpperBound(const dotk::Vector<Real> & upper_bound_)
{
    m_Control->setUpperBound(upper_bound_);
}

void DOTk_Primal::allocateUserDefinedDual(const dotk::Vector<Real> & dual_)
{
    assert(m_Dual.get() != nullptr);
    m_Dual.reset(new dotk::DOTk_Dual(dual_));
}

void DOTk_Primal::allocateSerialDualArray(size_t size_, Real value_)
{
    assert(m_Dual.get() != nullptr);
    m_Dual->allocateSerialArray(size_, value_);
}

void DOTk_Primal::allocateSerialDualVector(size_t size_, Real value_)
{
    assert(m_Dual.get() != nullptr);
    m_Dual->allocateSerialVector(size_, value_);
}

void DOTk_Primal::allocateUserDefinedState(const dotk::Vector<Real> & state_)
{
    assert(m_State.get() != nullptr);
    m_State.reset(new dotk::DOTk_State(state_));
}

void DOTk_Primal::allocateSerialStateArray(size_t size_, Real value_)
{
    assert(m_State.get() != nullptr);
    m_State->allocateSerialArray(size_, value_);
}

void DOTk_Primal::allocateSerialStateVector(size_t size_, Real value_)
{
    assert(m_State.get() != nullptr);
    m_State->allocateSerialVector(size_, value_);
}

void DOTk_Primal::allocateUserDefinedControl(const dotk::Vector<Real> & control_)
{
    assert(m_Control.get() != nullptr);
    m_Control.reset(new dotk::DOTk_Control(control_));
}

void DOTk_Primal::allocateSerialControlArray(size_t size_, Real value_)
{
    assert(m_Control.get() != nullptr);
    m_Control->allocateSerialArray(size_, value_);
}

void DOTk_Primal::allocateSerialControlVector(size_t size_, Real value_)
{
    assert(m_Control.get() != nullptr);
    m_Control->allocateSerialVector(size_, value_);
}

}
