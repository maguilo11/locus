/*
 * DOTk_Primal.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>

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

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::dual() const
{
    return (m_Dual->data());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::state() const
{
    return (m_State->data());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::control() const
{
    return (m_Control->data());
}

const std::tr1::shared_ptr<dotk::DOTk_Dual> & DOTk_Primal::dualStruc() const
{
    return (m_Dual);
}

const std::tr1::shared_ptr<dotk::DOTk_State> & DOTk_Primal::stateStruc() const
{
    return (m_State);
}

const std::tr1::shared_ptr<dotk::DOTk_Control> & DOTk_Primal::controlStruc() const
{
    return (m_Control);
}

dotk::types::variable_t DOTk_Primal::type() const
{
    return (dotk::types::PRIMAL);
}

size_t DOTk_Primal::getDualBasisSize() const
{
    return (m_Dual->getDualBasisSize());
}

size_t DOTk_Primal::getStateBasisSize() const
{
    return (m_State->getStateBasisSize());
}

size_t DOTk_Primal::getControlBasisSize() const
{
    return (m_Control->getControlBasisSize());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::getDualLowerBound() const
{
    return (m_Dual->lowerBound());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::getDualUpperBound() const
{
    return (m_Dual->upperBound());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::getStateLowerBound() const
{
    return (m_State->lowerBound());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::getStateUpperBound() const
{
    return (m_State->upperBound());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::getControlLowerBound() const
{
    return (m_Control->lowerBound());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Primal::getControlUpperBound() const
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

void DOTk_Primal::setDualLowerBound(const dotk::vector<Real> & lower_bound_)
{
    m_Dual->setLowerBound(lower_bound_);
}

void DOTk_Primal::setDualUpperBound(const dotk::vector<Real> & upper_bound_)
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

void DOTk_Primal::setStateLowerBound(const dotk::vector<Real> & lower_bound_)
{
    m_State->setLowerBound(lower_bound_);
}

void DOTk_Primal::setStateUpperBound(const dotk::vector<Real> & upper_bound_)
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

void DOTk_Primal::setControlLowerBound(const dotk::vector<Real> & lower_bound_)
{
    m_Control->setLowerBound(lower_bound_);
}

void DOTk_Primal::setControlUpperBound(const dotk::vector<Real> & upper_bound_)
{
    m_Control->setUpperBound(upper_bound_);
}

void DOTk_Primal::allocateUserDefinedDual(const dotk::vector<Real> & dual_)
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

void DOTk_Primal::allocateUserDefinedState(const dotk::vector<Real> & state_)
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

void DOTk_Primal::allocateUserDefinedControl(const dotk::vector<Real> & control_)
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