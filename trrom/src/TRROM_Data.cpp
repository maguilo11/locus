/*
 * TRROM_Data.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Dual.hpp"
#include "TRROM_State.hpp"
#include "TRROM_Slacks.hpp"
#include "TRROM_Control.hpp"

namespace trrom
{

Data::Data() :
        m_Dual(std::make_shared<trrom::Dual>()),
        m_State(std::make_shared<trrom::State>()),
        m_Slacks(std::make_shared<trrom::Slacks>()),
        m_Control(std::make_shared<trrom::Control>())
{
}

Data::~Data()
{
}

const std::shared_ptr<trrom::Vector<double> > & Data::dual() const
{
    return (m_Dual->data());
}

const std::shared_ptr<trrom::Vector<double> > & Data::state() const
{
    return (m_State->data());
}

const std::shared_ptr<trrom::Vector<double> > & Data::slacks() const
{
    return (m_Slacks->data());
}

const std::shared_ptr<trrom::Vector<double> > & Data::control() const
{
    return (m_Control->data());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getDualLowerBound() const
{
    return (m_Dual->lowerBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getDualUpperBound() const
{
    return (m_Dual->upperBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getStateLowerBound() const
{
    return (m_State->lowerBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getStateUpperBound() const
{
    return (m_State->upperBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getSlacksLowerBound() const
{
    return (m_Slacks->lowerBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getSlacksUpperBound() const
{
    return (m_Slacks->upperBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getControlLowerBound() const
{
    return (m_Control->lowerBound());
}

const std::shared_ptr<trrom::Vector<double> > & Data::getControlUpperBound() const
{
    return (m_Control->upperBound());
}

void Data::setDualLowerBound(double value_)
{
    m_Dual->setLowerBound(value_);
}

void Data::setDualUpperBound(double value_)
{
    m_Dual->setUpperBound(value_);
}

void Data::setDualLowerBound(const trrom::Vector<double> & lower_bound_)
{
    m_Dual->setLowerBound(lower_bound_);
}

void Data::setDualUpperBound(const trrom::Vector<double> & upper_bound_)
{
    m_Dual->setUpperBound(upper_bound_);
}

void Data::setStateLowerBound(double value_)
{
    m_State->setLowerBound(value_);
}

void Data::setStateUpperBound(double value_)
{
    m_State->setUpperBound(value_);
}

void Data::setStateLowerBound(const trrom::Vector<double> & lower_bound_)
{
    m_State->setLowerBound(lower_bound_);
}

void Data::setStateUpperBound(const trrom::Vector<double> & upper_bound_)
{
    m_State->setUpperBound(upper_bound_);
}

void Data::setSlacksLowerBound(double value_)
{
    m_Slacks->setLowerBound(value_);
}
void Data::setSlacksUpperBound(double value_)
{
    m_Slacks->setUpperBound(value_);
}
void Data::setSlacksLowerBound(const trrom::Vector<double> & lower_bound_)
{
    m_Slacks->setLowerBound(lower_bound_);
}
void Data::setSlacksUpperBound(const trrom::Vector<double> & upper_bound_)
{
    m_Slacks->setUpperBound(upper_bound_);
}

void Data::setControlLowerBound(double value_)
{
    m_Control->setLowerBound(value_);
}

void Data::setControlUpperBound(double value_)
{
    m_Control->setUpperBound(value_);
}

void Data::setControlLowerBound(const trrom::Vector<double> & lower_bound_)
{
    m_Control->setLowerBound(lower_bound_);
}

void Data::setControlUpperBound(const trrom::Vector<double> & upper_bound_)
{
    m_Control->setUpperBound(upper_bound_);
}

void Data::allocateDual(const trrom::Vector<double> & dual_)
{
    assert(m_Dual.get() != NULL);
    m_Dual = std::make_shared<trrom::Dual>(dual_);
}

void Data::allocateState(const trrom::Vector<double> & state_)
{
    assert(m_State.get() != NULL);
    m_State = std::make_shared<trrom::State>(state_);
}

void Data::allocateSlacks(const trrom::Vector<double> & slacks_)
{
    assert(m_Slacks.get() != NULL);
    m_Slacks = std::make_shared<trrom::Slacks>(slacks_);
    m_Slacks->setLowerBound(0);
    m_Slacks->setUpperBound(std::numeric_limits<double>::max());
}

void Data::allocateControl(const trrom::Vector<double> & control_)
{
    assert(m_Control.get() != NULL);
    m_Control = std::make_shared<trrom::Control>(control_);
}

}
