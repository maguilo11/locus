/*
 * DOTk_PrimalVector.cpp
 *
 *  Created on: Dec 20, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "DOTk_Primal.hpp"
#include "DOTk_PrimalVector.hpp"

namespace dotk
{

template<typename ScalarType>
DOTk_PrimalVector<ScalarType>::DOTk_PrimalVector(const dotk::DOTk_Primal & primal_) :
        m_Size(0),
        m_State(),
        m_Control()
{
    this->initialize(primal_);
}

template<typename ScalarType>
DOTk_PrimalVector<ScalarType>::DOTk_PrimalVector(const dotk::Vector<ScalarType> & control_) :
        m_Size(control_.size()),
        m_State(),
        m_Control(control_.clone())
{
    m_Control->update(1., control_, 0.);
}

template<typename ScalarType>
DOTk_PrimalVector<ScalarType>::DOTk_PrimalVector(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & state_) :
        m_Size(control_.size() + state_.size()),
        m_State(state_.clone()),
        m_Control(control_.clone())
{
    this->initialize(control_, state_);
}

template<typename ScalarType>
DOTk_PrimalVector<ScalarType>::~DOTk_PrimalVector()
{
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::scale(const ScalarType & alpha_)
{
    m_Control->scale(alpha_);
    if(m_State.use_count() > 0)
    {
        m_State->scale(alpha_);
    }
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::elementWiseMultiplication(const dotk::Vector<ScalarType> & input_)
{
    m_Control->elementWiseMultiplication(*input_.control());
    if(m_State.use_count() > 0)
    {
        m_State->elementWiseMultiplication(*input_.state());
    }
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::update(const ScalarType & alpha_,
                                           const dotk::Vector<ScalarType> & input_,
                                           const ScalarType & beta_)
{
    m_Control->update(alpha_, *input_.control(), beta_);
    if(m_State.use_count() > 0)
    {
        m_State->update(alpha_, *input_.state(), beta_);
    }
}

template<typename ScalarType>
ScalarType DOTk_PrimalVector<ScalarType>::max() const
{
    if(m_State.use_count() > 0)
    {
        ScalarType state_max = m_State->max();
        ScalarType control_max = m_Control->max();
        ScalarType max_value = std::max(state_max, control_max);
        return (max_value);
    }
    else
    {
        ScalarType max_value = m_Control->max();
        return (max_value);
    }
}

template<typename ScalarType>
ScalarType DOTk_PrimalVector<ScalarType>::min() const
{
    if(m_State.use_count() > 0)
    {
        ScalarType state_min = m_State->min();
        ScalarType control_min = m_Control->min();
        ScalarType min_value = std::min(state_min, control_min);
        return (min_value);
    }
    else
    {
        ScalarType min_value = m_Control->min();
        return (min_value);
    }
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::abs()
{
    m_Control->abs();
    if(m_State.use_count() > 0)
    {
        m_State->abs();
    }
}

template<typename ScalarType>
ScalarType DOTk_PrimalVector<ScalarType>::sum() const
{
    ScalarType result = m_Control->sum();
    if(m_State.use_count() > 0)
    {
        result += m_State->sum();
    }
    return (result);
}

template<typename ScalarType>
ScalarType DOTk_PrimalVector<ScalarType>::dot(const dotk::Vector<ScalarType> & input_) const
{
    ScalarType result = m_Control->dot(*input_.control());
    if(m_State.use_count() > 0)
    {
        result += m_State->dot(*input_.state());
    }
    return (result);
}

template<typename ScalarType>
ScalarType DOTk_PrimalVector<ScalarType>::norm() const
{
    ScalarType result = this->dot(*this);
    result = std::pow(result, 0.5);
    return (result);
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::fill(const ScalarType & value_)
{
    m_Control->fill(value_);
    if(m_State.use_count() > 0)
    {
        m_State->fill(value_);
    }
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::gather(ScalarType* input_) const
{
    if(m_State.use_count() > 0)
    {
        m_Control->gather(input_);
        size_t stride = m_Control->size();
        m_State->gather(input_ + stride);
    }
    else
    {
        m_Control->gather(input_);
    }
}

template<typename ScalarType>
size_t DOTk_PrimalVector<ScalarType>::size() const
{
    return (m_Size);
}

template<typename ScalarType>
std::tr1::shared_ptr<dotk::Vector<ScalarType> > DOTk_PrimalVector<ScalarType>::clone() const
{
    if(m_State.use_count() > 0)
    {
        std::tr1::shared_ptr<dotk::DOTk_PrimalVector<ScalarType> > x(new dotk::DOTk_PrimalVector<ScalarType>(*m_Control, *m_State));
        return (x);
    }
    else
    {
        std::tr1::shared_ptr<dotk::DOTk_PrimalVector<ScalarType> > x(new dotk::DOTk_PrimalVector<ScalarType>(*m_Control));
        return (x);
    }
}

template<typename ScalarType>
const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & DOTk_PrimalVector<ScalarType>::state() const
{
    return (m_State);
}

template<typename ScalarType>
const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & DOTk_PrimalVector<ScalarType>::control() const
{
    return (m_Control);
}

template<typename ScalarType>
ScalarType & DOTk_PrimalVector<ScalarType>::operator [](size_t index_)
{
    assert(index_ >= 0);
    assert(index_ < this->size());
    size_t num_controls = m_Control->size();
    if(index_ < num_controls)
    {
        return (m_Control->operator [](index_));
    }
    else
    {
        size_t index = index_ - num_controls;
        return (m_State->operator [](index));
    }
}

template<typename ScalarType>
const ScalarType & DOTk_PrimalVector<ScalarType>::operator [](size_t index_) const
{
    assert(index_ >= 0);
    assert(index_ < this->size());
    size_t num_controls = m_Control->size();
    if(index_ < num_controls)
    {
        return (m_Control->operator [](index_));
    }
    else
    {
        size_t index = index_ - num_controls;
        return (m_State->operator [](index));
    }
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::initialize(const dotk::DOTk_Primal & primal_)
{
    m_Size = primal_.control()->size();
    m_Control = primal_.control()->clone();
    m_Control->update(1., *primal_.control(), 0.);
    if(primal_.state().use_count() > 0)
    {
        m_Size += primal_.state()->size();
        m_State = primal_.state()->clone();
        m_State->update(1., *primal_.state(), 0.);
    }
}

template<typename ScalarType>
void DOTk_PrimalVector<ScalarType>::initialize(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & state_)
{
    m_State->update(1., state_, 0.);
    m_Control->update(1., control_, 0.);
}

}
