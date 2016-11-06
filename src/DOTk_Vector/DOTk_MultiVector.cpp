/*
 * DOTk_MultiVector.cpp
 *
 *  Created on: Jul 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>

#include "DOTk_Primal.hpp"
#include "DOTk_MultiVector.hpp"

namespace dotk
{

template<typename Type>
DOTk_MultiVector<Type>::DOTk_MultiVector(const dotk::DOTk_Primal & primal_) :
        m_Size(0),
        m_Dual(primal_.dual()->clone()),
        m_State(),
        m_Control(primal_.control()->clone())
{
    this->initialize(primal_);
}

template<typename Type>
DOTk_MultiVector<Type>::DOTk_MultiVector(const dotk::vector<Type> & control_, const dotk::vector<Type> & dual_) :
        m_Size(dual_.size() + control_.size()),
        m_Dual(dual_.clone()),
        m_State(),
        m_Control(control_.clone())
{
    this->initialize(control_, dual_);
}

template<typename Type>
DOTk_MultiVector<Type>::DOTk_MultiVector(const dotk::vector<Type> & control_,
                                         const dotk::vector<Type> & state_,
                                         const dotk::vector<Type> & dual_) :
        m_Size(dual_.size() + state_.size() + control_.size()),
        m_Dual(dual_.clone()),
        m_State(state_.clone()),
        m_Control(control_.clone())
{
    this->initialize(control_, state_, dual_);
}

template<typename Type>
DOTk_MultiVector<Type>::~DOTk_MultiVector()
{
}

template<typename Type>
void DOTk_MultiVector<Type>::scale(const Type & alpha_)
{
    m_Dual->scale(alpha_);
    m_Control->scale(alpha_);
    if(m_State.use_count() > 0)
    {
        m_State->scale(alpha_);
    }
}

template<typename Type>
void DOTk_MultiVector<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    assert(input_.size() == this->size());
    m_Dual->cwiseProd(*input_.dual());
    m_Control->cwiseProd(*input_.control());
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        m_State->cwiseProd(*input_.state());
    }
}

template<typename Type>
void DOTk_MultiVector<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    assert(input_.size() == this->size());
    m_Dual->axpy(alpha_, *input_.dual());
    m_Control->axpy(alpha_, *input_.control());
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        m_State->cwiseProd(*input_.state());
    }
}

template<typename Type>
Type DOTk_MultiVector<Type>::max() const
{
    Type dual_max = m_Dual->max();
    Type control_max = m_Control->max();
    Type max_value = std::max(dual_max, control_max);
    if(m_State.use_count() > 0)
    {
        Type state_max = m_State->max();
        max_value = std::max(state_max, max_value);
    }
    return (max_value);
}

template<typename Type>
Type DOTk_MultiVector<Type>::min() const
{
    Type dual_min = m_Dual->min();
    Type primal_min = m_Control->min();
    Type min_value = std::min(dual_min, primal_min);
    if(m_State.use_count() > 0)
    {
        Type state_min = m_State->min();
        min_value = std::min(state_min, min_value);
    }
    return (min_value);
}

template<typename Type>
void DOTk_MultiVector<Type>::abs()
{
    m_Dual->abs();
    m_Control->abs();
    if(m_State.use_count() > 0)
    {
        m_State->abs();
    }
}

template<typename Type>
Type DOTk_MultiVector<Type>::sum() const
{
    Type result = m_Dual->sum() + m_Control->sum();
    if(m_State.use_count() > 0)
    {
        result += m_State->sum();
    }
    return (result);
}

template<typename Type>
Type DOTk_MultiVector<Type>::dot(const dotk::vector<Type> & input_) const
{
    assert(input_.size() == this->size());
    Type result = m_Dual->dot(*input_.dual()) + m_Control->dot(*input_.control());
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        result += m_State->dot(*input_.state());
    }
    return (result);
}

template<typename Type>
Type DOTk_MultiVector<Type>::norm() const
{
    Type result = this->dot(*this);
    result = std::pow(result, 0.5);
    return (result);
}

template<typename Type>
void DOTk_MultiVector<Type>::fill(const Type & value_)
{
    m_Dual->fill(value_);
    m_Control->fill(value_);
    if(m_State.use_count() > 0)
    {
        m_State->fill(value_);
    }
}

template<typename Type>
void DOTk_MultiVector<Type>::copy(const dotk::vector<Type> & input_)
{
    assert(input_.size() == this->size());
    m_Dual->copy(*input_.dual());
    m_Control->copy(*input_.control());
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        m_State->copy(*input_.state());
    }
}

template<typename Type>
void DOTk_MultiVector<Type>::gather(Type* input_) const
{
    m_Control->gather(input_);
    size_t stride = m_Control->size();
    if(m_State.use_count() > 0)
    {
        m_State->gather(input_ + stride);
        stride += m_State->size();
    }
    m_Dual->gather(input_ + stride);
}

template<typename Type>
size_t DOTk_MultiVector<Type>::size() const
{
    return (m_Size);
}

template<typename Type>
std::tr1::shared_ptr<dotk::vector<Type> > DOTk_MultiVector<Type>::clone() const
{
    if(m_State.use_count() > 0)
    {
        std::tr1::shared_ptr<dotk::DOTk_MultiVector<Type> >
            vector(new dotk::DOTk_MultiVector<Type>(*m_Control, *m_State, *m_Dual));
        return (vector);
    }
    else
    {
        std::tr1::shared_ptr<dotk::DOTk_MultiVector<Type> >
            vector(new dotk::DOTk_MultiVector<Type>(*m_Control, *m_Dual));
        return (vector);
    }
}

template<typename Type>
const std::tr1::shared_ptr<dotk::vector<Type> > & DOTk_MultiVector<Type>::dual() const
{
    return (m_Dual);
}

template<typename Type>
const std::tr1::shared_ptr<dotk::vector<Type> > & DOTk_MultiVector<Type>::state() const
{
    return (m_State);
}

template<typename Type>
const std::tr1::shared_ptr<dotk::vector<Type> > & DOTk_MultiVector<Type>::control() const
{
    return (m_Control);
}

template<typename Type>
Type & DOTk_MultiVector<Type>::operator [](size_t index_)
{
    assert(index_ >= 0);
    assert(index_ < this->size());

    size_t num_controls = m_Control->size();
    if(index_ < num_controls)
    {
        return (m_Control->operator [](index_));
    }
    else if(m_State.use_count() > 0)
    {
        size_t num_states = m_State->size();
        size_t num_controls_plus_states = num_controls + num_states;
        if(index_ >= num_controls && index_ < num_controls_plus_states)
        {
            size_t index = index_ - num_controls;
            return (m_State->operator [](index));
        }
        else
        {
            size_t index = index_ - num_controls_plus_states;
            return (m_Dual->operator [](index));
        }
    }
    else
    {
        size_t index = index_ - num_controls;
        return (m_Dual->operator [](index));
    }
}

template<typename Type>
const Type & DOTk_MultiVector<Type>::operator [](size_t index_) const
{
    assert(index_ >= 0);
    assert(index_ < this->size());

    size_t num_controls = m_Control->size();
    if(index_ < num_controls)
    {
        return (m_Control->operator [](index_));
    }
    else if(m_State.use_count() > 0)
    {
        size_t num_states = m_State->size();
        size_t num_controls_plus_states = num_controls + num_states;
        if(index_ >= num_controls && index_ < num_controls_plus_states)
        {
            size_t index = index_ - num_controls;
            return (m_State->operator [](index));
        }
        else
        {
            size_t index = index_ - num_controls_plus_states;
            return (m_Dual->operator [](index));
        }
    }
    else
    {
        size_t index = index_ - num_controls;
        return (m_Dual->operator [](index));
    }
}

template<typename Type>
void DOTk_MultiVector<Type>::initialize(const dotk::DOTk_Primal & primal_)
{
    m_Dual->copy(*primal_.dual());
    m_Control->copy(*primal_.control());
    m_Size = m_Dual->size() + m_Control->size();
    if(primal_.state().use_count() > 0)
    {
        m_Size += m_State->size();
        m_State = primal_.state()->clone();
        m_State->copy(*primal_.state());
    }
}

template<typename Type>
void DOTk_MultiVector<Type>::initialize(const dotk::vector<Type> & control_, const dotk::vector<Type> & dual_)
{
    m_Dual->copy(dual_);
    m_Control->copy(control_);
}

template<typename Type>
void DOTk_MultiVector<Type>::initialize(const dotk::vector<Type> & control_,
                                        const dotk::vector<Type> & state_,
                                        const dotk::vector<Type> & dual_)
{
    m_Dual->copy(dual_);
    m_State->copy(state_);
    m_Control->copy(control_);
}

}
