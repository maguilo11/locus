/*
 * DOTk_PrimalVector.cpp
 *
 *  Created on: Dec 20, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_PrimalVector.hpp"

namespace dotk
{

template<class Type>
DOTk_PrimalVector<Type>::DOTk_PrimalVector(const dotk::DOTk_Primal & primal_) :
        m_Size(0),
        m_State(),
        m_Control()
{
    this->initialize(primal_);
}

template<class Type>
DOTk_PrimalVector<Type>::DOTk_PrimalVector(const dotk::vector<Type> & control_) :
        m_Size(control_.size()),
        m_State(),
        m_Control(control_.clone())
{
    m_Control->copy(control_);
}

template<class Type>
DOTk_PrimalVector<Type>::DOTk_PrimalVector(const dotk::vector<Type> & control_, const dotk::vector<Type> & state_) :
        m_Size(control_.size() + state_.size()),
        m_State(state_.clone()),
        m_Control(control_.clone())
{
    this->initialize(control_, state_);
}

template<class Type>
DOTk_PrimalVector<Type>::~DOTk_PrimalVector()
{
}

template<class Type>
void DOTk_PrimalVector<Type>::scale(const Type & alpha_)
{
    m_Control->scale(alpha_);
    if(m_State.use_count() > 0)
    {
        m_State->scale(alpha_);
    }
}

template<class Type>
void DOTk_PrimalVector<Type>::cwiseProd(const dotk::vector<Type> & input_)
{
    m_Control->cwiseProd(*input_.control());
    if(m_State.use_count() > 0)
    {
        m_State->cwiseProd(*input_.state());
    }
}

template<class Type>
void DOTk_PrimalVector<Type>::axpy(const Type & alpha_, const dotk::vector<Type> & input_)
{
    m_Control->axpy(alpha_, *input_.control());
    if(m_State.use_count() > 0)
    {
        m_State->axpy(alpha_, *input_.state());
    }
}

template<class Type>
Type DOTk_PrimalVector<Type>::max() const
{
    if(m_State.use_count() > 0)
    {
        Type state_max = m_State->max();
        Type control_max = m_Control->max();
        Type max_value = std::max(state_max, control_max);
        return (max_value);
    }
    else
    {
        Type max_value = m_Control->max();
        return (max_value);
    }
}

template<class Type>
Type DOTk_PrimalVector<Type>::min() const
{
    if(m_State.use_count() > 0)
    {
        Type state_min = m_State->min();
        Type control_min = m_Control->min();
        Type min_value = std::min(state_min, control_min);
        return (min_value);
    }
    else
    {
        Type min_value = m_Control->min();
        return (min_value);
    }
}

template<class Type>
void DOTk_PrimalVector<Type>::abs()
{
    m_Control->abs();
    if(m_State.use_count() > 0)
    {
        m_State->abs();
    }
}

template<class Type>
Type DOTk_PrimalVector<Type>::sum() const
{
    Type result = m_Control->sum();
    if(m_State.use_count() > 0)
    {
        result += m_State->sum();
    }
    return (result);
}

template<class Type>
Type DOTk_PrimalVector<Type>::dot(const dotk::vector<Type> & input_) const
{
    Type result = m_Control->dot(*input_.control());
    if(m_State.use_count() > 0)
    {
        result += m_State->dot(*input_.state());
    }
    return (result);
}

template<class Type>
Type DOTk_PrimalVector<Type>::norm() const
{
    Type result = this->dot(*this);
    result = std::pow(result, 0.5);
    return (result);
}

template<class Type>
void DOTk_PrimalVector<Type>::fill(const Type & value_)
{
    m_Control->fill(value_);
    if(m_State.use_count() > 0)
    {
        m_State->fill(value_);
    }
}

template<class Type>
void DOTk_PrimalVector<Type>::copy(const dotk::vector<Type> & input_)
{
    m_Control->copy(*input_.control());
    if(m_State.use_count() > 0)
    {
        m_State->copy(*input_.state());
    }
}

template<class Type>
void DOTk_PrimalVector<Type>::gather(Type* input_) const
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

template<class Type>
size_t DOTk_PrimalVector<Type>::size() const
{
    return (m_Size);
}

template<class Type>
std::tr1::shared_ptr<dotk::vector<Type> > DOTk_PrimalVector<Type>::clone() const
{
    if(m_State.use_count() > 0)
    {
        std::tr1::shared_ptr<dotk::DOTk_PrimalVector<Type> > x(new dotk::DOTk_PrimalVector<Type>(*m_Control, *m_State));
        return (x);
    }
    else
    {
        std::tr1::shared_ptr<dotk::DOTk_PrimalVector<Type> > x(new dotk::DOTk_PrimalVector<Type>(*m_Control));
        return (x);
    }
}

template<class Type>
const std::tr1::shared_ptr<dotk::vector<Type> > & DOTk_PrimalVector<Type>::state() const
{
    return (m_State);
}

template<class Type>
const std::tr1::shared_ptr<dotk::vector<Type> > & DOTk_PrimalVector<Type>::control() const
{
    return (m_Control);
}

template<class Type>
Type & DOTk_PrimalVector<Type>::operator [](size_t index_)
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

template<class Type>
const Type & DOTk_PrimalVector<Type>::operator [](size_t index_) const
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

template<class Type>
dotk::types::container_t DOTk_PrimalVector<Type>::type() const
{
    return (dotk::types::container_t::PRIMAL_VECTOR);
}

template<class Type>
void DOTk_PrimalVector<Type>::initialize(const dotk::DOTk_Primal & primal_)
{
    m_Size = primal_.control()->size();
    m_Control = primal_.control()->clone();
    m_Control->copy(*primal_.control());
    if(primal_.state().use_count() > 0)
    {
        m_Size += primal_.state()->size();
        m_State = primal_.state()->clone();
        m_State->copy(*primal_.state());
    }
}

template<class Type>
void DOTk_PrimalVector<Type>::initialize(const dotk::vector<Type> & control_, const dotk::vector<Type> & state_)
{
    m_State->copy(state_);
    m_Control->copy(control_);
}

template<class Type>
size_t DOTk_PrimalVector<Type>::rank() const
{
    std::perror("\n**** Unimplemented Function DOTk_PrimalVector::rank. ABORT. ****\n");
    std::abort();
    return (0);
}

template<class Type>
const std::tr1::shared_ptr<dotk::vector<Type> > &  DOTk_PrimalVector<Type>::dual() const
{
    std::perror("\n**** Unimplemented Function DOTk_PrimalVector::dual. ABORT. ****\n");
    std::abort();
}

}
