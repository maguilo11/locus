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

template<typename ScalarType>
DOTk_MultiVector<ScalarType>::DOTk_MultiVector(const dotk::DOTk_Primal & primal_) :
        m_Size(0),
        m_Dual(primal_.dual()->clone()),
        m_State(),
        m_Control(primal_.control()->clone())
{
    this->initialize(primal_);
}

template<typename ScalarType>
DOTk_MultiVector<ScalarType>::DOTk_MultiVector(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & dual_) :
        m_Size(dual_.size() + control_.size()),
        m_Dual(dual_.clone()),
        m_State(),
        m_Control(control_.clone())
{
    this->initialize(control_, dual_);
}

template<typename ScalarType>
DOTk_MultiVector<ScalarType>::DOTk_MultiVector(const dotk::Vector<ScalarType> & control_,
                                         const dotk::Vector<ScalarType> & state_,
                                         const dotk::Vector<ScalarType> & dual_) :
        m_Size(dual_.size() + state_.size() + control_.size()),
        m_Dual(dual_.clone()),
        m_State(state_.clone()),
        m_Control(control_.clone())
{
    this->initialize(control_, state_, dual_);
}

template<typename ScalarType>
DOTk_MultiVector<ScalarType>::~DOTk_MultiVector()
{
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::scale(const ScalarType & alpha_)
{
    m_Dual->scale(alpha_);
    m_Control->scale(alpha_);
    if(m_State.use_count() > 0)
    {
        m_State->scale(alpha_);
    }
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::elementWiseMultiplication(const dotk::Vector<ScalarType> & input_)
{
    assert(input_.size() == this->size());
    m_Dual->elementWiseMultiplication(*input_.dual());
    m_Control->elementWiseMultiplication(*input_.control());
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        m_State->elementWiseMultiplication(*input_.state());
    }
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::update(const ScalarType & alpha_,
                                          const dotk::Vector<ScalarType> & input_,
                                          const ScalarType & beta_)
{
    assert(input_.size() == this->size());
    m_Dual->update(alpha_, *input_.dual(), beta_);
    m_Control->update(alpha_, *input_.control(), beta_);
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        m_State->update(alpha_, *input_.state(), beta_);
    }
}

template<typename ScalarType>
ScalarType DOTk_MultiVector<ScalarType>::max() const
{
    ScalarType dual_max = m_Dual->max();
    ScalarType control_max = m_Control->max();
    ScalarType max_value = std::max(dual_max, control_max);
    if(m_State.use_count() > 0)
    {
        ScalarType state_max = m_State->max();
        max_value = std::max(state_max, max_value);
    }
    return (max_value);
}

template<typename ScalarType>
ScalarType DOTk_MultiVector<ScalarType>::min() const
{
    ScalarType dual_min = m_Dual->min();
    ScalarType primal_min = m_Control->min();
    ScalarType min_value = std::min(dual_min, primal_min);
    if(m_State.use_count() > 0)
    {
        ScalarType state_min = m_State->min();
        min_value = std::min(state_min, min_value);
    }
    return (min_value);
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::abs()
{
    m_Dual->abs();
    m_Control->abs();
    if(m_State.use_count() > 0)
    {
        m_State->abs();
    }
}

template<typename ScalarType>
ScalarType DOTk_MultiVector<ScalarType>::sum() const
{
    ScalarType result = m_Dual->sum() + m_Control->sum();
    if(m_State.use_count() > 0)
    {
        result += m_State->sum();
    }
    return (result);
}

template<typename ScalarType>
ScalarType DOTk_MultiVector<ScalarType>::dot(const dotk::Vector<ScalarType> & input_) const
{
    assert(input_.size() == this->size());
    ScalarType result = m_Dual->dot(*input_.dual()) + m_Control->dot(*input_.control());
    if(m_State.use_count() > 0)
    {
        assert(input_.state().use_count() > 0);
        result += m_State->dot(*input_.state());
    }
    return (result);
}

template<typename ScalarType>
ScalarType DOTk_MultiVector<ScalarType>::norm() const
{
    ScalarType result = this->dot(*this);
    result = std::pow(result, 0.5);
    return (result);
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::fill(const ScalarType & value_)
{
    m_Dual->fill(value_);
    m_Control->fill(value_);
    if(m_State.use_count() > 0)
    {
        m_State->fill(value_);
    }
}

template<typename ScalarType>
size_t DOTk_MultiVector<ScalarType>::size() const
{
    return (m_Size);
}

template<typename ScalarType>
std::shared_ptr<dotk::Vector<ScalarType> > DOTk_MultiVector<ScalarType>::clone() const
{
    if(m_State.use_count() > 0)
    {
        std::shared_ptr<dotk::DOTk_MultiVector<ScalarType> > vector =
                std::make_shared<dotk::DOTk_MultiVector<ScalarType>>(*m_Control, *m_State, *m_Dual);
        return (vector);
    }
    else
    {
        std::shared_ptr<dotk::DOTk_MultiVector<ScalarType> > vector =
                std::make_shared<dotk::DOTk_MultiVector<ScalarType>>(*m_Control, *m_Dual);
        return (vector);
    }
}

template<typename ScalarType>
const std::shared_ptr<dotk::Vector<ScalarType> > & DOTk_MultiVector<ScalarType>::dual() const
{
    return (m_Dual);
}

template<typename ScalarType>
const std::shared_ptr<dotk::Vector<ScalarType> > & DOTk_MultiVector<ScalarType>::state() const
{
    return (m_State);
}

template<typename ScalarType>
const std::shared_ptr<dotk::Vector<ScalarType> > & DOTk_MultiVector<ScalarType>::control() const
{
    return (m_Control);
}

template<typename ScalarType>
ScalarType & DOTk_MultiVector<ScalarType>::operator [](size_t index_)
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

template<typename ScalarType>
const ScalarType & DOTk_MultiVector<ScalarType>::operator [](size_t index_) const
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

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::initialize(const dotk::DOTk_Primal & primal_)
{
    m_Dual->update(1., *primal_.dual(), 0.);
    m_Control->update(1., *primal_.control(), 0.);
    m_Size = m_Dual->size() + m_Control->size();
    if(primal_.state().use_count() > 0)
    {
        m_Size += m_State->size();
        m_State = primal_.state()->clone();
        m_State->update(1., *primal_.state(), 0.);
    }
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::initialize(const dotk::Vector<ScalarType> & control_, const dotk::Vector<ScalarType> & dual_)
{
    m_Dual->update(1., dual_, 0.);
    m_Control->update(1., control_, 0.);
}

template<typename ScalarType>
void DOTk_MultiVector<ScalarType>::initialize(const dotk::Vector<ScalarType> & control_,
                                        const dotk::Vector<ScalarType> & state_,
                                        const dotk::Vector<ScalarType> & dual_)
{
    m_Dual->update(1., dual_, 0.);
    m_State->update(1., state_, 0.);
    m_Control->update(1., control_, 0.);
}

}
