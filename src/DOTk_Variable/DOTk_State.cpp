/*
 * DOTk_State.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Types.hpp"
#include "DOTk_State.hpp"

namespace dotk
{

DOTk_State::DOTk_State() :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::STATE),
        m_StateBasisSize(0)
{
}

DOTk_State::DOTk_State(const dotk::Vector<Real> & data_) :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::STATE, data_),
        m_StateBasisSize(0)
{
}

DOTk_State::DOTk_State(const dotk::Vector<Real> & data_,
                       const dotk::Vector<Real> & lower_bound_,
                       const dotk::Vector<Real> & upper_bound_) :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::STATE, data_, lower_bound_, upper_bound_),
        m_StateBasisSize(0)
{
}

DOTk_State::~DOTk_State()
{
}

size_t DOTk_State::getStateBasisSize() const
{
    return (m_StateBasisSize);
}

void DOTk_State::setStateBasisSize(size_t size_)
{
    m_StateBasisSize = size_;
}

}
