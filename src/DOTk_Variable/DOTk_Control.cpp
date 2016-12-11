/*
 * DOTk_Control.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Types.hpp"
#include "DOTk_Control.hpp"

namespace dotk
{

DOTk_Control::DOTk_Control() :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::CONTROL),
        m_ControlBasisSize(0)
{
}

DOTk_Control::DOTk_Control(const dotk::Vector<Real> & data_) :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::CONTROL, data_),
        m_ControlBasisSize(0)
{
}

DOTk_Control::DOTk_Control(const dotk::Vector<Real> & data_,
                           const dotk::Vector<Real> & lower_bound_,
                           const dotk::Vector<Real> & upper_bound_) :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::CONTROL, data_, lower_bound_, upper_bound_),
        m_ControlBasisSize(0)
{
}

DOTk_Control::~DOTk_Control()
{
}

size_t DOTk_Control::getControlBasisSize() const
{
    return (m_ControlBasisSize);
}

void DOTk_Control::setControlBasisSize(size_t size_)
{
    m_ControlBasisSize = size_;
}

}
