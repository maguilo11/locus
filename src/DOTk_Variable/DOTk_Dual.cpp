/*
 * DOTk_Dual.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Types.hpp"
#include "DOTk_Dual.hpp"

namespace dotk
{

DOTk_Dual::DOTk_Dual() :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::DUAL),
        m_DualBasisSize(0)
{
}

DOTk_Dual::DOTk_Dual(const dotk::vector<Real> & data_) :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::DUAL, data_),
        m_DualBasisSize(0)
{
}

DOTk_Dual::DOTk_Dual(const dotk::vector<Real> & data_,
                     const dotk::vector<Real> & lower_bound_,
                     const dotk::vector<Real> & upper_bound_) :
        dotk::DOTk_Variable::DOTk_Variable(dotk::types::DUAL, data_, lower_bound_, upper_bound_),
        m_DualBasisSize(0)
{
}

DOTk_Dual::~DOTk_Dual()
{
}

size_t DOTk_Dual::getDualBasisSize() const
{
    return (m_DualBasisSize);
}

void DOTk_Dual::setDualBasisSize(size_t size_)
{
    m_DualBasisSize = size_;
}

}
