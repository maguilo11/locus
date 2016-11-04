/*
 * DOTk_NumericalDifferentiation.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

DOTk_NumericalDifferentiation::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t type_, Real epsilon_) :
        m_Epsilon(epsilon_),
        m_Type(type_)
{
}

DOTk_NumericalDifferentiation::~DOTk_NumericalDifferentiation()
{
}

Real DOTk_NumericalDifferentiation::getEpsilon() const
{
    return (m_Epsilon);
}

void DOTk_NumericalDifferentiation::setEpsilon(Real epsilon_)
{
    m_Epsilon = epsilon_;
}

dotk::types::numerical_integration_t DOTk_NumericalDifferentiation::type() const
{
    return (m_Type);
}

}
