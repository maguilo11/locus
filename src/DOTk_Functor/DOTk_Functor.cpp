/*
 * DOTk_Functor.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Types.hpp"
#include "DOTk_Functor.hpp"

namespace dotk
{

DOTk_Functor::DOTk_Functor(dotk::types::functor_t type_) :
        m_FunctorType(type_)
{
}

DOTk_Functor::~DOTk_Functor()
{
}

dotk::types::functor_t DOTk_Functor::getFunctorType() const
{
    return (m_FunctorType);
}

void DOTk_Functor::operator()(const dotk::vector<Real> & control_, dotk::vector<Real> & output_)
{
    return;
}

void DOTk_Functor::operator()(const dotk::vector<Real> & state_,
                              const dotk::vector<Real> & control_,
                              const dotk::vector<Real> & dual_,
                              dotk::vector<Real> & output_)
{
    return;
}

}
