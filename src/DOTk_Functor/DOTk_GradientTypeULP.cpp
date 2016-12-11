/*
 * DOTk_GradientTypeULP.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_GradientTypeULP.hpp"

namespace dotk
{

DOTk_GradientTypeULP::DOTk_GradientTypeULP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_) :
        dotk::DOTk_Functor::DOTk_Functor(dotk::types::GRADIENT_TYPE_ULP),
        m_ObjectiveFunction(operators_)
{
}

DOTk_GradientTypeULP::~DOTk_GradientTypeULP()
{
}

void DOTk_GradientTypeULP::operator()(const dotk::Vector<Real> & control_, dotk::Vector<Real> & gradient_)
{
    m_ObjectiveFunction->gradient(control_, gradient_);
}

}
