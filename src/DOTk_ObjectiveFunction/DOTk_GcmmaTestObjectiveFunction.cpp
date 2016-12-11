/*
 * DOTk_GcmmaTestObjectiveFunction.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_GcmmaTestObjectiveFunction.hpp"

namespace dotk
{

DOTk_GcmmaTestObjectiveFunction::DOTk_GcmmaTestObjectiveFunction() :
        m_Constant(0.0624)
{
}

DOTk_GcmmaTestObjectiveFunction::~DOTk_GcmmaTestObjectiveFunction()
{
}

Real DOTk_GcmmaTestObjectiveFunction::getConstant() const
{
    return (m_Constant);
}

Real DOTk_GcmmaTestObjectiveFunction::value(const dotk::Vector<Real> & primal_)
{
    Real value = this->getConstant() * primal_.sum();
    return (value);
}

void DOTk_GcmmaTestObjectiveFunction::gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_)
{
    Real constant = this->getConstant();
    output_.fill(constant);
}

}
