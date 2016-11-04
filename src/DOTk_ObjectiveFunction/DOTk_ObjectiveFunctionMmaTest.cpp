/*
 * DOTk_ObjectiveFunctionMmaTest.cpp
 *
 *  Created on: Mar 28, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_ObjectiveFunctionMmaTest.hpp"

namespace dotk
{

DOTk_ObjectiveFunctionMmaTest::DOTk_ObjectiveFunctionMmaTest() :
        m_Constant(0.0624)
{
}

DOTk_ObjectiveFunctionMmaTest::~DOTk_ObjectiveFunctionMmaTest()
{
}

Real DOTk_ObjectiveFunctionMmaTest::getConstant() const
{
    return (m_Constant);
}

Real DOTk_ObjectiveFunctionMmaTest::value(const dotk::vector<Real> & primal_)
{
    Real value = this->getConstant() * primal_.sum();
    return (value);
}

void DOTk_ObjectiveFunctionMmaTest::gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & gradient_)
{
    Real constant = this->getConstant();
    gradient_.fill(constant);
}

}
