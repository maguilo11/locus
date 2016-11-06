/*
 * DOTk_ObjectiveFunctionMmaTest.hpp
 *
 *  Created on: Mar 28, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OBJECTIVEFUNCTIONMMATEST_HPP_
#define DOTK_OBJECTIVEFUNCTIONMMATEST_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename Type>
class vector;

class DOTk_ObjectiveFunctionMmaTest: public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_ObjectiveFunctionMmaTest();
    virtual ~DOTk_ObjectiveFunctionMmaTest();

    Real getConstant() const;

    virtual Real value(const dotk::vector<Real> & primal_);
    virtual void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_);

private:
    Real m_Constant;

private:
    DOTk_ObjectiveFunctionMmaTest(const dotk::DOTk_ObjectiveFunctionMmaTest &);
    dotk::DOTk_ObjectiveFunctionMmaTest & operator=(const dotk::DOTk_ObjectiveFunctionMmaTest &);
};

}

#endif /* DOTK_OBJECTIVEFUNCTIONMMATEST_HPP_ */
