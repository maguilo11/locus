/*
 * DOTk_GcmmaTestObjectiveFunction.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GCMMATESTOBJECTIVEFUNCTION_HPP_
#define DOTK_GCMMATESTOBJECTIVEFUNCTION_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_GcmmaTestObjectiveFunction : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_GcmmaTestObjectiveFunction();
    virtual ~DOTk_GcmmaTestObjectiveFunction();

    Real getConstant() const;
    virtual Real value(const dotk::Vector<Real> & primal_);
    virtual void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_);

private:
    Real m_Constant;

private:
    DOTk_GcmmaTestObjectiveFunction(const dotk::DOTk_GcmmaTestObjectiveFunction &);
    dotk::DOTk_GcmmaTestObjectiveFunction & operator=(const dotk::DOTk_GcmmaTestObjectiveFunction &);
};

}

#endif /* DOTK_GCMMATESTOBJECTIVEFUNCTION_HPP_ */
