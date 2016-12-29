/*
 * DOTk_SecondOrderForwardFiniteDifference.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SECONDORDERFORWARDFINITEDIFFERENCE_HPP_
#define DOTK_SECONDORDERFORWARDFINITEDIFFERENCE_HPP_

#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

class DOTk_Functor;

template<typename ScalarType>
class Vector;

class DOTk_SecondOrderForwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_SecondOrderForwardFiniteDifference(const dotk::Vector<Real> & input_);
    DOTk_SecondOrderForwardFiniteDifference(const dotk::Vector<Real> & input_, Real epsilon_);
    virtual ~DOTk_SecondOrderForwardFiniteDifference();

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::Vector<Real> & primal_,
                               const dotk::Vector<Real> & direction_,
                               const dotk::Vector<Real> & first_derivative_,
                               dotk::Vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_Gradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OriginalPrimal;

private:
    DOTk_SecondOrderForwardFiniteDifference(const dotk::DOTk_SecondOrderForwardFiniteDifference&);
    dotk::DOTk_SecondOrderForwardFiniteDifference operator=(const dotk::DOTk_SecondOrderForwardFiniteDifference&);
};

}

#endif /* DOTK_SECONDORDERFORWARDFINITEDIFFERENCE_HPP_ */
