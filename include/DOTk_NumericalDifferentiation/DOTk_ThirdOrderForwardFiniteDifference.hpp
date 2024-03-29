/*
 * DOTk_ThirdOrderForwardFiniteDifference.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_THIRDORDERFORWARDFINITEDIFFERENCE_HPP_
#define DOTK_THIRDORDERFORWARDFINITEDIFFERENCE_HPP_

#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

class DOTk_Functor;

template<typename ScalarType>
class Vector;

class DOTk_ThirdOrderForwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_ThirdOrderForwardFiniteDifference(const dotk::Vector<Real> & input_);
    DOTk_ThirdOrderForwardFiniteDifference(const dotk::Vector<Real> & input_, Real epsilon_);
    virtual ~DOTk_ThirdOrderForwardFiniteDifference();

    virtual void differentiate(const std::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::Vector<Real> & primal_,
                               const dotk::Vector<Real> & direction_,
                               const dotk::Vector<Real> & first_derivative_,
                               dotk::Vector<Real> & second_derivative_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_Gradient;
    std::shared_ptr<dotk::Vector<Real> > m_OriginalPrimal;

private:
    DOTk_ThirdOrderForwardFiniteDifference(const dotk::DOTk_ThirdOrderForwardFiniteDifference&);
    dotk::DOTk_ThirdOrderForwardFiniteDifference operator=(const dotk::DOTk_ThirdOrderForwardFiniteDifference&);
};

}

#endif /* DOTK_THIRDORDERFORWARDFINITEDIFFERENCE_HPP_ */
