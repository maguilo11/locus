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

class DOTk_Primal;
class DOTk_Functor;

template<class Type>
class vector;

class DOTk_ThirdOrderForwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_ThirdOrderForwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_ThirdOrderForwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_);
    virtual ~DOTk_ThirdOrderForwardFiniteDifference();

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::vector<Real> & primal_,
                               const dotk::vector<Real> & direction_,
                               const dotk::vector<Real> & first_derivative_,
                               dotk::vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Gradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OriginalPrimal;

private:
    DOTk_ThirdOrderForwardFiniteDifference(const dotk::DOTk_ThirdOrderForwardFiniteDifference&);
    dotk::DOTk_ThirdOrderForwardFiniteDifference operator=(const dotk::DOTk_ThirdOrderForwardFiniteDifference&);
};

}

#endif /* DOTK_THIRDORDERFORWARDFINITEDIFFERENCE_HPP_ */