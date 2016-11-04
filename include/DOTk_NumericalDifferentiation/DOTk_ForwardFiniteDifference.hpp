/*
 * DOTk_ForwardFiniteDifference.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FORWARDFINITEDIFFERENCE_HPP_
#define DOTK_FORWARDFINITEDIFFERENCE_HPP_

#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_Functor;

template<class Type>
class vector;

class DOTk_ForwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_ForwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_ForwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_);
    virtual ~DOTk_ForwardFiniteDifference();

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::vector<Real> & primal_,
                               const dotk::vector<Real> & direction_,
                               const dotk::vector<Real> & first_derivative_,
                               dotk::vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_PrimalOriginal;

private:
    DOTk_ForwardFiniteDifference(const dotk::DOTk_ForwardFiniteDifference&);
    dotk::DOTk_ForwardFiniteDifference operator=(const dotk::DOTk_ForwardFiniteDifference&);
};

}

#endif /* DOTK_FORWARDFINITEDIFFERENCE_HPP_ */
