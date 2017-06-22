/*
 * DOTk_CentralFiniteDifference.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: miguelaguilo
 */

#ifndef DOTK_CENTRALFINITEDIFFERENCE_HPP_
#define DOTK_CENTRALFINITEDIFFERENCE_HPP_

#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

class DOTk_Functor;

template<typename ScalarType>
class Vector;

class DOTk_CentralFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_CentralFiniteDifference(const dotk::Vector<Real> & primal_);
    DOTk_CentralFiniteDifference(const dotk::Vector<Real> & primal_, Real epsilon_);
    virtual ~DOTk_CentralFiniteDifference();

    virtual void differentiate(const std::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::Vector<Real> & primal_,
                               const dotk::Vector<Real> & direction_,
                               const dotk::Vector<Real> & first_derivative_,
                               dotk::Vector<Real> & second_derivative_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_Gradient;
    std::shared_ptr<dotk::Vector<Real> > m_OriginalPrimal;

private:
    DOTk_CentralFiniteDifference(const dotk::DOTk_CentralFiniteDifference&);
    dotk::DOTk_CentralFiniteDifference operator=(const dotk::DOTk_CentralFiniteDifference&);
};

}

#endif /* DOTK_CENTRALFINITEDIFFERENCE_HPP_ */
