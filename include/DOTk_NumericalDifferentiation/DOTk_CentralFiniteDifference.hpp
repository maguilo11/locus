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

class DOTk_Primal;
class DOTk_Functor;

template<class Type>
class vector;

class DOTk_CentralFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_CentralFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_CentralFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_);
    virtual ~DOTk_CentralFiniteDifference();

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::vector<Real> & primal_,
                               const dotk::vector<Real> & direction_,
                               const dotk::vector<Real> & first_derivative_,
                               dotk::vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Gradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OriginalPrimal;

private:
    DOTk_CentralFiniteDifference(const dotk::DOTk_CentralFiniteDifference&);
    dotk::DOTk_CentralFiniteDifference operator=(const dotk::DOTk_CentralFiniteDifference&);
};

}

#endif /* DOTK_CENTRALFINITEDIFFERENCE_HPP_ */
