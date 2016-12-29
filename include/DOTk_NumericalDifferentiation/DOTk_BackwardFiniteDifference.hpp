/*
 * DOTk_BackwardFiniteDifference.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BACKWARDFINITEDIFFERENCE_HPP_
#define DOTK_BACKWARDFINITEDIFFERENCE_HPP_

#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

class DOTk_Functor;

template<typename ScalarType>
class Vector;

class DOTk_BackwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_BackwardFiniteDifference(const dotk::Vector<Real> & primal_);
    DOTk_BackwardFiniteDifference(const dotk::Vector<Real> & primal_, Real epsilon_);
    virtual ~DOTk_BackwardFiniteDifference();

    void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                       const dotk::Vector<Real> & primal_,
                       const dotk::Vector<Real> & direction_,
                       const dotk::Vector<Real> & first_derivative_,
                       dotk::Vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_PrimalOriginal;

private:
    DOTk_BackwardFiniteDifference(const dotk::DOTk_BackwardFiniteDifference&);
    dotk::DOTk_BackwardFiniteDifference operator=(const dotk::DOTk_BackwardFiniteDifference&);
};

}

#endif /* DOTK_BACKWARDFINITEDIFFERENCE_HPP_ */
