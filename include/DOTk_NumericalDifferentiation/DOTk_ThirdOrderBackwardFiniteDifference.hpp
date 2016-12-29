/*
 * DOTk_ThirdOrderBackwardFiniteDifference.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_THIRDORDERBACKWARDFINITEDIFFERENCE_HPP_
#define DOTK_THIRDORDERBACKWARDFINITEDIFFERENCE_HPP_

#include "DOTk_NumericalDifferentiation.hpp"

namespace dotk
{

class DOTk_Functor;

template<typename ScalarType>
class Vector;

class DOTk_ThirdOrderBackwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_ThirdOrderBackwardFiniteDifference(const dotk::Vector<Real> & input_);
    DOTk_ThirdOrderBackwardFiniteDifference(const dotk::Vector<Real> & input_, Real epsilon_);
    virtual ~DOTk_ThirdOrderBackwardFiniteDifference();

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::Vector<Real> & primal_,
                               const dotk::Vector<Real> & direction_,
                               const dotk::Vector<Real> & first_derivative_,
                               dotk::Vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_Gradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OriginalPrimal;

private:
    DOTk_ThirdOrderBackwardFiniteDifference(const dotk::DOTk_ThirdOrderBackwardFiniteDifference&);
    dotk::DOTk_ThirdOrderBackwardFiniteDifference operator=(const dotk::DOTk_ThirdOrderBackwardFiniteDifference&);
};

}

#endif /* DOTK_THIRDORDERBACKWARDFINITEDIFFERENCE_HPP_ */
