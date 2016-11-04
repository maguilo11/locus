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

class DOTk_Primal;
class DOTk_Functor;

template<class Type>
class vector;

class DOTk_ThirdOrderBackwardFiniteDifference : public dotk::DOTk_NumericalDifferentiation
{
public:
    explicit DOTk_ThirdOrderBackwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_ThirdOrderBackwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_);
    virtual ~DOTk_ThirdOrderBackwardFiniteDifference();

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                               const dotk::vector<Real> & primal_,
                               const dotk::vector<Real> & direction_,
                               const dotk::vector<Real> & first_derivative_,
                               dotk::vector<Real> & second_derivative_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Gradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OriginalPrimal;

private:
    DOTk_ThirdOrderBackwardFiniteDifference(const dotk::DOTk_ThirdOrderBackwardFiniteDifference&);
    dotk::DOTk_ThirdOrderBackwardFiniteDifference operator=(const dotk::DOTk_ThirdOrderBackwardFiniteDifference&);
};

}

#endif /* DOTK_THIRDORDERBACKWARDFINITEDIFFERENCE_HPP_ */
