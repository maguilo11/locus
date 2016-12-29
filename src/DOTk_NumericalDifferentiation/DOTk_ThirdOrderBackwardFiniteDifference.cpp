/*
 * DOTk_ThirdOrderBackwardFiniteDifference.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_ThirdOrderBackwardFiniteDifference.hpp"

namespace dotk
{

DOTk_ThirdOrderBackwardFiniteDifference::DOTk_ThirdOrderBackwardFiniteDifference(const dotk::Vector<Real> & input_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::THIRD_ORDER_BACKWARD_FINITE_DIFF),
        m_Gradient(input_.clone()),
        m_OriginalPrimal(input_.clone())
{
}

DOTk_ThirdOrderBackwardFiniteDifference::DOTk_ThirdOrderBackwardFiniteDifference(const dotk::Vector<Real> & input_, Real epsilon_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::THIRD_ORDER_BACKWARD_FINITE_DIFF, epsilon_),
        m_Gradient(input_.clone()),
        m_OriginalPrimal(input_.clone())
{
}

DOTk_ThirdOrderBackwardFiniteDifference::~DOTk_ThirdOrderBackwardFiniteDifference()
{
}

void DOTk_ThirdOrderBackwardFiniteDifference::differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                                                            const dotk::Vector<Real> & primal_,
                                                            const dotk::Vector<Real> & direction_,
                                                            const dotk::Vector<Real> & first_derivative_,
                                                            dotk::Vector<Real> & second_derivative_)
{
    m_OriginalPrimal->update(1., primal_, 0.);

    Real epsilon = dotk::DOTk_NumericalDifferentiation::getEpsilon();
    m_OriginalPrimal->update(-epsilon, direction_, 1.);
    functor_->operator()(*m_OriginalPrimal, second_derivative_);
    second_derivative_.scale(static_cast<Real>(-6.));

    m_Gradient->fill(0.);
    m_OriginalPrimal->update(1., primal_, 0.);
    m_OriginalPrimal->update(epsilon, direction_, 1.);
    functor_->operator()(*m_OriginalPrimal, *m_Gradient);
    second_derivative_.update(static_cast<Real>(2.), *m_Gradient, 1.);

    m_Gradient->fill(0.);
    m_OriginalPrimal->update(1., primal_, 0.);
    Real scale_factor = static_cast<Real>(-2.) * epsilon;
    m_OriginalPrimal->update(scale_factor, direction_, 1.);
    functor_->operator()(*m_OriginalPrimal, *m_Gradient);
    second_derivative_.update(static_cast<Real>(1.), *m_Gradient, 1.);

    second_derivative_.update(static_cast<Real>(3.), first_derivative_, 1.);

    scale_factor = static_cast<Real>(1.) / (static_cast<Real>(6.) * epsilon);
    second_derivative_.scale(scale_factor);
}

}
