/*
 * DOTk_SecondOrderForwardFiniteDifference.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_SecondOrderForwardFiniteDifference.hpp"

namespace dotk
{

DOTk_SecondOrderForwardFiniteDifference::DOTk_SecondOrderForwardFiniteDifference
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::SECOND_ORDER_FORWARD_FINITE_DIFF),
        m_Gradient(primal_->control()->clone()),
        m_OriginalPrimal(primal_->control()->clone())
{
}

DOTk_SecondOrderForwardFiniteDifference::DOTk_SecondOrderForwardFiniteDifference
(const std::tr1::shared_ptr< dotk::DOTk_Primal> & primal_, Real epsilon_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::SECOND_ORDER_FORWARD_FINITE_DIFF,
                                            epsilon_),
        m_Gradient(primal_->control()->clone()),
        m_OriginalPrimal(primal_->control()->clone())
{
}

DOTk_SecondOrderForwardFiniteDifference::~DOTk_SecondOrderForwardFiniteDifference()
{
}

void DOTk_SecondOrderForwardFiniteDifference::differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                                                            const dotk::Vector<Real> & primal_,
                                                            const dotk::Vector<Real> & direction_,
                                                            const dotk::Vector<Real> & first_derivative_,
                                                            dotk::Vector<Real> & second_derivative_)
{
    m_OriginalPrimal->update(1., primal_, 0.);

    Real epsilon = dotk::DOTk_NumericalDifferentiation::getEpsilon();
    Real scale_factor = static_cast<Real>(2.) * epsilon;
    m_OriginalPrimal->update(scale_factor, direction_, 1.);
    functor_->operator()(*m_OriginalPrimal, second_derivative_);

    m_OriginalPrimal->update(1., primal_, 0.);
    m_OriginalPrimal->update(epsilon, direction_, 1.);
    m_Gradient->fill(0.);
    functor_->operator()(*m_OriginalPrimal, *m_Gradient);
    second_derivative_.update(static_cast<Real>(-4.), *m_Gradient, 1.);
    second_derivative_.update(static_cast<Real>(3.), first_derivative_, 1.);

    scale_factor = static_cast<Real>(-1.) / (static_cast<Real>(2.) * epsilon);
    second_derivative_.scale(scale_factor);
}

}
