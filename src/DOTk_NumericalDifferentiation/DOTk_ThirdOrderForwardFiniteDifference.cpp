/*
 * DOTk_ThirdOrderForwardFiniteDifference.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_ThirdOrderForwardFiniteDifference.hpp"

namespace dotk
{

DOTk_ThirdOrderForwardFiniteDifference::DOTk_ThirdOrderForwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::THIRD_ORDER_FORWARD_FINITE_DIFF),
        m_Gradient(primal_->control()->clone()),
        m_OriginalPrimal(primal_->control()->clone())
{
}

DOTk_ThirdOrderForwardFiniteDifference::DOTk_ThirdOrderForwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                               Real epsilon_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::THIRD_ORDER_FORWARD_FINITE_DIFF, epsilon_),
        m_Gradient(primal_->control()->clone()),
        m_OriginalPrimal(primal_->control()->clone())
{
}

DOTk_ThirdOrderForwardFiniteDifference::~DOTk_ThirdOrderForwardFiniteDifference()
{
}

void DOTk_ThirdOrderForwardFiniteDifference::differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                                                           const dotk::vector<Real> & primal_,
                                                           const dotk::vector<Real> & direction_,
                                                           const dotk::vector<Real> & first_derivative_,
                                                           dotk::vector<Real> & second_derivative_)
{
    m_OriginalPrimal->copy(primal_);

    Real epsilon = dotk::DOTk_NumericalDifferentiation::getEpsilon();
    m_OriginalPrimal->axpy(epsilon, direction_);
    functor_->operator()(*m_OriginalPrimal, second_derivative_);
    second_derivative_.scale(static_cast<Real>(6.));

    m_Gradient->fill(0.);
    m_OriginalPrimal->copy(primal_);
    m_OriginalPrimal->axpy(-epsilon, direction_);
    functor_->operator()(*m_OriginalPrimal, *m_Gradient);
    second_derivative_.axpy(static_cast<Real>(-2.), *m_Gradient);

    m_Gradient->fill(0.);
    m_OriginalPrimal->copy(primal_);
    Real scale_factor = static_cast<Real>(2.) * epsilon;
    m_OriginalPrimal->axpy(scale_factor, direction_);
    functor_->operator()(*m_OriginalPrimal, *m_Gradient);
    second_derivative_.axpy(static_cast<Real>(-1.), *m_Gradient);

    second_derivative_.axpy(static_cast<Real>(-3.), first_derivative_);

    scale_factor = static_cast<Real>(1.) / (static_cast<Real>(6.) * epsilon);
    second_derivative_.scale(scale_factor);
}

}