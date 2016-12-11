/*
 * DOTk_CentralFiniteDifference.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_CentralFiniteDifference.hpp"

namespace dotk
{

DOTk_CentralFiniteDifference::DOTk_CentralFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::CENTRAL_FINITE_DIFF),
        m_Gradient(primal_->control()->clone()),
        m_OriginalPrimal(primal_->control()->clone())
{
}

DOTk_CentralFiniteDifference::DOTk_CentralFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                           Real epsilon_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::CENTRAL_FINITE_DIFF, epsilon_),
        m_Gradient(primal_->control()->clone()),
        m_OriginalPrimal(primal_->control()->clone())
{
}

DOTk_CentralFiniteDifference::~DOTk_CentralFiniteDifference()
{
}

void DOTk_CentralFiniteDifference::differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                                                 const dotk::Vector<Real> & primal_,
                                                 const dotk::Vector<Real> & direction_,
                                                 const dotk::Vector<Real> & first_derivative_,
                                                 dotk::Vector<Real> & second_derivative_)
{
    m_OriginalPrimal->update(1., primal_, 0.);

    Real epsilon = dotk::DOTk_NumericalDifferentiation::getEpsilon();
    m_OriginalPrimal->update(epsilon, direction_, 1.);
    functor_->operator()(*m_OriginalPrimal, second_derivative_);

    m_OriginalPrimal->update(1., primal_, 0.);
    m_OriginalPrimal->update(-epsilon, direction_, 1.);
    m_Gradient->fill(0.);
    functor_->operator()(*m_OriginalPrimal, *m_Gradient);

    second_derivative_.update(-1., *m_Gradient, 1.);

    Real scale_factor = 1. / (2. * epsilon);
    second_derivative_.scale(scale_factor);
}

}
