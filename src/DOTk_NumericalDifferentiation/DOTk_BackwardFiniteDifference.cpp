/*
 * DOTk_BackwardFiniteDifference.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_BackwardFiniteDifference.hpp"

namespace dotk
{

DOTk_BackwardFiniteDifference::DOTk_BackwardFiniteDifference(const dotk::Vector<Real> & primal_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::BACKWARD_FINITE_DIFF),
        m_PrimalOriginal(primal_.clone())
{
}

DOTk_BackwardFiniteDifference::DOTk_BackwardFiniteDifference(const dotk::Vector<Real> & primal_, Real epsilon_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::BACKWARD_FINITE_DIFF, epsilon_),
        m_PrimalOriginal(primal_.clone())
{
}

DOTk_BackwardFiniteDifference::~DOTk_BackwardFiniteDifference()
{
}

void DOTk_BackwardFiniteDifference::differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                                                  const dotk::Vector<Real> & primal_,
                                                  const dotk::Vector<Real> & direction_,
                                                  const dotk::Vector<Real> & first_derivative_,
                                                  dotk::Vector<Real> & second_derivative_)
{
    Real epsilon = dotk::DOTk_NumericalDifferentiation::getEpsilon();
    m_PrimalOriginal->update(1., primal_, 0.);
    m_PrimalOriginal->update(-epsilon, direction_, 1.);

    functor_->operator()(*m_PrimalOriginal, second_derivative_);
    second_derivative_.update(static_cast<Real>(1.), first_derivative_, -1.);

    Real scale_factor = static_cast<Real>(1.) / epsilon;
    second_derivative_.scale(scale_factor);
}

}
