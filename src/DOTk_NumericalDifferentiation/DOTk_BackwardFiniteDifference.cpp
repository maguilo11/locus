/*
 * DOTk_BackwardFiniteDifference.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */


#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Functor.hpp"
#include "DOTk_BackwardFiniteDifference.hpp"

namespace dotk
{

DOTk_BackwardFiniteDifference::DOTk_BackwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::BACKWARD_FINITE_DIFF),
        m_PrimalOriginal(primal_->control()->clone())
{
}

DOTk_BackwardFiniteDifference::DOTk_BackwardFiniteDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                             Real epsilon_) :
        dotk::DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t::BACKWARD_FINITE_DIFF, epsilon_),
        m_PrimalOriginal(primal_->control()->clone())
{
}

DOTk_BackwardFiniteDifference::~DOTk_BackwardFiniteDifference()
{
}

void DOTk_BackwardFiniteDifference::differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & functor_,
                                                  const dotk::vector<Real> & primal_,
                                                  const dotk::vector<Real> & direction_,
                                                  const dotk::vector<Real> & first_derivative_,
                                                  dotk::vector<Real> & second_derivative_)
{
    Real epsilon = dotk::DOTk_NumericalDifferentiation::getEpsilon();
    m_PrimalOriginal->copy(primal_);
    m_PrimalOriginal->axpy(-epsilon, direction_);

    functor_->operator()(*m_PrimalOriginal, second_derivative_);
    second_derivative_.scale(static_cast<Real>(-1.));
    second_derivative_.axpy(static_cast<Real>(1.), first_derivative_);

    Real scale_factor = static_cast<Real>(1.) / epsilon;
    second_derivative_.scale(scale_factor);
}

}
