/*
 * DOTk_NumericalDifferentiaton.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NUMERICALDIFFERENTIATION_HPP_
#define DOTK_NUMERICALDIFFERENTIATION_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Functor;

template<typename Type>
class vector;

class DOTk_NumericalDifferentiation
{
public:
    DOTk_NumericalDifferentiation(dotk::types::numerical_integration_t type_, Real epsilon_ = 1e-6);
    virtual ~DOTk_NumericalDifferentiation();

    Real getEpsilon() const;
    void setEpsilon(Real epsilon_);
    dotk::types::numerical_integration_t type() const;

    virtual void differentiate(const std::tr1::shared_ptr<dotk::DOTk_Functor> & operator_,
                               const dotk::vector<Real> & primal_,
                               const dotk::vector<Real> & direction_,
                               const dotk::vector<Real> & first_derivative_,
                               dotk::vector<Real> & second_derivative_) = 0;

private:
    Real m_Epsilon;
    dotk::types::numerical_integration_t m_Type;
};

}

#endif /* DOTK_NUMERICALDIFFERENTIATION_HPP_ */
