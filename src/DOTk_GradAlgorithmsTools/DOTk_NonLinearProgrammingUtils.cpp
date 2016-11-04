/*
 * DOTk_NonLinearProgrammingUtils.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"

namespace dotk
{

namespace nlp
{

std::tr1::shared_ptr<dotk::vector<Real> > clone(dotk::nlp::variables & variables_,
                                                dotk::types::variable_t codomain_)
{
    switch(codomain_)
    {
        case dotk::types::STATE:
        {
            return (variables_.mState->clone());
            break;
        }
        case dotk::types::CONTROL:
        {
            return(variables_.mControl->clone());
            break;
        }
        case dotk::types::DUAL:
        {
            return(variables_.mDual->clone());
            break;
        }
        case dotk::types::UNDEFINED_VARIABLE:
        {
            return (variables_.mState->clone());
            break;
        }
    }
}

void resetField(const dotk::vector<Real> & data_, dotk::nlp::variables & variables_, dotk::types::derivative_t type_)
{
    switch(type_)
    {
        case dotk::types::U:
        case dotk::types::ZU:
        case dotk::types::UU:
        {
            variables_.mState->copy(data_);
            break;
        }
        case dotk::types::Z:
        case dotk::types::UZ:
        case dotk::types::ZZ:
        {
            variables_.mControl->copy(data_);
            break;
        }
        case dotk::types::ZERO_ORDER_DERIVATIVE:
        {
            break;
        }
    }
}

void perturbField(const Real epsilon_,
                  const dotk::vector<Real> & direction_,
                  dotk::nlp::variables & variables_,
                  dotk::types::derivative_t type_)
{
    switch(type_)
    {
        case dotk::types::U:
        case dotk::types::ZU:
        case dotk::types::UU:
        {
            variables_.mState->axpy(epsilon_, direction_);
            break;
        }
        case dotk::types::Z:
        case dotk::types::UZ:
        case dotk::types::ZZ:
        {
            variables_.mControl->axpy(epsilon_, direction_);
            break;
        }
        case dotk::types::ZERO_ORDER_DERIVATIVE:
        {
            break;
        }
    }
}

}

}
