/*
 * DOTk_NonLinearProgrammingUtils.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>
#include <iostream>

#include "vector.hpp"
#include "DOTk_NonLinearProgrammingUtils.hpp"

namespace dotk
{

namespace nlp
{

std::tr1::shared_ptr<dotk::Vector<Real> > clone(dotk::nlp::variables & variables_,
                                                dotk::types::variable_t codomain_)
{
    std::tr1::shared_ptr<dotk::Vector<Real> > output;
    switch(codomain_)
    {
        case dotk::types::STATE:
        {
            output = variables_.mState->clone();
            break;
        }
        case dotk::types::CONTROL:
        {
            output = variables_.mControl->clone();
            break;
        }
        case dotk::types::DUAL:
        {
            output = variables_.mDual->clone();
            break;
        }
        case dotk::types::UNDEFINED_VARIABLE:
        {
            output = variables_.mState->clone();
            break;
        }
        case dotk::types::PRIMAL:
        default:
        {
            std::string msg(" VARIABLE TYPE WAS NOT DEFINED. **** \n");
            std::cerr << "\n **** ERROR IN: " << __FILE__ << ", LINE: "
                    << __LINE__ << ", MSG: "<< msg.c_str() << std::flush;
            break;
        }
    }
    return (output);
}

void resetField(const dotk::Vector<Real> & data_, dotk::nlp::variables & variables_, dotk::types::derivative_t type_)
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
                  const dotk::Vector<Real> & direction_,
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
