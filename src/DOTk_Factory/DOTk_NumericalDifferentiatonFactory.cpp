/*
 * DOTk_NumericalDifferentiatonFactory.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>

#include "vector.hpp"
#include "DOTk_CentralFiniteDifference.hpp"
#include "DOTk_ForwardFiniteDifference.hpp"
#include "DOTk_BackwardFiniteDifference.hpp"
#include "DOTk_NumericalDifferentiatonFactory.hpp"
#include "DOTk_SecondOrderForwardFiniteDifference.hpp"
#include "DOTk_ThirdOrderBackwardFiniteDifference.hpp"
#include "DOTk_ThirdOrderForwardFiniteDifference.hpp"

namespace dotk
{

DOTk_NumericalDifferentiatonFactory::DOTk_NumericalDifferentiatonFactory() :
        m_Type(dotk::types::NUM_INTG_DISABLED)
{
}

DOTk_NumericalDifferentiatonFactory::DOTk_NumericalDifferentiatonFactory(dotk::types::numerical_integration_t type_) :
        m_Type(type_)
{
}

DOTk_NumericalDifferentiatonFactory::~DOTk_NumericalDifferentiatonFactory()
{
}

dotk::types::numerical_integration_t DOTk_NumericalDifferentiatonFactory::type() const
{
    return (m_Type);
}

void DOTk_NumericalDifferentiatonFactory::type(dotk::types::numerical_integration_t type_)
{
    m_Type = type_;
}

void DOTk_NumericalDifferentiatonFactory::buildForwardDifferenceHessian
(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    this->type(dotk::types::FORWARD_FINITE_DIFF);
    output_.reset(new dotk::DOTk_ForwardFiniteDifference(input_));
}

void DOTk_NumericalDifferentiatonFactory::buildBackwardDifferenceHessian
(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    this->type(dotk::types::BACKWARD_FINITE_DIFF);
    output_.reset(new dotk::DOTk_BackwardFiniteDifference(input_));
}

void DOTk_NumericalDifferentiatonFactory::buildCentralDifferenceHessian
(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    this->type(dotk::types::CENTRAL_FINITE_DIFF);
    output_.reset(new dotk::DOTk_CentralFiniteDifference(input_));
}

void DOTk_NumericalDifferentiatonFactory::buildSecondOrderForwardDifferenceHessian
(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    this->type(dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF);
    output_.reset(new dotk::DOTk_SecondOrderForwardFiniteDifference(input_));
}

void DOTk_NumericalDifferentiatonFactory::buildThirdOrderForwardDifferenceHessian
(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    this->type(dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF);
    output_.reset(new dotk::DOTk_ThirdOrderForwardFiniteDifference(input_));
}

void DOTk_NumericalDifferentiatonFactory::buildThirdOrderBackwardDifferenceHessian
(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    this->type(dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF);
    output_.reset(new dotk::DOTk_ThirdOrderBackwardFiniteDifference(input_));
}

void DOTk_NumericalDifferentiatonFactory::build(const dotk::Vector<Real> & input_,
                                                std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_)
{
    switch(this->type())
    {
        case dotk::types::FORWARD_FINITE_DIFF:
        {
            output_.reset(new dotk::DOTk_ForwardFiniteDifference(input_));
            break;
        }
        case dotk::types::BACKWARD_FINITE_DIFF:
        {
            output_.reset(new dotk::DOTk_BackwardFiniteDifference(input_));
            break;
        }
        case dotk::types::CENTRAL_FINITE_DIFF:
        {
            output_.reset(new dotk::DOTk_CentralFiniteDifference(input_));
            break;
        }
        case dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF:
        {
            output_.reset(new dotk::DOTk_SecondOrderForwardFiniteDifference(input_));
            break;
        }
        case dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF:
        {
            output_.reset(new dotk::DOTk_ThirdOrderForwardFiniteDifference(input_));
            break;
        }
        case dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF:
        {
            output_.reset(new dotk::DOTk_ThirdOrderBackwardFiniteDifference(input_));
            break;
        }
        case dotk::types::NUM_INTG_DISABLED:
        {
            break;
        }
        default:
        {
            std::cout << "\nDOTk WARNING: Invalid numerical integration scheme. Default numerical integration "
                    << " will be set to FORWARD FINITE DIFFERENCE.\n" << std::flush;
            output_.reset(new dotk::DOTk_ForwardFiniteDifference(input_));
            break;
        }
    }
}

}
