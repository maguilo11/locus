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

DOTk_NumericalDifferentiatonFactory::DOTk_NumericalDifferentiatonFactory(dotk::types::numerical_integration_t aType) :
        m_Type(aType)
{
}

DOTk_NumericalDifferentiatonFactory::~DOTk_NumericalDifferentiatonFactory()
{
}

dotk::types::numerical_integration_t DOTk_NumericalDifferentiatonFactory::type() const
{
    return (m_Type);
}

void DOTk_NumericalDifferentiatonFactory::type(dotk::types::numerical_integration_t aType)
{
    m_Type = aType;
}

void DOTk_NumericalDifferentiatonFactory::buildForwardDifferenceHessian
(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    this->type(dotk::types::FORWARD_FINITE_DIFF);
    aOutput = std::make_shared<dotk::DOTk_ForwardFiniteDifference>(aInput);
}

void DOTk_NumericalDifferentiatonFactory::buildBackwardDifferenceHessian
(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    this->type(dotk::types::BACKWARD_FINITE_DIFF);
    aOutput = std::make_shared<dotk::DOTk_BackwardFiniteDifference>(aInput);
}

void DOTk_NumericalDifferentiatonFactory::buildCentralDifferenceHessian
(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    this->type(dotk::types::CENTRAL_FINITE_DIFF);
    aOutput = std::make_shared<dotk::DOTk_CentralFiniteDifference>(aInput);
}

void DOTk_NumericalDifferentiatonFactory::buildSecondOrderForwardDifferenceHessian
(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    this->type(dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF);
    aOutput = std::make_shared<dotk::DOTk_SecondOrderForwardFiniteDifference>(aInput);
}

void DOTk_NumericalDifferentiatonFactory::buildThirdOrderForwardDifferenceHessian
(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    this->type(dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF);
    aOutput = std::make_shared<dotk::DOTk_ThirdOrderForwardFiniteDifference>(aInput);
}

void DOTk_NumericalDifferentiatonFactory::buildThirdOrderBackwardDifferenceHessian
(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    this->type(dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF);
    aOutput = std::make_shared<dotk::DOTk_ThirdOrderBackwardFiniteDifference>(aInput);
}

void DOTk_NumericalDifferentiatonFactory::build(const dotk::Vector<Real> & aInput,
                                                std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput)
{
    switch(this->type())
    {
        case dotk::types::FORWARD_FINITE_DIFF:
        {
            aOutput = std::make_shared<dotk::DOTk_ForwardFiniteDifference>(aInput);
            break;
        }
        case dotk::types::BACKWARD_FINITE_DIFF:
        {
            aOutput = std::make_shared<dotk::DOTk_BackwardFiniteDifference>(aInput);
            break;
        }
        case dotk::types::CENTRAL_FINITE_DIFF:
        {
            aOutput = std::make_shared<dotk::DOTk_CentralFiniteDifference>(aInput);
            break;
        }
        case dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF:
        {
            aOutput = std::make_shared<dotk::DOTk_SecondOrderForwardFiniteDifference>(aInput);
            break;
        }
        case dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF:
        {
            aOutput = std::make_shared<dotk::DOTk_ThirdOrderForwardFiniteDifference>(aInput);
            break;
        }
        case dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF:
        {
            aOutput = std::make_shared<dotk::DOTk_ThirdOrderBackwardFiniteDifference>(aInput);
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
            aOutput = std::make_shared<dotk::DOTk_ForwardFiniteDifference>(aInput);
            break;
        }
    }
}

}
