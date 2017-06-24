/*
 * DOTk_FirstOrderOperatorFactory.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_ForwardDifferenceGrad.hpp"
#include "DOTk_BackwardDifferenceGrad.hpp"
#include "DOTk_CentralDifferenceGrad.hpp"
#include "DOTk_UserDefinedGrad.hpp"
#include "DOTk_ParallelForwardDiffGrad.hpp"
#include "DOTk_ParallelBackwardDiffGrad.hpp"
#include "DOTk_ParallelCentralDiffGrad.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"

namespace dotk
{

DOTk_FirstOrderOperatorFactory::DOTk_FirstOrderOperatorFactory() :
        mFactoryType(dotk::types::GRADIENT_OPERATOR_DISABLED)
{
}

DOTk_FirstOrderOperatorFactory::DOTk_FirstOrderOperatorFactory(dotk::types::gradient_t aType) :
        mFactoryType(aType)
{
}

DOTk_FirstOrderOperatorFactory::~DOTk_FirstOrderOperatorFactory()
{
}

void DOTk_FirstOrderOperatorFactory::setFactoryType(dotk::types::gradient_t aType)
{
    mFactoryType = aType;
}

dotk::types::gradient_t DOTk_FirstOrderOperatorFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_FirstOrderOperatorFactory::buildForwardFiniteDiffGradient
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_ForwardDifferenceGrad>(aVector);
    this->setFactoryType(dotk::types::FORWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildBackwardFiniteDiffGradient
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_BackwardDifferenceGrad>(aVector);
    this->setFactoryType(dotk::types::BACKWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildCentralFiniteDiffGradient
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_CentralDifferenceGrad>(aVector);
    this->setFactoryType(dotk::types::CENTRAL_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildUserDefinedGradient
(std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_UserDefinedGrad>();
    this->setFactoryType(dotk::types::USER_DEFINED_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildParallelForwardFiniteDiffGradient
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_ParallelForwardDiffGrad>(aVector);
    this->setFactoryType(dotk::types::PARALLEL_FORWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildParallelBackwardFiniteDiffGradient
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_ParallelBackwardDiffGrad>(aVector);
    this->setFactoryType(dotk::types::PARALLEL_BACKWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildParallelCentralFiniteDiffGradient
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_ParallelCentralDiffGrad>(aVector);
    this->setFactoryType(dotk::types::PARALLEL_CENTRAL_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::build
(const dotk::DOTk_OptimizationDataMng * const aMng,
 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput)
{
    switch(this->getFactoryType())
    {
        case dotk::types::FORWARD_DIFF_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_ForwardDifferenceGrad>(aMng->getNewGradient());
            break;
        }
        case dotk::types::BACKWARD_DIFF_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_BackwardDifferenceGrad>(aMng->getNewGradient());
            break;
        }
        case dotk::types::CENTRAL_DIFF_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_CentralDifferenceGrad>(aMng->getNewGradient());
            break;
        }
        case dotk::types::USER_DEFINED_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_UserDefinedGrad>();
            break;
        }
        case dotk::types::PARALLEL_FORWARD_DIFF_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_ParallelForwardDiffGrad>(aMng->getNewGradient());
            break;
        }
        case dotk::types::PARALLEL_BACKWARD_DIFF_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_ParallelBackwardDiffGrad>(aMng->getNewGradient());
            break;
        }
        case dotk::types::PARALLEL_CENTRAL_DIFF_GRAD:
        {
            aOutput = std::make_shared<dotk::DOTk_ParallelCentralDiffGrad>(aMng->getNewGradient());
            break;
        }
        case dotk::types::GRADIENT_OPERATOR_DISABLED:
        {
            break;
        }
        default:
        {
            std::cout << "\nDOTk WARNING: Invalid gradient type, Default step set to Central Difference.\n"
                      << std::flush;
            aOutput = std::make_shared<dotk::DOTk_CentralDifferenceGrad>(aMng->getNewGradient());
            break;
        }
    }
}

}
