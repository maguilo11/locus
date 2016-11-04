/*
 * DOTk_FirstOrderOperatorFactory.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Types.hpp"
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

DOTk_FirstOrderOperatorFactory::DOTk_FirstOrderOperatorFactory(dotk::types::gradient_t type_) :
        mFactoryType(type_)
{
}

DOTk_FirstOrderOperatorFactory::~DOTk_FirstOrderOperatorFactory()
{
}

void DOTk_FirstOrderOperatorFactory::setFactoryType(dotk::types::gradient_t type_)
{
    mFactoryType = type_;
}

dotk::types::gradient_t DOTk_FirstOrderOperatorFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_FirstOrderOperatorFactory::buildForwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_ForwardDifferenceGrad(vector_));
    this->setFactoryType(dotk::types::FORWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildBackwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_BackwardDifferenceGrad(vector_));
    this->setFactoryType(dotk::types::BACKWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildCentralFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_CentralDifferenceGrad(vector_));
    this->setFactoryType(dotk::types::CENTRAL_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildUserDefinedGradient
(std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_UserDefinedGrad);
    this->setFactoryType(dotk::types::USER_DEFINED_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildParallelForwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_ParallelForwardDiffGrad(vector_));
    this->setFactoryType(dotk::types::PARALLEL_FORWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildParallelBackwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_ParallelBackwardDiffGrad(vector_));
    this->setFactoryType(dotk::types::PARALLEL_BACKWARD_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::buildParallelCentralFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    grad_.reset(new dotk::DOTk_ParallelCentralDiffGrad(vector_));
    this->setFactoryType(dotk::types::PARALLEL_CENTRAL_DIFF_GRAD);
}

void DOTk_FirstOrderOperatorFactory::build
(const dotk::DOTk_OptimizationDataMng * const mng_,
 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_)
{
    switch(this->getFactoryType())
    {
        case dotk::types::FORWARD_DIFF_GRAD:
        {
            grad_.reset(new dotk::DOTk_ForwardDifferenceGrad(mng_->getNewGradient()));
            break;
        }
        case dotk::types::BACKWARD_DIFF_GRAD:
        {
            grad_.reset(new dotk::DOTk_BackwardDifferenceGrad(mng_->getNewGradient()));
            break;
        }
        case dotk::types::CENTRAL_DIFF_GRAD:
        {
            grad_.reset(new dotk::DOTk_CentralDifferenceGrad(mng_->getNewGradient()));
            break;
        }
        case dotk::types::USER_DEFINED_GRAD:
        {
            grad_.reset(new dotk::DOTk_UserDefinedGrad);
            break;
        }
        case dotk::types::PARALLEL_FORWARD_DIFF_GRAD:
        {
            grad_.reset(new dotk::DOTk_ParallelForwardDiffGrad(mng_->getNewGradient()));
            break;
        }
        case dotk::types::PARALLEL_BACKWARD_DIFF_GRAD:
        {
            grad_.reset(new dotk::DOTk_ParallelBackwardDiffGrad(mng_->getNewGradient()));
            break;
        }
        case dotk::types::PARALLEL_CENTRAL_DIFF_GRAD:
        {
            grad_.reset(new dotk::DOTk_ParallelCentralDiffGrad(mng_->getNewGradient()));
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
            grad_.reset(new dotk::DOTk_CentralDifferenceGrad(mng_->getNewGradient()));
            break;
        }
    }
}

}
