/*
 * DOTk_LinearOperator.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>
#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_LinearOperator.hpp"

namespace dotk
{

DOTk_LinearOperator::DOTk_LinearOperator(dotk::types::linear_operator_t type_) :
        m_LinearOperatorType(type_)
{
}

DOTk_LinearOperator::~DOTk_LinearOperator()
{
}

dotk::types::linear_operator_t DOTk_LinearOperator::type() const
{
    return (m_LinearOperatorType);
}

void DOTk_LinearOperator::setNumOtimizationItrDone(size_t itr_)
{
    std::perror("\n**** Calling unimplemented function DOTk_LinearOperator::setNumOtimizationItrDone. ABORT. ****\n");
    std::abort();
}

void DOTk_LinearOperator::updateLimitedMemoryStorage(bool update_)
{
    std::perror("\n**** Calling unimplemented function DOTk_LinearOperator::updateLimitedMemoryStorage. ABORT. ****\n");
    std::abort();
}

void DOTk_LinearOperator::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    output_->copy(*vector_);
}

}
