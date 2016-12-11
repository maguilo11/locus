/*
 * DOTk_UserDefinedHessian.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_UserDefinedHessian.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_UserDefinedHessian::DOTk_UserDefinedHessian() :
        dotk::DOTk_SecondOrderOperator()
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::USER_DEFINED_HESS);
}

DOTk_UserDefinedHessian::~DOTk_UserDefinedHessian()
{
}

void DOTk_UserDefinedHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & hessian_times_vector_)
{
    mng_->getRoutinesMng()->hessian(mng_->getNewPrimal(), vector_, hessian_times_vector_);
}

}
