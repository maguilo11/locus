/*
 * DOTk_UserDefinedHessianTypeCNP.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_UserDefinedHessianTypeCNP.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_UserDefinedHessianTypeCNP::DOTk_UserDefinedHessianTypeCNP() :
        dotk::DOTk_SecondOrderOperator()
{
    dotk::DOTk_SecondOrderOperator::setHessianType(dotk::types::USER_DEFINED_HESS_TYPE_CNP);
}

DOTk_UserDefinedHessianTypeCNP::~DOTk_UserDefinedHessianTypeCNP()
{
}

void DOTk_UserDefinedHessianTypeCNP::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & hessian_times_vector_)
{
    mng_->getRoutinesMng()->hessian(mng_->getNewPrimal(), mng_->getNewDual(), vector_, hessian_times_vector_);
}

}
