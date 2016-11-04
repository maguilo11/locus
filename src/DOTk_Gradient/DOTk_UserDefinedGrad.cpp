/*
 * DOTk_UserDefinedGrad.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_UserDefinedGrad.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_UserDefinedGrad::DOTk_UserDefinedGrad() :
        dotk::DOTk_FirstOrderOperator(dotk::types::USER_DEFINED_GRAD)
{
}

DOTk_UserDefinedGrad::~DOTk_UserDefinedGrad()
{
}

void DOTk_UserDefinedGrad::gradient(const dotk::DOTk_OptimizationDataMng * const mng_)
{
    mng_->getRoutinesMng()->gradient(mng_->getNewPrimal(), mng_->getNewGradient());
}

}
