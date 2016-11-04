/*
 * DOTk_AugmentedSystem.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_AugmentedSystem.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_AugmentedSystem::DOTk_AugmentedSystem() :
        dotk::DOTk_LinearOperator(dotk::types::AUGMENTED_SYSTEM)
{
}

DOTk_AugmentedSystem::~DOTk_AugmentedSystem()
{
}

void DOTk_AugmentedSystem::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    /// Apply left hand side vector to augmented system.  Thus, we have the following matrix vector product for solution: \n
    /// \n
    /// |       I        | grad_x(C(x_k))* | |  X1 |   | X1 + (grad_x(C(x_k))*) X2 | \n
    /// |----------------------------------| |-----| = |---------------------------| \n
    /// | grad_x(C(x_k)) |        0        | |  X2 |   |       grad_x(C(x_k)) X1   | \n
    /// \n
    ///
    mng_->getRoutinesMng()->adjointJacobian(mng_->getNewPrimal(), input_->dual(), output_);
    output_->axpy(static_cast<Real>(1.), *input_);
    mng_->getRoutinesMng()->jacobian(mng_->getNewPrimal(), input_, output_->dual());
}

}
