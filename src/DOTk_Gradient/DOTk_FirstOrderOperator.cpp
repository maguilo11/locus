/*
 * DOTk_FirstOrderOperator.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_FirstOrderOperator.hpp"

namespace dotk
{

DOTk_FirstOrderOperator::DOTk_FirstOrderOperator(dotk::types::gradient_t type_) :
        m_GradientType(type_)
{
}

DOTk_FirstOrderOperator::~DOTk_FirstOrderOperator()
{
}

dotk::types::gradient_t DOTk_FirstOrderOperator::type() const
{
    return (m_GradientType);
}

void DOTk_FirstOrderOperator::checkGrad(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_gradient_,
                                        std::tr1::shared_ptr<dotk::Vector<Real> > & new_gradient_)
{
    Real norm_new_grad = new_gradient_->norm();

    if(std::isnan(norm_new_grad))
    {
        new_gradient_->update(1., *old_gradient_, 0.);
        std::cout << "DOTk WARNING: There was a NaN entry in the new gradient operator. \n" << std::flush;
        std::cout << "              The new gradient operator will be set to the old \n" << std::flush;
        std::cout << "              gradient operator.\n" << std::flush;
    }
}

void DOTk_FirstOrderOperator::setFiniteDiffPerturbationVec(const dotk::Vector<Real> & input_)
{
    std::perror("\n**** Unimplemented Function DOTk_FirstOrderOperator::setFiniteDiffPerturbationVec. ABORT. ****\n");
    std::abort();
}

void DOTk_FirstOrderOperator::gradient(const dotk::DOTk_OptimizationDataMng * const mng_)
{
    std::perror("\n**** Unimplemented Function DOTk_FirstOrderOperator::gradient. ABORT. ****\n");
    std::abort();
}

}
