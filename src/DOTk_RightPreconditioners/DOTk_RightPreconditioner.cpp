/*
 * DOTk_RightPreconditioner.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Types.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_RightPreconditioner.hpp"

namespace dotk
{

DOTk_RightPreconditioner::DOTk_RightPreconditioner(dotk::types::right_prec_t type_) :
        mRightPreconditionerType(type_)
{
}

DOTk_RightPreconditioner::~DOTk_RightPreconditioner()
{
}

void DOTk_RightPreconditioner::setRightPreconditionerType(dotk::types::right_prec_t type_)
{
    mRightPreconditionerType = type_;
}

dotk::types::right_prec_t DOTk_RightPreconditioner::getRightPreconditionerType() const
{
    return (mRightPreconditionerType);
}

void DOTk_RightPreconditioner::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_,
                                     const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                                     const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_)
{
    matrix_times_vec_->copy(*vec_);
}

}
