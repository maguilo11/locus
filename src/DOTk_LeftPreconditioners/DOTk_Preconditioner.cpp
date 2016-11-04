/*
 * DOTk_Preconditioner.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Preconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_Preconditioner::DOTk_Preconditioner(dotk::types::left_prec_t type_) :
        m_Type(type_)
{
}

DOTk_Preconditioner::~DOTk_Preconditioner()
{
}

dotk::types::left_prec_t DOTk_Preconditioner::type() const
{
    return (m_Type);
}
void DOTk_Preconditioner::applyPreconditioner(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                              const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                              std::tr1::shared_ptr<dotk::vector<Real> > & prec_times_vector_)
{
    prec_times_vector_->copy(*vector_);
}
void DOTk_Preconditioner::applyInvPreconditioner(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                 std::tr1::shared_ptr<dotk::vector<Real> > & inv_prec_times_vector_)
{
    inv_prec_times_vector_->copy(*vector_);
}

}
