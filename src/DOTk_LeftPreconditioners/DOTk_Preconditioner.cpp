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
void DOTk_Preconditioner::applyPreconditioner(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                              const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                              std::shared_ptr<dotk::Vector<Real> > & prec_times_vector_)
{
    prec_times_vector_->update(1., *vector_, 0.);
}
void DOTk_Preconditioner::applyInvPreconditioner(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                 const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                                 std::shared_ptr<dotk::Vector<Real> > & inv_prec_times_vector_)
{
    inv_prec_times_vector_->update(1., *vector_, 0.);
}

}
