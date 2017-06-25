/*
 * TRROM_Preconditioner.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Vector.hpp"
#include "TRROM_Preconditioner.hpp"
#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

Preconditioner::Preconditioner(trrom::types::left_prec_t type_) :
        m_Type(type_)
{
}

Preconditioner::~Preconditioner()
{
}

trrom::types::left_prec_t Preconditioner::type() const
{
    return (m_Type);
}
void Preconditioner::applyPreconditioner(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                         const std::shared_ptr<trrom::Vector<double> > & vector_,
                                         std::shared_ptr<trrom::Vector<double> > & prec_times_vector_)
{
    prec_times_vector_->update(1., *vector_, 0.);
}
void Preconditioner::applyInvPreconditioner(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                            const std::shared_ptr<trrom::Vector<double> > & vector_,
                                            std::shared_ptr<trrom::Vector<double> > & inv_prec_times_vector_)
{
    inv_prec_times_vector_->update(1., *vector_, 0.);
}

}
