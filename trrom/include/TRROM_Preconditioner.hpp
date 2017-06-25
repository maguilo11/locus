/*
 * TRROM_Preconditioner.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_PRECONDITIONER_HPP_
#define TRROM_PRECONDITIONER_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class Preconditioner
{
public:
    explicit Preconditioner(trrom::types::left_prec_t type_ = trrom::types::LEFT_PRECONDITIONER_DISABLED);
    virtual ~Preconditioner();

    trrom::types::left_prec_t type() const;

    virtual void applyPreconditioner(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                     const std::shared_ptr<trrom::Vector<double> > & vector_,
                                     std::shared_ptr<trrom::Vector<double> > & prec_times_vector_);
    virtual void applyInvPreconditioner(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                        const std::shared_ptr<trrom::Vector<double> > & vector_,
                                        std::shared_ptr<trrom::Vector<double> > & inv_prec_times_vector_);

private:
    trrom::types::left_prec_t m_Type;

private:
    Preconditioner(const trrom::Preconditioner &);
    trrom::Preconditioner & operator=(const trrom::Preconditioner & rhs_);
};

}

#endif /* TRROM_PRECONDITIONER_HPP_ */
