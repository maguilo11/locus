/*
 * DOTk_Preconditioner.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PRECONDITIONER_HPP_
#define DOTK_PRECONDITIONER_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_Preconditioner
{
public:
    explicit DOTk_Preconditioner(dotk::types::left_prec_t type_ = dotk::types::LEFT_PRECONDITIONER_DISABLED);
    virtual ~DOTk_Preconditioner();

    dotk::types::left_prec_t type() const;

    virtual void applyPreconditioner(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                     const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                     std::shared_ptr<dotk::Vector<Real> > & prec_times_vector_);
    virtual void applyInvPreconditioner(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                        const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                        std::shared_ptr<dotk::Vector<Real> > & inv_prec_times_vector_);

private:
    dotk::types::left_prec_t m_Type;

private:
    DOTk_Preconditioner(const dotk::DOTk_Preconditioner &);
    dotk::DOTk_Preconditioner & operator=(const dotk::DOTk_Preconditioner & rhs_);
};

}

#endif /* DOTK_PRECONDITIONER_HPP_ */
