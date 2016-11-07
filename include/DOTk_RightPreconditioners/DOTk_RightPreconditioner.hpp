/*
 * DOTk_RightPreconditioner.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_RIGHTPRECONDITIONER_HPP_
#define DOTK_RIGHTPRECONDITIONER_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename Type>
class vector;

class DOTk_RightPreconditioner
{
public:
    explicit DOTk_RightPreconditioner(dotk::types::right_prec_t type_);
    virtual ~DOTk_RightPreconditioner();

    void setRightPreconditionerType(dotk::types::right_prec_t type_);
    dotk::types::right_prec_t getRightPreconditionerType() const;

    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_);

private:
    dotk::types::right_prec_t mRightPreconditionerType;

private:
    DOTk_RightPreconditioner(const dotk::DOTk_RightPreconditioner &);
    dotk::DOTk_RightPreconditioner & operator=(const dotk::DOTk_RightPreconditioner &);
};

}

#endif /* DOTK_RIGHTPRECONDITIONER_HPP_ */
