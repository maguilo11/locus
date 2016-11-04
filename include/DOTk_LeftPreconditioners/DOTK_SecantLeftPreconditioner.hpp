/*
 * DOTK_SecantLeftPreconditioner.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_SECANTLEFTPRECONDITIONER_HPP_
#define DOTK_SECANTLEFTPRECONDITIONER_HPP_

#include "DOTk_LeftPreconditioner.hpp"

namespace dotk
{

class DOTk_SecondOrderOperator;
class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTK_SecantLeftPreconditioner: public dotk::DOTk_LeftPreconditioner
{
public:
    DOTK_SecantLeftPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                  dotk::types::invhessian_t type_,
                                  size_t secant_storage_ = 0);
    virtual ~DOTK_SecantLeftPreconditioner();

    dotk::types::invhessian_t getSecantLeftPrecType() const;

    void setSr1Preconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    void setBfgsPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    void setBarzilaiBorweinPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    void setLbfgsPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t secant_storage_);
    void setLdfpPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t secant_storage_);
    void setLsr1Preconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t secant_storage_);

    virtual void setNumOptimizationItrDone(size_t itr_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_);

private:
    void setSecantLeftPrecType(dotk::types::invhessian_t type_);

private:
    dotk::types::invhessian_t m_SecantLeftPrecType;
    std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> m_SecantLeftPrec;

private:
    DOTK_SecantLeftPreconditioner(const dotk::DOTK_SecantLeftPreconditioner &);
    dotk::DOTK_SecantLeftPreconditioner & operator=(const dotk::DOTK_SecantLeftPreconditioner &);
};

}

#endif /* DOTK_SECANTLEFTPRECONDITIONER_HPP_ */
