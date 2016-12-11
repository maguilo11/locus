/*
 * DOTk_LeftPreconditioner.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECONDITIONER_HPP_
#define DOTK_LEFTPRECONDITIONER_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_LeftPreconditioner
{
public:
    explicit DOTk_LeftPreconditioner(dotk::types::left_prec_t type_);
    virtual ~DOTk_LeftPreconditioner();

    size_t getNumOptimizationItrDone() const;
    virtual void setNumOptimizationItrDone(size_t itr_);
    virtual Real getParameter(dotk::types::stopping_criterion_param_t type_) const;
    virtual void setParameter(dotk::types::stopping_criterion_param_t type_, Real parameter_);

    dotk::types::left_prec_t getLeftPreconditionerType() const;

    virtual void setLeftPrecCgSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    virtual void setLeftPrecCrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    virtual void setLeftPrecGcrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                      size_t max_num_itr_ = 200);
    virtual void setLeftPrecCgneSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);
    virtual void setLeftPrecCgnrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);
    virtual void setPrecGmresSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);

    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    size_t m_NumOptimizationItrDone;
    dotk::types::left_prec_t m_LeftPreconditionerType;

private:
    DOTk_LeftPreconditioner(const dotk::DOTk_LeftPreconditioner &);
    dotk::DOTk_LeftPreconditioner & operator=(const dotk::DOTk_LeftPreconditioner &);
};

}

#endif /* DOTK_LEFTPRECONDITIONER_HPP_ */
