/*
 * DOTk_LeftPreconditioner.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LeftPreconditioner::DOTk_LeftPreconditioner(dotk::types::left_prec_t type_) :
        m_NumOptimizationItrDone(0),
        m_LeftPreconditionerType(type_)
{
}

DOTk_LeftPreconditioner::~DOTk_LeftPreconditioner()
{
}

void DOTk_LeftPreconditioner::setNumOptimizationItrDone(size_t itr_)
{
    m_NumOptimizationItrDone = itr_;
}

size_t DOTk_LeftPreconditioner::getNumOptimizationItrDone() const
{
    return (m_NumOptimizationItrDone);
}

Real DOTk_LeftPreconditioner::getParameter(dotk::types::stopping_criterion_param_t type_) const
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::getParameter. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::setParameter(dotk::types::stopping_criterion_param_t type_, Real parameter_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setParameter. User did not allocate data. ABORT. ****\n");
    std::abort();
}

dotk::types::left_prec_t DOTk_LeftPreconditioner::getLeftPreconditionerType() const
{
    return (m_LeftPreconditionerType);
}

void DOTk_LeftPreconditioner::setLeftPrecCgSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                  size_t max_num_itr_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setLeftPrecCgSolver. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::setLeftPrecCrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                  size_t max_num_itr_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setLeftPrecCrSolver. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::setLeftPrecGcrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                   size_t max_num_itr_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setLeftPrecGcrSolver. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::setLeftPrecCgneSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                    size_t max_num_itr_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setLeftPrecCgneSolver. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::setLeftPrecCgnrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                    size_t max_num_itr_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setLeftPrecCgnrSolver. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::setPrecGmresSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                 size_t max_num_itr_)
{
    std::perror("\n**** DOTk ERROR in DOTk_LeftPreconditioner::setPrecGmresSolver. User did not allocate data. ABORT. ****\n");
    std::abort();
}

void DOTk_LeftPreconditioner::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_)
{
    matrix_times_vec_->copy(*vec_);
}

}
