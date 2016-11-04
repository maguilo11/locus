/*
 * DOTk_LeftPrecGenConjResDataMng.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_RightPreconditioner.hpp"
#include "DOTk_OrthogonalProjection.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_LeftPrecGenConjResDataMng.hpp"
#include "DOTk_SecantLeftPreconditionerFactory.hpp"

namespace dotk
{

DOTk_LeftPrecGenConjResDataMng::DOTk_LeftPrecGenConjResDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                               const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                               size_t max_num_itr_) :
        dotk::DOTk_KrylovSolverDataMng(primal_, linear_operator_),
        m_LeftPreconditioner(new dotk::DOTk_LeftPreconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_LeftPrecTimesResidual()
{
    dotk::DOTk_KrylovSolverDataMng::setMaxNumSolverItr(max_num_itr_);
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::krylov_solver_t::LEFT_PREC_GCR);
    this->allocate(primal_);
}

DOTk_LeftPrecGenConjResDataMng::~DOTk_LeftPrecGenConjResDataMng()
{
}

void DOTk_LeftPrecGenConjResDataMng::setLbfgsSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLbfgsSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecGenConjResDataMng::setLdfpSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLdfpSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecGenConjResDataMng::setLsr1SecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLsr1SecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecGenConjResDataMng::setSr1SecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildSr1SecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecGenConjResDataMng::setBfgsSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBfgsSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecGenConjResDataMng::setBarzilaiBorweinSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBarzilaiBorweinSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & DOTk_LeftPrecGenConjResDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LeftPrecGenConjResDataMng::getLeftPrecTimesVector() const
{
    return (m_LeftPrecTimesResidual);
}

void DOTk_LeftPrecGenConjResDataMng::allocate(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    bool is_dual_allocated = primal_->dual().use_count() > 0;
    bool is_state_allocated = primal_->state().use_count() > 0;
    bool is_control_allocated = primal_->control().use_count() > 0;

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == true) )
    {
        m_LeftPrecTimesResidual = primal_->control()->clone();
    }
    else
    {
        m_LeftPrecTimesResidual.reset(new dotk::DOTk_MultiVector<Real>(*primal_));
        m_LeftPrecTimesResidual->fill(0);
    }

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == false) )
    {
        std::perror("\n**** DOTk ERROR in DOTk_LeftPrecGenConjResDataMng::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
