/*
 * DOTk_LeftPrecConjResDataMng.cpp
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
#include "DOTk_LeftPrecConjResDataMng.hpp"
#include "DOTk_SecantLeftPreconditionerFactory.hpp"

namespace dotk
{

DOTk_LeftPrecConjResDataMng::DOTk_LeftPrecConjResDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                         const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_KrylovSolverDataMng(primal_, linear_operator_),
        m_LeftPrecTimesResidual(),
        m_LeftPreconditioner(new dotk::DOTk_LeftPreconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED))
{
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::LEFT_PREC_CR);
    this->allocate(primal_);
}

DOTk_LeftPrecConjResDataMng::~DOTk_LeftPrecConjResDataMng()
{
}

void DOTk_LeftPrecConjResDataMng::setLbfgsSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLbfgsSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecConjResDataMng::setLdfpSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLdfpSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecConjResDataMng::setLsr1SecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLsr1SecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecConjResDataMng::setSr1SecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildSr1SecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecConjResDataMng::setBfgsSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBfgsSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecConjResDataMng::setBarzilaiBorweinSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBarzilaiBorweinSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & DOTk_LeftPrecConjResDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecConjResDataMng::getLeftPrecTimesVector() const
{
    return (m_LeftPrecTimesResidual);
}

void DOTk_LeftPrecConjResDataMng::allocate(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
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
        std::perror("\n**** DOTk ERROR in DOTk_LeftPrecConjResDataMng::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
