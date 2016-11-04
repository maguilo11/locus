/*
 * DOTk_LeftPrecCGNResDataMng.cpp
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
#include "DOTk_LeftPrecCGNResDataMng.hpp"
#include "DOTk_SecantLeftPreconditionerFactory.hpp"

namespace dotk
{

DOTk_LeftPrecCGNResDataMng::DOTk_LeftPrecCGNResDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_KrylovSolverDataMng::DOTk_KrylovSolverDataMng(primal_, linear_operator_),
        m_LeftPreconditioner(new dotk::DOTk_LeftPreconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_LeftPrecTimesResidual()
{
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::LEFT_PREC_CGNR);
    this->allocate(primal_);
}

DOTk_LeftPrecCGNResDataMng::~DOTk_LeftPrecCGNResDataMng()
{
}

void DOTk_LeftPrecCGNResDataMng::setLbfgsSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLbfgsSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecCGNResDataMng::setLdfpSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLdfpSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecCGNResDataMng::setLsr1SecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLsr1SecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecCGNResDataMng::setSr1SecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildSr1SecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecCGNResDataMng::setBfgsSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBfgsSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_LeftPrecCGNResDataMng::setBarzilaiBorweinSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBarzilaiBorweinSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}


const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & DOTk_LeftPrecCGNResDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LeftPrecCGNResDataMng::getLeftPrecTimesVector() const
{
    return (m_LeftPrecTimesResidual);
}

void DOTk_LeftPrecCGNResDataMng::allocate(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
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
        std::perror("\n**** DOTk ERROR in DOTk_LeftPrecCGNResDataMng::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
