/*
 * DOTk_PrecGenMinResDataMng.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_ArnoldiProjection.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_RightPreconditioner.hpp"
#include "DOTk_PrecGenMinResDataMng.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_RightPreconditionerFactory.hpp"
#include "DOTk_OrthogonalProjectionFactory.hpp"
#include "DOTk_SecantLeftPreconditionerFactory.hpp"

namespace dotk
{

DOTk_PrecGenMinResDataMng::DOTk_PrecGenMinResDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & operator_,
                                                     size_t max_num_itr_) :
        dotk::DOTk_KrylovSolverDataMng(primal_, operator_),
        m_LeftPreconditioner(new dotk::DOTk_LeftPreconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_RightPreconditioner(new dotk::DOTk_RightPreconditioner(dotk::types::RIGHT_PRECONDITIONER_DISABLED)),
        m_OrthogonalProjection(new dotk::DOTk_ArnoldiProjection(primal_, max_num_itr_)),
        m_LeftPrecTimesResidual(),
        m_RightPrecTimesResidual()
{
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::PREC_GMRES);
    dotk::DOTk_KrylovSolverDataMng::setMaxNumSolverItr(max_num_itr_);
    this->allocate(primal_);
}

DOTk_PrecGenMinResDataMng::~DOTk_PrecGenMinResDataMng()
{
}

void DOTk_PrecGenMinResDataMng::setArnoldiProjection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    dotk::DOTk_OrthogonalProjectionFactory factory(krylov_subspace_dim, dotk::types::ARNOLDI);
    factory.build(primal_, m_OrthogonalProjection);
}

void DOTk_PrecGenMinResDataMng::setGramSchmidtProjection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    dotk::DOTk_OrthogonalProjectionFactory factory(krylov_subspace_dim, dotk::types::GRAM_SCHMIDT);
    factory.build(primal_, m_OrthogonalProjection);
}

void DOTk_PrecGenMinResDataMng::setLbfgsSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLbfgsSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setLdfpSecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLdfpSecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setLsr1SecantLeftPreconditioner(size_t secant_storage_)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLsr1SecantPreconditioner(secant_storage_, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setSr1SecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildSr1SecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setBfgsSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBfgsSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setBarzilaiBorweinSecantLeftPreconditioner()
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildBarzilaiBorweinSecantPreconditioner(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setRightPreconditioner(dotk::types::right_prec_t type_)
{
    dotk::DOTk_RightPreconditionerFactory factory(type_);
    factory.build(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_RightPreconditioner);
}

const std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> &
DOTk_PrecGenMinResDataMng::getProjection() const
{
    return (m_OrthogonalProjection);
}

const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> &
DOTk_PrecGenMinResDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

const std::tr1::shared_ptr<dotk::DOTk_RightPreconditioner> &
DOTk_PrecGenMinResDataMng::getRightPrec() const
{
    return (m_RightPreconditioner);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_PrecGenMinResDataMng::getLeftPrecTimesVector() const
{
    return (m_LeftPrecTimesResidual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_PrecGenMinResDataMng::getRightPrecTimesVector() const
{
    return (m_RightPrecTimesResidual);
}

void DOTk_PrecGenMinResDataMng::allocate(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    bool is_dual_allocated = primal_->dual().use_count() > 0;
    bool is_state_allocated = primal_->state().use_count() > 0;
    bool is_control_allocated = primal_->control().use_count() > 0;

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == true) )
    {
        m_LeftPrecTimesResidual = primal_->control()->clone();
        m_RightPrecTimesResidual = primal_->control()->clone();
    }
    else
    {
        m_LeftPrecTimesResidual.reset(new dotk::DOTk_MultiVector<Real>(*primal_));
        m_LeftPrecTimesResidual->fill(0);
        m_RightPrecTimesResidual.reset(new dotk::DOTk_MultiVector<Real>(*primal_));
        m_RightPrecTimesResidual->fill(0);
    }

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == false) )
    {
        std::perror("\n**** DOTk ERROR in DOTk_PrecGenMinResDataMng::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
