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

DOTk_PrecGenMinResDataMng::DOTk_PrecGenMinResDataMng(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                     const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                     size_t aMaxNumIterations ) :
        dotk::DOTk_KrylovSolverDataMng(aPrimal, aLinearOperator),
        m_LeftPreconditioner(std::make_shared<dotk::DOTk_LeftPreconditioner>(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_RightPreconditioner(std::make_shared<dotk::DOTk_RightPreconditioner>(dotk::types::RIGHT_PRECONDITIONER_DISABLED)),
        m_OrthogonalProjection(std::make_shared<dotk::DOTk_ArnoldiProjection>(aPrimal, aMaxNumIterations )),
        m_LeftPrecTimesResidual(),
        m_RightPrecTimesResidual()
{
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::PREC_GMRES);
    dotk::DOTk_KrylovSolverDataMng::setMaxNumSolverItr(aMaxNumIterations );
    this->allocate(aPrimal);
}

DOTk_PrecGenMinResDataMng::~DOTk_PrecGenMinResDataMng()
{
}

void DOTk_PrecGenMinResDataMng::setArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    dotk::DOTk_OrthogonalProjectionFactory factory(krylov_subspace_dim, dotk::types::ARNOLDI);
    factory.build(aPrimal, m_OrthogonalProjection);
}

void DOTk_PrecGenMinResDataMng::setGramSchmidtProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    dotk::DOTk_OrthogonalProjectionFactory factory(krylov_subspace_dim, dotk::types::GRAM_SCHMIDT);
    factory.build(aPrimal, m_OrthogonalProjection);
}

void DOTk_PrecGenMinResDataMng::setLbfgsSecantLeftPreconditioner(size_t aSecantStorageSize)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLbfgsSecantPreconditioner(aSecantStorageSize, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setLdfpSecantLeftPreconditioner(size_t aSecantStorageSize)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLdfpSecantPreconditioner(aSecantStorageSize, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
}

void DOTk_PrecGenMinResDataMng::setLsr1SecantLeftPreconditioner(size_t aSecantStorageSize)
{
    dotk::DOTk_SecantLeftPreconditionerFactory factory;
    factory.buildLsr1SecantPreconditioner(aSecantStorageSize, dotk::DOTk_KrylovSolverDataMng::getSolution(), m_LeftPreconditioner);
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

void DOTk_PrecGenMinResDataMng::setRightPreconditioner(dotk::types::right_prec_t aType)
{
    dotk::DOTk_RightPreconditionerFactory factory(aType);
    factory.build(dotk::DOTk_KrylovSolverDataMng::getSolution(), m_RightPreconditioner);
}

const std::shared_ptr<dotk::DOTk_OrthogonalProjection> &
DOTk_PrecGenMinResDataMng::getProjection() const
{
    return (m_OrthogonalProjection);
}

const std::shared_ptr<dotk::DOTk_LeftPreconditioner> &
DOTk_PrecGenMinResDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

const std::shared_ptr<dotk::DOTk_RightPreconditioner> &
DOTk_PrecGenMinResDataMng::getRightPrec() const
{
    return (m_RightPreconditioner);
}

const std::shared_ptr<dotk::Vector<Real> > &
DOTk_PrecGenMinResDataMng::getLeftPrecTimesVector() const
{
    return (m_LeftPrecTimesResidual);
}

const std::shared_ptr<dotk::Vector<Real> > &
DOTk_PrecGenMinResDataMng::getRightPrecTimesVector() const
{
    return (m_RightPrecTimesResidual);
}

void DOTk_PrecGenMinResDataMng::allocate(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    bool is_dual_allocated = aPrimal->dual().use_count() > 0;
    bool is_state_allocated = aPrimal->state().use_count() > 0;
    bool is_control_allocated = aPrimal->control().use_count() > 0;

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == true) )
    {
        m_LeftPrecTimesResidual = aPrimal->control()->clone();
        m_RightPrecTimesResidual = aPrimal->control()->clone();
    }
    else
    {
        m_LeftPrecTimesResidual = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_LeftPrecTimesResidual->fill(0);
        m_RightPrecTimesResidual = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
        m_RightPrecTimesResidual->fill(0);
    }

    if( (is_dual_allocated == false) && (is_state_allocated == false) && (is_control_allocated == false) )
    {
        std::perror("\n**** DOTk ERROR in DOTk_PrecGenMinResDataMng::initialize. User did not allocate data. ABORT. ****\n");
        std::abort();
    }
}

}
