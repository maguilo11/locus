/*
 * DOTk_ProjLeftPrecCgDataMng.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_GramSchmidt.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OrthogonalProjection.hpp"
#include "DOTk_ProjLeftPrecCgDataMng.hpp"
#include "DOTk_AugmentedSystemPrecFactory.hpp"
#include "DOTk_OrthogonalProjectionFactory.hpp"

namespace dotk
{

DOTk_ProjLeftPrecCgDataMng::DOTk_ProjLeftPrecCgDataMng(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                       const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                       size_t aMaxNumIterations ) :
        dotk::DOTk_KrylovSolverDataMng(aPrimal, aLinearOperator),
        m_ProjectionMethod(std::make_shared<dotk::DOTk_GramSchmidt>(aPrimal, aMaxNumIterations )),
        m_LeftPreconditioner(std::make_shared<dotk::DOTk_LeftPreconditioner>(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_Residual(aMaxNumIterations  + 1),
        m_LeftPrecTimesResidual(aMaxNumIterations )
{
    this->initialize(aMaxNumIterations , dotk::DOTk_KrylovSolverDataMng::getResidual());
}

DOTk_ProjLeftPrecCgDataMng::~DOTk_ProjLeftPrecCgDataMng()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_ProjLeftPrecCgDataMng::getResidual(size_t aIndex) const
{
    return (m_Residual[aIndex]);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_ProjLeftPrecCgDataMng::getLeftPrecTimesVector(size_t aIndex) const
{
    return (m_LeftPrecTimesResidual[aIndex]);
}

const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & DOTk_ProjLeftPrecCgDataMng::getProjection() const
{
    return (m_ProjectionMethod);
}

void DOTk_ProjLeftPrecCgDataMng::setGramSchmidtProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t max_num_itr = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    m_ProjectionMethod = std::make_shared<dotk::DOTk_GramSchmidt>(aPrimal, max_num_itr);
}

void DOTk_ProjLeftPrecCgDataMng::setArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    dotk::DOTk_OrthogonalProjectionFactory factory(krylov_subspace_dim, dotk::types::ARNOLDI);
    factory.build(aPrimal, m_ProjectionMethod);
}

const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & DOTk_ProjLeftPrecCgDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setLeftPrec(const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    m_LeftPreconditioner = aPreconditioner;
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithPcgSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithPcgSolver(aPrimal, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithGcrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithGcrSolver(aPrimal, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithCrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithCrSolver(aPrimal, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithCgneSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithCgneSolver(aPrimal, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithCgnrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithCgnrSolver(aPrimal, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithGmresSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithGmresSolver(aPrimal, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::initialize(size_t aMaxNumIterations , const std::shared_ptr<dotk::Vector<Real> > aVector)
{
    dotk::DOTk_KrylovSolverDataMng::setMaxNumSolverItr(aMaxNumIterations );
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::PROJECTED_PREC_CG);

    for(size_t row = 0; row < aMaxNumIterations ; ++row)
    {
        m_Residual[row] = aVector->clone();
        m_LeftPrecTimesResidual[row] = aVector->clone();
    }
    m_Residual[aMaxNumIterations ] = aVector->clone();

}

}
