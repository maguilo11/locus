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

DOTk_ProjLeftPrecCgDataMng::DOTk_ProjLeftPrecCgDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::shared_ptr<dotk::DOTk_LinearOperator> & operator_,
                                                       size_t max_num_itr_) :
        dotk::DOTk_KrylovSolverDataMng(primal_, operator_),
        m_ProjectionMethod(new dotk::DOTk_GramSchmidt(primal_, max_num_itr_)),
        m_LeftPreconditioner(new dotk::DOTk_LeftPreconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_Residual(max_num_itr_ + 1),
        m_LeftPrecTimesResidual(max_num_itr_)
{
    this->initialize(max_num_itr_, dotk::DOTk_KrylovSolverDataMng::getResidual());
}

DOTk_ProjLeftPrecCgDataMng::~DOTk_ProjLeftPrecCgDataMng()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_ProjLeftPrecCgDataMng::getResidual(size_t index_) const
{
    return (m_Residual[index_]);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_ProjLeftPrecCgDataMng::getLeftPrecTimesVector(size_t index_) const
{
    return (m_LeftPrecTimesResidual[index_]);
}

const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & DOTk_ProjLeftPrecCgDataMng::getProjection() const
{
    return (m_ProjectionMethod);
}

void DOTk_ProjLeftPrecCgDataMng::setGramSchmidtProjection(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t max_num_itr = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    m_ProjectionMethod.reset(new dotk::DOTk_GramSchmidt(primal_, max_num_itr));
}

void DOTk_ProjLeftPrecCgDataMng::setArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = dotk::DOTk_KrylovSolverDataMng::getMaxNumSolverItr();
    dotk::DOTk_OrthogonalProjectionFactory factory(krylov_subspace_dim, dotk::types::ARNOLDI);
    factory.build(primal_, m_ProjectionMethod);
}

const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & DOTk_ProjLeftPrecCgDataMng::getLeftPrec() const
{
    return (m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setLeftPrec(const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    m_LeftPreconditioner = prec_;
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithPcgSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithPcgSolver(primal_, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithGcrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithGcrSolver(primal_, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithCrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithCrSolver(primal_, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithCgneSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithCgneSolver(primal_, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithCgnrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithCgnrSolver(primal_, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::setAugmentedSystemPrecWithGmresSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t krylov_subspace_dim = this->getMaxNumSolverItr();
    dotk::DOTk_AugmentedSystemPrecFactory factory(krylov_subspace_dim);
    factory.buildAugmentedSystemPrecWithGmresSolver(primal_, m_LeftPreconditioner);
}

void DOTk_ProjLeftPrecCgDataMng::initialize(size_t max_num_itr_, const std::shared_ptr<dotk::Vector<Real> > vector_)
{
    dotk::DOTk_KrylovSolverDataMng::setMaxNumSolverItr(max_num_itr_);
    dotk::DOTk_KrylovSolverDataMng::setSolverType(dotk::types::PROJECTED_PREC_CG);

    for(size_t row = 0; row < max_num_itr_; ++row)
    {
        m_Residual[row] = vector_->clone();
        m_LeftPrecTimesResidual[row] = vector_->clone();
    }
    m_Residual[max_num_itr_] = vector_->clone();

}

}
