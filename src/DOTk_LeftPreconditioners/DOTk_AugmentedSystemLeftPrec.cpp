/*
 * DOTk_AugmentedSystemLeftPrec.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MathUtils.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_AugmentedSystem.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_KrylovSolverFactory.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_AugmentedSystemLeftPrec.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"
#include "DOTk_TangentialSubProblemCriterion.hpp"

namespace dotk
{

DOTk_AugmentedSystemLeftPrec::DOTk_AugmentedSystemLeftPrec(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_LeftPreconditioner(dotk::types::AUGMENTED_SYSTEM_LEFT_PRECONDITIONER),
        m_Solver(),
        m_AugmentedSystem(new dotk::DOTk_AugmentedSystem),
        m_Criterion(new dotk::DOTk_TangentialSubProblemCriterion),
        m_RhsVector()
{
    this->initialize(primal_);
}

DOTk_AugmentedSystemLeftPrec::~DOTk_AugmentedSystemLeftPrec()
{
}

void DOTk_AugmentedSystemLeftPrec::setParameter(dotk::types::stopping_criterion_param_t type_, Real parameter_)
{
    m_Criterion->set(type_, parameter_);
}

Real DOTk_AugmentedSystemLeftPrec::getParameter(dotk::types::stopping_criterion_param_t type_) const
{
    return (m_Criterion->get(type_));
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCgSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgSolver(primal_, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(max_num_itr_);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCrSolver(primal_, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(max_num_itr_);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecGcrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                        size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecGcrSolver(primal_, m_AugmentedSystem, max_num_itr_, m_Solver);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCgneSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                         size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgneSolver(primal_, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(max_num_itr_);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCgnrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                         size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgnrSolver(primal_, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(max_num_itr_);
}

void DOTk_AugmentedSystemLeftPrec::setPrecGmresSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                      size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildPrecGmresSolver(primal_, m_AugmentedSystem, max_num_itr_, m_Solver);
}

void DOTk_AugmentedSystemLeftPrec::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                         const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                         const std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    Real norm_residual = dotk::norm(vector_);
    m_Criterion->set(dotk::types::NORM_RESIDUAL, norm_residual);

    dotk::copy(vector_, m_RhsVector);
    m_Solver->solve(m_RhsVector, m_Criterion, mng_);
    dotk::copy(m_Solver->getDataMng()->getSolution(), output_);
}

void DOTk_AugmentedSystemLeftPrec::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgSolver(primal_, m_AugmentedSystem, m_Solver);
    m_RhsVector.reset(new dotk::DOTk_MultiVector<Real>(*primal_));
    m_RhsVector->fill(0);
}

}
