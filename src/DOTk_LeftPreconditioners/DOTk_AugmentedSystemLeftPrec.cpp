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

DOTk_AugmentedSystemLeftPrec::DOTk_AugmentedSystemLeftPrec(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal) :
        dotk::DOTk_LeftPreconditioner(dotk::types::AUGMENTED_SYSTEM_LEFT_PRECONDITIONER),
        m_Solver(),
        m_AugmentedSystem(std::make_shared<dotk::DOTk_AugmentedSystem>()),
        m_Criterion(std::make_shared<dotk::DOTk_TangentialSubProblemCriterion>()),
        m_RhsVector()
{
    this->initialize(aPrimal);
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

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                       size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgSolver(aPrimal, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(aMaxNumIterations);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                       size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCrSolver(aPrimal, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(aMaxNumIterations);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecGcrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                        size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecGcrSolver(aPrimal, m_AugmentedSystem, aMaxNumIterations, m_Solver);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCgneSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                         size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgneSolver(aPrimal, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(aMaxNumIterations);
}

void DOTk_AugmentedSystemLeftPrec::setLeftPrecCgnrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                         size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgnrSolver(aPrimal, m_AugmentedSystem, m_Solver);
    m_Solver->getDataMng()->setMaxNumSolverItr(aMaxNumIterations);
}

void DOTk_AugmentedSystemLeftPrec::setPrecGmresSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                      size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildPrecGmresSolver(aPrimal, m_AugmentedSystem, aMaxNumIterations, m_Solver);
}

void DOTk_AugmentedSystemLeftPrec::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                         const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                         const std::shared_ptr<dotk::Vector<Real> > & aOutput)
{
    Real norm_residual = dotk::norm(aVector);
    m_Criterion->set(dotk::types::NORM_RESIDUAL, norm_residual);

    dotk::update(1., aVector, 0., m_RhsVector);
    m_Solver->solve(m_RhsVector, m_Criterion, aMng);
    dotk::update(1., m_Solver->getDataMng()->getSolution(), 0., aOutput);
}

void DOTk_AugmentedSystemLeftPrec::initialize(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgSolver(aPrimal, m_AugmentedSystem, m_Solver);
    m_RhsVector = std::make_shared<dotk::DOTk_MultiVector<Real>>(*aPrimal);
    m_RhsVector->fill(0);
}

}
