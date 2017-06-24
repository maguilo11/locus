/*
 * DOTk_LeftPrecCGNE.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPrecCGNE.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_LeftPrecCGNEqDataMng.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_LeftPrecCGNE::DOTk_LeftPrecCGNE(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aMng) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CGNE),
        m_DataMng(aMng),
        m_AuxiliaryVector(aMng->getSolution()->clone()),
        m_ConjugateDirection(aMng->getSolution()->clone()),
        m_ConjugateDirectionNormalEq(aMng->getSolution()->clone())
{
}

DOTk_LeftPrecCGNE::DOTk_LeftPrecCGNE(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                     const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CGNE),
        m_DataMng(std::make_shared<dotk::DOTk_LeftPrecCGNEqDataMng>(aPrimal, aLinearOperator)),
        m_AuxiliaryVector(),
        m_ConjugateDirection(),
        m_ConjugateDirectionNormalEq()
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecCGNE::~DOTk_LeftPrecCGNE()
{
}

void DOTk_LeftPrecCGNE::initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                                   const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                                   const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->update(1., *aRhsVector, 0.);
    m_DataMng->getLeftPrec()->apply(aMng, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getLeftPrecTimesVector(), m_ConjugateDirection);
    m_DataMng->getSolution()->fill(0.);

    Real res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
    Real residual_norm = std::sqrt(res_dot_prec_times_res);
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
    Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCGNE::cgne(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                             const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                             const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->initialize(aRhsVector, aCriterion, aMng);
    if(dotk::DOTk_KrylovSolver::checkCurvature(dotk::DOTk_KrylovSolver::getSolverResidualNorm()) == true)
    {
        return;
    }
    size_t itr = 1;
    while (1)
    {
        if (itr > m_DataMng->getMaxNumSolverItr())
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr);
        m_DataMng->getLinearOperator()->apply(aMng, m_ConjugateDirection, m_AuxiliaryVector);
        Real curvature = m_ConjugateDirection->dot(*m_AuxiliaryVector);
        if (dotk::DOTk_KrylovSolver::checkCurvature(curvature) == true)
        {
            break;
        }
        Real conjugate_dir_dot_conjugate_dir = m_ConjugateDirection->dot(*m_ConjugateDirection);
        Real old_res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
        Real alpha = old_res_dot_prec_times_res / conjugate_dir_dot_conjugate_dir;
        m_DataMng->getPreviousSolution()->update(1., *m_DataMng->getSolution(), 0.);
        m_DataMng->getSolution()->update(alpha, *m_ConjugateDirection, 1.);
        Real norm_solution = m_DataMng->getSolution()->norm();
        if (norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getResidual()->update(-alpha, *m_AuxiliaryVector, 1.);
        m_DataMng->getLeftPrec()->apply(aMng, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
        Real new_res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
        Real residual_norm = std::sqrt(new_res_dot_prec_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
        Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        Real beta = new_res_dot_prec_times_res / old_res_dot_prec_times_res;
        m_ConjugateDirection->scale(beta);
        m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getLeftPrecTimesVector(), m_ConjugateDirectionNormalEq);
        m_ConjugateDirection->update(static_cast<Real>(1.0), *m_ConjugateDirectionNormalEq, 1.);
        ++itr;
    }
}

void DOTk_LeftPrecCGNE::setMaxNumKrylovSolverItr(size_t aMaxNumIterations)
{
    m_DataMng->setMaxNumSolverItr(aMaxNumIterations);
}

const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> &
DOTk_LeftPrecCGNE::getDataMng() const
{
    return (m_DataMng);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> &
DOTk_LeftPrecCGNE::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecCGNE::getDescentDirection()
{
    return (m_ConjugateDirection);
}

void DOTk_LeftPrecCGNE::solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                              const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                              const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->cgne(aRhsVector, aCriterion, aMng);
}

void DOTk_LeftPrecCGNE::initialize(const std::shared_ptr<dotk::Vector<Real> > vector_)
{
    m_AuxiliaryVector = vector_->clone();
    m_ConjugateDirection = vector_->clone();
    m_ConjugateDirectionNormalEq = vector_->clone();
}

}
