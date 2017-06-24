/*
 * DOTk_LeftPrecCG.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPrecCG.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_LeftPrecConjGradDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_LeftPrecCG::DOTk_LeftPrecCG(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CG),
        m_SolverDataMng(aSolverDataMng),
        m_ConjugateDirection(aSolverDataMng->getSolution()->clone())
{
}

DOTk_LeftPrecCG::DOTk_LeftPrecCG(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                 const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CG),
        m_SolverDataMng(std::make_shared<dotk::DOTk_LeftPrecConjGradDataMng>(aPrimal, aLinearOperator)),
        m_ConjugateDirection()
{
    m_ConjugateDirection = m_SolverDataMng->getSolution()->clone();
}

DOTk_LeftPrecCG::~DOTk_LeftPrecCG()
{
}

void DOTk_LeftPrecCG::setMaxNumKrylovSolverItr(size_t aMaxNumIterations)
{
    m_SolverDataMng->setMaxNumSolverItr(aMaxNumIterations);
}

const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecCG::getDataMng() const
{
    return (m_SolverDataMng);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecCG::getLinearOperator() const
{
    return (m_SolverDataMng->getLinearOperator());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecCG::getDescentDirection()
{
    return (m_ConjugateDirection);
}

void DOTk_LeftPrecCG::initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                                 const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                                 const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_SolverDataMng->getResidual()->update(1., *aRhsVector, 0.);
    m_SolverDataMng->getLeftPrec()->apply(aMng, m_SolverDataMng->getResidual(), m_SolverDataMng->getLeftPrecTimesVector());
    m_ConjugateDirection->update(1., *m_SolverDataMng->getLeftPrecTimesVector(), 0.);
    m_SolverDataMng->getSolution()->fill(0.);

    Real res_dot_prec_times_res = m_SolverDataMng->getResidual()->dot(*m_SolverDataMng->getLeftPrecTimesVector());
    Real residual_norm = std::sqrt(res_dot_prec_times_res);
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
    Real stopping_tolerance = aCriterion->evaluate(this, m_SolverDataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCG::pcg(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
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
        if (itr > m_SolverDataMng->getMaxNumSolverItr())
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr);
        m_SolverDataMng->getLinearOperator()->apply(aMng, m_ConjugateDirection, m_SolverDataMng->getMatrixTimesVector());
        Real curvature = m_ConjugateDirection->dot(*m_SolverDataMng->getMatrixTimesVector());
        if (dotk::DOTk_KrylovSolver::checkCurvature(curvature) == true)
        {
            break;
        }
        m_SolverDataMng->getPreviousSolution()->update(1., *m_SolverDataMng->getSolution(), 0.);
        Real old_res_dot_prec_times_res = m_SolverDataMng->getResidual()->dot(*m_SolverDataMng->getLeftPrecTimesVector());
        Real alpha = old_res_dot_prec_times_res / curvature;
        m_SolverDataMng->getSolution()->update(alpha, *m_ConjugateDirection, 1.);
        Real norm_solution = m_SolverDataMng->getSolution()->norm();
        Real trust_region_radius = dotk::DOTk_KrylovSolver::getTrustRegionRadius();
        if (norm_solution >= trust_region_radius)
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_SolverDataMng->getResidual()->update(-alpha, *m_SolverDataMng->getMatrixTimesVector(), 1.);
        m_SolverDataMng->getLeftPrec()->apply(aMng, m_SolverDataMng->getResidual(), m_SolverDataMng->getLeftPrecTimesVector());
        Real new_res_dot_prec_times_res = m_SolverDataMng->getResidual()->dot(*m_SolverDataMng->getLeftPrecTimesVector());
        Real residual_norm = std::sqrt(new_res_dot_prec_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
        Real stopping_tolerance = aCriterion->evaluate(this, m_SolverDataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        Real beta = new_res_dot_prec_times_res / old_res_dot_prec_times_res;
        m_ConjugateDirection->scale(beta);
        m_ConjugateDirection->update(1., *m_SolverDataMng->getLeftPrecTimesVector(), 1.);
        ++itr;
    }
}

void DOTk_LeftPrecCG::solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                            const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                            const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->pcg(aRhsVector, aCriterion, aMng);
}

}
