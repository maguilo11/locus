/*
 * DOTk_LeftPrecCGNR.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPrecCGNR.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_LeftPrecCGNResDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_LeftPrecCGNR::DOTk_LeftPrecCGNR(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CGNR),
        m_DataMng(aSolverDataMng),
        m_AuxiliaryVector(aSolverDataMng->getSolution()->clone()),
        m_ConjugateDirection(aSolverDataMng->getSolution()->clone()),
        m_ResidualNormalEq(aSolverDataMng->getSolution()->clone())
{
}

DOTk_LeftPrecCGNR::DOTk_LeftPrecCGNR(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                     const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CGNR),
        m_DataMng(std::make_shared<dotk::DOTk_LeftPrecCGNResDataMng>(aPrimal, aLinearOperator)),
        m_AuxiliaryVector(),
        m_ConjugateDirection(),
        m_ResidualNormalEq()
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecCGNR::~DOTk_LeftPrecCGNR()
{
}

void DOTk_LeftPrecCGNR::setMaxNumKrylovSolverItr(size_t aMaxNumIterations)
{
    m_DataMng->setMaxNumSolverItr(aMaxNumIterations);
}

const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecCGNR::getDataMng() const
{
    return (m_DataMng);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecCGNR::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecCGNR::getDescentDirection()
{
    return (m_ConjugateDirection);
}

void DOTk_LeftPrecCGNR::initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                                   const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                                   const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->update(1., *aRhsVector, 0.);
    m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getResidual(), m_ResidualNormalEq);
    m_DataMng->getLeftPrec()->apply(aMng, m_ResidualNormalEq, m_DataMng->getLeftPrecTimesVector());
    m_ConjugateDirection->update(1., *m_DataMng->getLeftPrecTimesVector(), 0.);
    m_DataMng->getSolution()->fill(0.);

    Real normal_eq_res_dot_prec_times_normal_eq_res = m_ResidualNormalEq->dot(*m_DataMng->getLeftPrecTimesVector());
    Real normal_eq_residual_norm = std::sqrt(normal_eq_res_dot_prec_times_normal_eq_res);
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(normal_eq_residual_norm);
    Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCGNR::cgnr(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
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
        Real aux_vec_dot_aux_vec = m_AuxiliaryVector->dot(*m_AuxiliaryVector);
        Real old_normal_eq_res_dot_prec_times_normal_eq_res =
                m_ResidualNormalEq->dot(*m_DataMng->getLeftPrecTimesVector());
        Real alpha = old_normal_eq_res_dot_prec_times_normal_eq_res / aux_vec_dot_aux_vec;
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
        m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getResidual(), m_ResidualNormalEq);
        m_DataMng->getLeftPrec()->apply(aMng, m_ResidualNormalEq, m_DataMng->getLeftPrecTimesVector());
        Real new_normal_eq_res_dot_prec_times_normal_eq_res =
                m_ResidualNormalEq->dot(*m_DataMng->getLeftPrecTimesVector());
        Real normal_eq_residual_norm = std::sqrt(new_normal_eq_res_dot_prec_times_normal_eq_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(normal_eq_residual_norm);
        Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(normal_eq_residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        Real beta = new_normal_eq_res_dot_prec_times_normal_eq_res / old_normal_eq_res_dot_prec_times_normal_eq_res;
        m_ConjugateDirection->scale(beta);
        m_ConjugateDirection->update(1., *m_DataMng->getLeftPrecTimesVector(), 1.);
        ++itr;
    }
}

void DOTk_LeftPrecCGNR::solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                              const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                              const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->cgnr(aRhsVector, aCriterion, aMng);
}

void DOTk_LeftPrecCGNR::initialize(const std::shared_ptr<dotk::Vector<Real> > vector_)
{
    m_AuxiliaryVector = vector_->clone();
    m_ConjugateDirection = vector_->clone();
    m_ResidualNormalEq = vector_->clone();
}

}
