/*
 * DOTk_LeftPrecCR.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPrecCR.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_LeftPrecConjResDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_LeftPrecCR::DOTk_LeftPrecCR(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CR),
        m_DataMng(aSolverDataMng),
        mConjugateDirection(aSolverDataMng->getSolution()->clone()),
        mLinearOperatorTimesRes(aSolverDataMng->getSolution()->clone())
{
}

DOTk_LeftPrecCR::DOTk_LeftPrecCR(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                 const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CR),
        m_DataMng(std::make_shared<dotk::DOTk_LeftPrecConjResDataMng>(aPrimal, aLinearOperator)),
        mConjugateDirection(),
        mLinearOperatorTimesRes()
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecCR::~DOTk_LeftPrecCR()
{
}

void DOTk_LeftPrecCR::initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                                 const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                                 const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->update(1., *aRhsVector, 0.);
    m_DataMng->getLeftPrec()->apply(aMng, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    mConjugateDirection->update(1., *m_DataMng->getLeftPrecTimesVector(), 0.);
    m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getResidual(), mLinearOperatorTimesRes);
    m_DataMng->getLinearOperator()->apply(aMng, mConjugateDirection, m_DataMng->getMatrixTimesVector());
    m_DataMng->getSolution()->fill(0.);

    Real prec_residual_norm = m_DataMng->getLeftPrecTimesVector()->norm();
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(prec_residual_norm);
    Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCR::pcr(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
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
        m_DataMng->getLeftPrec()->apply(aMng, m_DataMng->getMatrixTimesVector(),
                m_DataMng->getLeftPrecTimesVector());
        Real scaled_curvature = m_DataMng->getMatrixTimesVector()->dot(*m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkCurvature(scaled_curvature) == true)
        {
            break;
        }
        Real old_res_dot_linear_operator_times_res = m_DataMng->getResidual()->dot(*mLinearOperatorTimesRes);
        Real alpha = old_res_dot_linear_operator_times_res / scaled_curvature;
        m_DataMng->getSolution()->update(alpha, *mConjugateDirection, 1.);
        Real norm_solution = m_DataMng->getSolution()->norm();
        if (norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getPreviousSolution()->update(1., *m_DataMng->getSolution(), 0.);
        m_DataMng->getResidual()->update(-alpha, *m_DataMng->getLeftPrecTimesVector(), 1.);
        m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getResidual(), mLinearOperatorTimesRes);
        Real new_res_dot_linear_operator_times_res = m_DataMng->getResidual()->dot(*mLinearOperatorTimesRes);
        Real scaled_residual_norm = std::sqrt(new_res_dot_linear_operator_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(scaled_residual_norm);
        Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(scaled_residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        Real beta = new_res_dot_linear_operator_times_res / old_res_dot_linear_operator_times_res;
        mConjugateDirection->scale(beta);
        mConjugateDirection->update(static_cast<Real>(1.0), *m_DataMng->getResidual(), 1.);
        m_DataMng->getMatrixTimesVector()->scale(beta);
        m_DataMng->getMatrixTimesVector()->update(1., *mLinearOperatorTimesRes, 1.);
        ++itr;
    }
}

void DOTk_LeftPrecCR::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecCR::getDataMng() const
{
    return (m_DataMng);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecCR::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecCR::getDescentDirection()
{
    return (mConjugateDirection);
}

void DOTk_LeftPrecCR::solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                            const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                            const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->pcr(aRhsVector, aCriterion, aMng);
}

void DOTk_LeftPrecCR::initialize(const std::shared_ptr<dotk::Vector<Real> > vec_)
{
    mConjugateDirection = vec_->clone();
    mLinearOperatorTimesRes = vec_->clone();
}

}
