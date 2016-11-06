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

DOTk_LeftPrecCR::DOTk_LeftPrecCR(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CR),
        m_DataMng(mng_),
        mConjugateDirection(mng_->getSolution()->clone()),
        mLinearOperatorTimesRes(mng_->getSolution()->clone())
{
}

DOTk_LeftPrecCR::DOTk_LeftPrecCR(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CR),
        m_DataMng(new dotk::DOTk_LeftPrecConjResDataMng(primal_, linear_operator_)),
        mConjugateDirection(),
        mLinearOperatorTimesRes()
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecCR::~DOTk_LeftPrecCR()
{
}

void DOTk_LeftPrecCR::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                                 const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                 const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->copy(*rhs_vec_);
    m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    mConjugateDirection->copy(*m_DataMng->getLeftPrecTimesVector());
    m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getResidual(), mLinearOperatorTimesRes);
    m_DataMng->getLinearOperator()->apply(opt_mng_, mConjugateDirection, m_DataMng->getMatrixTimesVector());
    m_DataMng->getSolution()->fill(0.);

    Real prec_residual_norm = m_DataMng->getLeftPrecTimesVector()->norm();
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(prec_residual_norm);
    Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCR::pcr(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                          const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                          const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->initialize(rhs_vec_, criterion_, opt_mng_);
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
        m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getMatrixTimesVector(),
                m_DataMng->getLeftPrecTimesVector());
        Real scaled_curvature = m_DataMng->getMatrixTimesVector()->dot(*m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkCurvature(scaled_curvature) == true)
        {
            break;
        }
        Real old_res_dot_linear_operator_times_res = m_DataMng->getResidual()->dot(*mLinearOperatorTimesRes);
        Real alpha = old_res_dot_linear_operator_times_res / scaled_curvature;
        m_DataMng->getSolution()->axpy(alpha, *mConjugateDirection);
        Real norm_solution = m_DataMng->getSolution()->norm();
        if (norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getPreviousSolution()->copy(*m_DataMng->getSolution());
        m_DataMng->getResidual()->axpy(-alpha, *m_DataMng->getLeftPrecTimesVector());
        m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getResidual(), mLinearOperatorTimesRes);
        Real new_res_dot_linear_operator_times_res = m_DataMng->getResidual()->dot(*mLinearOperatorTimesRes);
        Real scaled_residual_norm = std::sqrt(new_res_dot_linear_operator_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(scaled_residual_norm);
        Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(scaled_residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        Real beta = new_res_dot_linear_operator_times_res / old_res_dot_linear_operator_times_res;
        mConjugateDirection->scale(beta);
        mConjugateDirection->axpy(static_cast<Real>(1.0), *m_DataMng->getResidual());
        m_DataMng->getMatrixTimesVector()->scale(beta);
        m_DataMng->getMatrixTimesVector()->axpy(static_cast<Real>(1.0), *mLinearOperatorTimesRes);
        ++itr;
    }
}

void DOTk_LeftPrecCR::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecCR::getDataMng() const
{
    return (m_DataMng);
}

const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecCR::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LeftPrecCR::getDescentDirection()
{
    return (mConjugateDirection);
}

void DOTk_LeftPrecCR::solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                            const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                            const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->pcr(rhs_vec_, criterion_, opt_mng_);
}

void DOTk_LeftPrecCR::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > vec_)
{
    mConjugateDirection = vec_->clone();
    mLinearOperatorTimesRes = vec_->clone();
}

}
