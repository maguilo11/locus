/*
 * DOTk_PrecGMRES.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_PrecGMRES.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_RightPreconditioner.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_OrthogonalProjection.hpp"
#include "DOTk_PrecGenMinResDataMng.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_PrecGMRES::DOTk_PrecGMRES(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_) :
        dotk::DOTk_KrylovSolver(dotk::types::PREC_GMRES),
        m_DataMng(mng_),
        m_ProjectionOperatorTimesVec(mng_->getSolution()->clone())
{
    this->allocate(m_DataMng->getSolution());
}

DOTk_PrecGMRES::DOTk_PrecGMRES(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                   size_t max_num_itr_) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_GCR),
        m_DataMng(new dotk::DOTk_PrecGenMinResDataMng(primal_, linear_operator_, max_num_itr_)),
        m_ProjectionOperatorTimesVec()
{
    this->allocate(m_DataMng->getSolution());
}

DOTk_PrecGMRES::~DOTk_PrecGMRES()
{
}

void DOTk_PrecGMRES::initialize(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                                const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_)
{
    m_DataMng->getProjection()->clear();
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getSolution()->fill(0.);
    m_DataMng->getResidual()->update(1., *rhs_vec_, 0.);
    m_DataMng->getLeftPrec()->apply(opt_prob_mng_,
                                     m_DataMng->getResidual(),
                                     m_DataMng->getLeftPrecTimesVector());

    Real res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
    Real initial_residual_norm = std::sqrt(res_dot_prec_times_res);
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(initial_residual_norm);

    m_DataMng->getProjection()->setInitialResidual(initial_residual_norm);
    Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getSolution());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_PrecGMRES::gmres(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                           const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                           const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->initialize(rhs_vec_, criterion_, opt_mng_);
    if(dotk::DOTk_KrylovSolver::checkCurvature(dotk::DOTk_KrylovSolver::getSolverResidualNorm()) == true)
    {
        return;
    }
    size_t itr = 0;
    Real scale_factor = static_cast<Real>(1.) / dotk::DOTk_KrylovSolver::getSolverResidualNorm();
    m_DataMng->getResidual()->scale(scale_factor);
    m_DataMng->getProjection()->setOrthogonalVector(itr, m_DataMng->getResidual());
    while (1)
    {
        if (itr == m_DataMng->getMaxNumSolverItr())
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr + 1);
        m_DataMng->getRightPrec()->apply(opt_mng_,
                                         m_DataMng->getProjection()->getOrthogonalVector(itr),
                                         m_DataMng->getRightPrecTimesVector());

        m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getRightPrecTimesVector(), m_DataMng->getMatrixTimesVector());

        m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getMatrixTimesVector(), m_DataMng->getLeftPrecTimesVector());
        m_ProjectionOperatorTimesVec->update(1., *m_DataMng->getLeftPrecTimesVector(), 0.);
        m_DataMng->getProjection()->apply(this, m_ProjectionOperatorTimesVec);

        m_DataMng->getPreviousSolution()->update(1., *m_DataMng->getSolution(), 0.);
        m_DataMng->getRightPrec()->apply(opt_mng_, m_ProjectionOperatorTimesVec, m_DataMng->getRightPrecTimesVector());
        m_DataMng->getRightPrecTimesVector()->update(static_cast<Real>(1.), *m_DataMng->getSolution(), 1.);
        Real norm_solution = m_DataMng->getRightPrecTimesVector()->norm();
        if (norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getRightPrecTimesVector(), m_DataMng->getMatrixTimesVector());
        Real curvature = m_DataMng->getRightPrecTimesVector()->dot(*m_DataMng->getMatrixTimesVector());
        if (dotk::DOTk_KrylovSolver::checkCurvature(curvature) == true)
        {
            break;
        }
        m_DataMng->getResidual()->update(1., *rhs_vec_, 0.);
        m_DataMng->getResidual()->update(static_cast<Real>(-1.), *m_DataMng->getMatrixTimesVector(), 1.);
        Real norm_residual = m_DataMng->getResidual()->norm();
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(norm_residual);
        Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getRightPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(norm_residual, stopping_tolerance) == true)
        {
            break;
        }
        ++itr;
    }
    m_DataMng->getSolution()->update(1., *m_DataMng->getRightPrecTimesVector(), 0.);
}

void DOTk_PrecGMRES::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

void DOTk_PrecGMRES::solve(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                           const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                           const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_)
{
    this->gmres(rhs_vec_, criterion_, opt_prob_mng_);
}

const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_PrecGMRES::getDataMng() const
{
    return (m_DataMng);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_PrecGMRES::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_PrecGMRES::getDescentDirection()
{
    return (m_ProjectionOperatorTimesVec);
}

void DOTk_PrecGMRES::allocate(const std::shared_ptr<dotk::Vector<Real> > vec_)
{
    m_ProjectionOperatorTimesVec = vec_->clone();
}

}
