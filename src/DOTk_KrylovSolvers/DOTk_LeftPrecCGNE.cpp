/*
 * DOTk_LeftPrecCGNE.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

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

DOTk_LeftPrecCGNE::DOTk_LeftPrecCGNE(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CGNE),
        m_DataMng(mng_),
        m_AuxiliaryVector(mng_->getSolution()->clone()),
        m_ConjugateDirection(mng_->getSolution()->clone()),
        m_ConjugateDirectionNormalEq(mng_->getSolution()->clone())
{
}

DOTk_LeftPrecCGNE::DOTk_LeftPrecCGNE(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CGNE),
        m_DataMng(new dotk::DOTk_LeftPrecCGNEqDataMng(primal_, linear_operator_)),
        m_AuxiliaryVector(),
        m_ConjugateDirection(),
        m_ConjugateDirectionNormalEq()
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecCGNE::~DOTk_LeftPrecCGNE()
{
}

void DOTk_LeftPrecCGNE::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                                   const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                   const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->copy(*rhs_vec_);
    m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getLeftPrecTimesVector(), m_ConjugateDirection);
    m_DataMng->getSolution()->fill(0.);

    Real res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
    Real residual_norm = std::sqrt(res_dot_prec_times_res);
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
    Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCGNE::cgne(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
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
        m_DataMng->getLinearOperator()->apply(opt_mng_, m_ConjugateDirection, m_AuxiliaryVector);
        Real curvature = m_ConjugateDirection->dot(*m_AuxiliaryVector);
        if (dotk::DOTk_KrylovSolver::checkCurvature(curvature) == true)
        {
            break;
        }
        Real conjugate_dir_dot_conjugate_dir = m_ConjugateDirection->dot(*m_ConjugateDirection);
        Real old_res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
        Real alpha = old_res_dot_prec_times_res / conjugate_dir_dot_conjugate_dir;
        m_DataMng->getPreviousSolution()->copy(*m_DataMng->getSolution());
        m_DataMng->getSolution()->axpy(alpha, *m_ConjugateDirection);
        Real norm_solution = m_DataMng->getSolution()->norm();
        if (norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getResidual()->axpy(-alpha, *m_AuxiliaryVector);
        m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
        Real new_res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
        Real residual_norm = std::sqrt(new_res_dot_prec_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
        Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        Real beta = new_res_dot_prec_times_res / old_res_dot_prec_times_res;
        m_ConjugateDirection->scale(beta);
        m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getLeftPrecTimesVector(), m_ConjugateDirectionNormalEq);
        m_ConjugateDirection->axpy(static_cast<Real>(1.0), *m_ConjugateDirectionNormalEq);
        ++itr;
    }
}

void DOTk_LeftPrecCGNE::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> &
DOTk_LeftPrecCGNE::getDataMng() const
{
    return (m_DataMng);
}

const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> &
DOTk_LeftPrecCGNE::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LeftPrecCGNE::getDescentDirection()
{
    return (m_ConjugateDirection);
}

void DOTk_LeftPrecCGNE::solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                              const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                              const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->cgne(rhs_vec_, criterion_, opt_mng_);
}

void DOTk_LeftPrecCGNE::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > vector_)
{
    m_AuxiliaryVector = vector_->clone();
    m_ConjugateDirection = vector_->clone();
    m_ConjugateDirectionNormalEq = vector_->clone();
}

}
