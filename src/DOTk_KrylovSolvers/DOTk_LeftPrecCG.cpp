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

DOTk_LeftPrecCG::DOTk_LeftPrecCG(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & solver_mng_) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CG),
        m_DataMng(solver_mng_),
        m_ConjugateDirection(solver_mng_->getSolution()->clone())
{
}

DOTk_LeftPrecCG::DOTk_LeftPrecCG(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_CG),
        m_DataMng(new dotk::DOTk_LeftPrecConjGradDataMng(primal_, linear_operator_)),
        m_ConjugateDirection()
{
    m_ConjugateDirection = m_DataMng->getSolution()->clone();
}

DOTk_LeftPrecCG::~DOTk_LeftPrecCG()
{
}

void DOTk_LeftPrecCG::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecCG::getDataMng() const
{
    return (m_DataMng);
}

const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecCG::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecCG::getDescentDirection()
{
    return (m_ConjugateDirection);
}

void DOTk_LeftPrecCG::initialize(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                                 const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                 const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(0);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->copy(*rhs_vec_);
    m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    m_ConjugateDirection->copy(*m_DataMng->getLeftPrecTimesVector());
    m_DataMng->getSolution()->fill(0.);

    Real res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
    Real residual_norm = std::sqrt(res_dot_prec_times_res);
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(residual_norm);
    Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecCG::pcg(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
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
        m_DataMng->getLinearOperator()->apply(opt_mng_, m_ConjugateDirection, m_DataMng->getMatrixTimesVector());
        Real curvature = m_ConjugateDirection->dot(*m_DataMng->getMatrixTimesVector());
        if (dotk::DOTk_KrylovSolver::checkCurvature(curvature) == true)
        {
            break;
        }
        m_DataMng->getPreviousSolution()->copy(*m_DataMng->getSolution());
        Real old_res_dot_prec_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getLeftPrecTimesVector());
        Real alpha = old_res_dot_prec_times_res / curvature;
        m_DataMng->getSolution()->axpy(alpha, *m_ConjugateDirection);
        Real norm_solution = m_DataMng->getSolution()->norm();
        Real trust_region_radius = dotk::DOTk_KrylovSolver::getTrustRegionRadius();
        if (norm_solution >= trust_region_radius)
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getResidual()->axpy(-alpha, *m_DataMng->getMatrixTimesVector());
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
        m_ConjugateDirection->axpy(static_cast<Real>(1.0), *m_DataMng->getLeftPrecTimesVector());
        ++itr;
    }
}

void DOTk_LeftPrecCG::solve(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                            const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                            const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->pcg(rhs_vec_, criterion_, opt_mng_);
}

}
