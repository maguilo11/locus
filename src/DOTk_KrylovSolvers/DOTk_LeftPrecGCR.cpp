/*
 * DOTk_LeftPrecGCR.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPrecGCR.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_LeftPrecGenConjResDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_LeftPrecGCR::DOTk_LeftPrecGCR(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & solver_mng_) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_GCR),
        m_DataMng(solver_mng_),
        mBetaCoefficients(solver_mng_->getMaxNumSolverItr()),
        mConjugateDirectionStorage(solver_mng_->getMaxNumSolverItr()),
        mLinearOperatorTimesConjugateDirStorage(solver_mng_->getMaxNumSolverItr())
{
    this->initialize(solver_mng_->getSolution());
}

DOTk_LeftPrecGCR::DOTk_LeftPrecGCR(const std::tr1::shared_ptr<dotk::DOTk_Primal> & variable_,
                                   const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                   size_t max_num_itr_) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_GCR),
        m_DataMng(new dotk::DOTk_LeftPrecGenConjResDataMng(variable_, linear_operator_, max_num_itr_)),
        mBetaCoefficients(max_num_itr_),
        mConjugateDirectionStorage(max_num_itr_),
        mLinearOperatorTimesConjugateDirStorage(max_num_itr_)
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecGCR::~DOTk_LeftPrecGCR()
{
}

void DOTk_LeftPrecGCR::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                                  const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                  const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    size_t itr_done = 0;
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr_done);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->copy(*rhs_vec_);
    m_DataMng->getLeftPrec()->apply(opt_mng_, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    mConjugateDirectionStorage[itr_done]->copy(*m_DataMng->getLeftPrecTimesVector());
    m_DataMng->getLinearOperator()->apply(opt_mng_,
                                          m_DataMng->getResidual(),
                                          mLinearOperatorTimesConjugateDirStorage[itr_done]);
    m_DataMng->getLinearOperator()->apply(opt_mng_,
                                          mConjugateDirectionStorage[itr_done],
                                          m_DataMng->getMatrixTimesVector());
    m_DataMng->getSolution()->fill(0.);

    Real prec_residual_norm = m_DataMng->getLeftPrecTimesVector()->norm();
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(prec_residual_norm);
    Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecGCR::pgcr(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                            const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                            const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->initialize(rhs_vec_, criterion_, opt_mng_);
    if(dotk::DOTk_KrylovSolver::checkCurvature(dotk::DOTk_KrylovSolver::getSolverResidualNorm()) == true)
    {
        return;
    }
    size_t itr = 0;
    while (1)
    {
        if (itr == m_DataMng->getMaxNumSolverItr())
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr + 1);
        m_DataMng->getLeftPrec()->apply(opt_mng_, mLinearOperatorTimesConjugateDirStorage[itr],
                m_DataMng->getLeftPrecTimesVector());
        Real scaled_curvature =
                mLinearOperatorTimesConjugateDirStorage[itr]->dot(*m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkCurvature(scaled_curvature) == true)
        {
            break;
        }
        Real old_res_dot_linear_operator_times_conjugate_dir =
                m_DataMng->getResidual()->dot(*mLinearOperatorTimesConjugateDirStorage[itr]);
        Real alpha = old_res_dot_linear_operator_times_conjugate_dir / scaled_curvature;
        m_DataMng->getPreviousSolution()->copy(*m_DataMng->getSolution());
        m_DataMng->getSolution()->axpy(alpha, *mConjugateDirectionStorage[itr]);
        Real norm_solution = m_DataMng->getSolution()->norm();
        if (norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getResidual()->axpy(-alpha, *m_DataMng->getLeftPrecTimesVector());
        m_DataMng->getLinearOperator()->apply(opt_mng_, m_DataMng->getResidual(), m_DataMng->getMatrixTimesVector());
        Real res_dot_linear_operator_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getMatrixTimesVector());
        Real scaled_residual_norm = std::sqrt(res_dot_linear_operator_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(scaled_residual_norm);
        Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if (dotk::DOTk_KrylovSolver::checkResidualNorm(scaled_residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        this->updateBetaCoefficientStorage(itr);
        this->updateConjugateDirectionStorage(itr);
        this->updateLinearOperatorTimesConjugateDirStorage(itr);
        ++itr;
    }
}

void DOTk_LeftPrecGCR::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecGCR::getDataMng() const
{
    return (m_DataMng);
}

const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecGCR::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_LeftPrecGCR::getDescentDirection()
{
    size_t index = dotk::DOTk_KrylovSolver::getNumSolverItrDone() - 1;
    return (mConjugateDirectionStorage[index]);
}

void DOTk_LeftPrecGCR::solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                             const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                             const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_)
{
    this->pgcr(rhs_vec_, criterion_, opt_mng_);
}

void DOTk_LeftPrecGCR::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > vector_)
{
    size_t dimensions = m_DataMng->getMaxNumSolverItr();
    for(size_t row = 0; row < dimensions; ++row)
    {
        mConjugateDirectionStorage[row] = vector_->clone();
        mLinearOperatorTimesConjugateDirStorage[row] = vector_->clone();
    }
}

void DOTk_LeftPrecGCR::updateBetaCoefficientStorage(size_t current_itr_)
{
    Real innr_linear_operator_times_res_dot_linear_operator_times_conj_dir =
            m_DataMng->getMatrixTimesVector()->dot(*mLinearOperatorTimesConjugateDirStorage[current_itr_]);
    Real innr_linear_operator_times_conj_dir_dot_linear_operator_times_conj_dir =
            mLinearOperatorTimesConjugateDirStorage[current_itr_]->dot(*mLinearOperatorTimesConjugateDirStorage[current_itr_]);
    mBetaCoefficients[current_itr_] = -innr_linear_operator_times_res_dot_linear_operator_times_conj_dir
            / innr_linear_operator_times_conj_dir_dot_linear_operator_times_conj_dir;
}

void DOTk_LeftPrecGCR::updateConjugateDirectionStorage(size_t current_itr_)
{
    size_t next_itr = current_itr_ + 1;
    if(next_itr < m_DataMng->getMaxNumSolverItr())
    {
        mConjugateDirectionStorage[next_itr]->copy(*m_DataMng->getResidual());
        for(size_t i = 0; i <= current_itr_; ++i)
        {
            mConjugateDirectionStorage[next_itr]->axpy(mBetaCoefficients[i], *mConjugateDirectionStorage[i]);
        }
    }
    else
    {
        return;
    }
}

void DOTk_LeftPrecGCR::updateLinearOperatorTimesConjugateDirStorage(size_t current_itr_)
{
    size_t next_itr = current_itr_ + 1;
    if(next_itr < m_DataMng->getMaxNumSolverItr())
    {
        mLinearOperatorTimesConjugateDirStorage[next_itr]->copy(*m_DataMng->getMatrixTimesVector());
        for(size_t i = 0; i <= current_itr_; ++i)
        {
            mLinearOperatorTimesConjugateDirStorage[next_itr]->axpy(mBetaCoefficients[i],
                                                                    *mLinearOperatorTimesConjugateDirStorage[i]);
        }
    }
    else
    {
        return;
    }
}

}
