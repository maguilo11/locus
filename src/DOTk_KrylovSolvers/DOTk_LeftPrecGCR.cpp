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

DOTk_LeftPrecGCR::DOTk_LeftPrecGCR(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_GCR),
        m_DataMng(aSolverDataMng),
        mBetaCoefficients(aSolverDataMng->getMaxNumSolverItr()),
        mConjugateDirectionStorage(aSolverDataMng->getMaxNumSolverItr()),
        mLinearOperatorTimesConjugateDirStorage(aSolverDataMng->getMaxNumSolverItr())
{
    this->initialize(aSolverDataMng->getSolution());
}

DOTk_LeftPrecGCR::DOTk_LeftPrecGCR(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                   const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                   size_t max_num_itr_) :
        dotk::DOTk_KrylovSolver(dotk::types::LEFT_PREC_GCR),
        m_DataMng(std::make_shared<dotk::DOTk_LeftPrecGenConjResDataMng>(aPrimal, linear_operator_, max_num_itr_)),
        mBetaCoefficients(max_num_itr_),
        mConjugateDirectionStorage(max_num_itr_),
        mLinearOperatorTimesConjugateDirStorage(max_num_itr_)
{
    this->initialize(m_DataMng->getSolution());
}

DOTk_LeftPrecGCR::~DOTk_LeftPrecGCR()
{
}

void DOTk_LeftPrecGCR::initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                                  const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                                  const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    size_t itr_done = 0;
    dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr_done);
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);

    m_DataMng->getResidual()->update(1., *aRhsVector, 0.);
    m_DataMng->getLeftPrec()->apply(aMng, m_DataMng->getResidual(), m_DataMng->getLeftPrecTimesVector());
    mConjugateDirectionStorage[itr_done]->update(1., *m_DataMng->getLeftPrecTimesVector(), 0.);
    m_DataMng->getLinearOperator()->apply(aMng,
                                          m_DataMng->getResidual(),
                                          mLinearOperatorTimesConjugateDirStorage[itr_done]);
    m_DataMng->getLinearOperator()->apply(aMng,
                                          mConjugateDirectionStorage[itr_done],
                                          m_DataMng->getMatrixTimesVector());
    m_DataMng->getSolution()->fill(0.);

    Real prec_residual_norm = m_DataMng->getLeftPrecTimesVector()->norm();
    dotk::DOTk_KrylovSolver::setSolverResidualNorm(prec_residual_norm);
    Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
    dotk::DOTk_KrylovSolver::setInitialStoppingTolerance(stopping_tolerance);
}

void DOTk_LeftPrecGCR::pgcr(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                            const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                            const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->initialize(aRhsVector, aCriterion, aMng);
    if(dotk::DOTk_KrylovSolver::checkCurvature(dotk::DOTk_KrylovSolver::getSolverResidualNorm()) == true)
    {
        return;
    }
    size_t itr = 0;
    while(1)
    {
        if(itr == m_DataMng->getMaxNumSolverItr())
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr + 1);
        m_DataMng->getLeftPrec()->apply(aMng,
                                        mLinearOperatorTimesConjugateDirStorage[itr],
                                        m_DataMng->getLeftPrecTimesVector());
        Real scaled_curvature = mLinearOperatorTimesConjugateDirStorage[itr]->dot(*m_DataMng->getLeftPrecTimesVector());
        if(dotk::DOTk_KrylovSolver::checkCurvature(scaled_curvature) == true)
        {
            break;
        }
        Real old_res_dot_linear_operator_times_conjugate_dir =
                m_DataMng->getResidual()->dot(*mLinearOperatorTimesConjugateDirStorage[itr]);
        Real alpha = old_res_dot_linear_operator_times_conjugate_dir / scaled_curvature;
        m_DataMng->getPreviousSolution()->update(1., *m_DataMng->getSolution(), 0.);
        m_DataMng->getSolution()->update(alpha, *mConjugateDirectionStorage[itr], 1.);
        Real norm_solution = m_DataMng->getSolution()->norm();
        if(norm_solution >= dotk::DOTk_KrylovSolver::getTrustRegionRadius())
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getResidual()->update(-alpha, *m_DataMng->getLeftPrecTimesVector(), 1.);
        m_DataMng->getLinearOperator()->apply(aMng, m_DataMng->getResidual(), m_DataMng->getMatrixTimesVector());
        Real res_dot_linear_operator_times_res = m_DataMng->getResidual()->dot(*m_DataMng->getMatrixTimesVector());
        Real scaled_residual_norm = std::sqrt(res_dot_linear_operator_times_res);
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(scaled_residual_norm);
        Real stopping_tolerance = aCriterion->evaluate(this, m_DataMng->getLeftPrecTimesVector());
        if(dotk::DOTk_KrylovSolver::checkResidualNorm(scaled_residual_norm, stopping_tolerance) == true)
        {
            break;
        }
        this->updateBetaCoefficientStorage(itr);
        this->updateConjugateDirectionStorage(itr);
        this->updateLinearOperatorTimesConjugateDirStorage(itr);
        ++ itr;
    }
}

void DOTk_LeftPrecGCR::setMaxNumKrylovSolverItr(size_t aMaxNumIterations)
{
    m_DataMng->setMaxNumSolverItr(aMaxNumIterations);
}

const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_LeftPrecGCR::getDataMng() const
{
    return (m_DataMng);
}

const std::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_LeftPrecGCR::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_LeftPrecGCR::getDescentDirection()
{
    size_t index = dotk::DOTk_KrylovSolver::getNumSolverItrDone() - 1;
    return (mConjugateDirectionStorage[index]);
}

void DOTk_LeftPrecGCR::solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                             const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                             const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    this->pgcr(aRhsVector, aCriterion, aMng);
}

void DOTk_LeftPrecGCR::initialize(const std::shared_ptr<dotk::Vector<Real> > aVector)
{
    size_t dimensions = m_DataMng->getMaxNumSolverItr();
    for(size_t row = 0; row < dimensions; ++ row)
    {
        mConjugateDirectionStorage[row] = aVector->clone();
        mLinearOperatorTimesConjugateDirStorage[row] = aVector->clone();
    }
}

void DOTk_LeftPrecGCR::updateBetaCoefficientStorage(size_t aCurrentSolverIteration)
{
    Real innr_linear_operator_times_res_dot_linear_operator_times_conj_dir =
            m_DataMng->getMatrixTimesVector()->dot(*mLinearOperatorTimesConjugateDirStorage[aCurrentSolverIteration]);
    Real innr_linear_operator_times_conj_dir_dot_linear_operator_times_conj_dir =
            mLinearOperatorTimesConjugateDirStorage[aCurrentSolverIteration]->dot(*mLinearOperatorTimesConjugateDirStorage[aCurrentSolverIteration]);
    mBetaCoefficients[aCurrentSolverIteration] = -innr_linear_operator_times_res_dot_linear_operator_times_conj_dir
            / innr_linear_operator_times_conj_dir_dot_linear_operator_times_conj_dir;
}

void DOTk_LeftPrecGCR::updateConjugateDirectionStorage(size_t aCurrentSolverIteration)
{
    size_t tNexSolverIteration = aCurrentSolverIteration + 1;
    if(tNexSolverIteration < m_DataMng->getMaxNumSolverItr())
    {
        mConjugateDirectionStorage[tNexSolverIteration]->update(1., *m_DataMng->getResidual(), 0.);
        for(size_t tIndex = 0; tIndex <= aCurrentSolverIteration; ++ tIndex)
        {
            mConjugateDirectionStorage[tNexSolverIteration]->update(mBetaCoefficients.operator [](tIndex),
                                                                    *mConjugateDirectionStorage[tIndex],
                                                                    static_cast<Real>(1.));
        }
    }
    else
    {
        return;
    }
}

void DOTk_LeftPrecGCR::updateLinearOperatorTimesConjugateDirStorage(size_t aCurrentSolverIteration)
{
    size_t tNextIteration = aCurrentSolverIteration + 1;
    if(tNextIteration < m_DataMng->getMaxNumSolverItr())
    {
        mLinearOperatorTimesConjugateDirStorage[tNextIteration]->update(1., *m_DataMng->getMatrixTimesVector(), 0.);
        for(size_t tIndex = 0; tIndex <= aCurrentSolverIteration; ++ tIndex)
        {
            mLinearOperatorTimesConjugateDirStorage[tNextIteration]->update(mBetaCoefficients.operator [](tIndex),
                                                                            *mLinearOperatorTimesConjugateDirStorage[tIndex],
                                                                            static_cast<Real>(1.));
        }
    }
    else
    {
        return;
    }
}

}
