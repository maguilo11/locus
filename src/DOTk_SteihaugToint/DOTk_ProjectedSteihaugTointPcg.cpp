/*
 * DOTk_ProjectedSteihaugTointPcg.cpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Preconditioner.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_ProjectedSteihaugTointPcg.hpp"

namespace dotk
{

DOTk_ProjectedSteihaugTointPcg::DOTk_ProjectedSteihaugTointPcg(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_SteihaugTointSolver(),
        m_Residual(primal_->control()->clone()),
        m_ActiveSet(primal_->control()->clone()),
        m_NewtonStep(primal_->control()->clone()),
        m_CauchyStep(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_InactiveSet(primal_->control()->clone()),
        m_ActiveVector(primal_->control()->clone()),
        m_InactiveVector(primal_->control()->clone()),
        m_ConjugateDirection(primal_->control()->clone()),
        m_PrecTimesNewtonStep(primal_->control()->clone()),
        m_InvPrecTimesResidual(primal_->control()->clone()),
        m_PrecTimesConjugateDirection(primal_->control()->clone()),
        m_HessTimesConjugateDirection(primal_->control()->clone())
{
    this->initialize();
}

DOTk_ProjectedSteihaugTointPcg::~DOTk_ProjectedSteihaugTointPcg()
{
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_ProjectedSteihaugTointPcg::getActiveSet() const
{
    return (m_ActiveSet);
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_ProjectedSteihaugTointPcg::getInactiveSet() const
{
    return (m_InactiveSet);
}

void DOTk_ProjectedSteihaugTointPcg::solve(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                           const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                           const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_NewtonStep->fill(0.);
    m_ConjugateDirection->fill(0);

    m_Residual->copy(*mng_->getNewGradient());
    m_Residual->cwiseProd(*m_InactiveSet);
    m_Residual->scale(static_cast<Real>(-1.));

    this->iterate(preconditioner_, linear_operator_, mng_);
    if(dotk::DOTk_SteihaugTointSolver::getNumItrDone() == 1)
    {
        dotk::DOTk_SteihaugTointSolver::setStoppingCriterion(dotk::types::SOLVER_TOLERANCE_SATISFIED);
    }

    Real norm_newton_step = m_NewtonStep->norm();
    if(norm_newton_step <= static_cast<Real>(0.))
    {
        m_NewtonStep->copy(*mng_->getNewGradient());
        m_NewtonStep->scale(static_cast<Real>(-1.));
        m_NewtonStep->cwiseProd(*m_InactiveSet);
    }
    mng_->getTrialStep()->copy(*m_NewtonStep);
}

void DOTk_ProjectedSteihaugTointPcg::iterate(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                             const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                             const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real previous_tau = 0;
    Real norm_residual = m_Residual->norm();
    dotk::DOTk_SteihaugTointSolver::setResidualNorm(norm_residual);
    Real tolerance = dotk::DOTk_SteihaugTointSolver::getSolverTolerance();
    Real current_trust_region_radius = dotk::DOTk_SteihaugTointSolver::getTrustRegionRadius();

    size_t itr = 1;
    size_t max_num_itr = dotk::DOTk_SteihaugTointSolver::getMaxNumItr();
    while(norm_residual > tolerance)
    {
        if(itr > max_num_itr)
        {
            dotk::DOTk_SteihaugTointSolver::setStoppingCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        this->applyVectorToInvPreconditioner(mng_, preconditioner_, m_Residual, m_InvPrecTimesResidual);
        //compute scaling
        Real current_tau = m_Residual->dot(*m_InvPrecTimesResidual);
        if(itr > 1)
        {
            Real beta = current_tau / previous_tau;
            m_ConjugateDirection->scale(beta);
            m_ConjugateDirection->axpy(static_cast<Real>(1.), *m_InvPrecTimesResidual);
        }
        else
        {
            m_ConjugateDirection->copy(*m_InvPrecTimesResidual);
        }
        this->applyVectorToHessian(mng_, linear_operator_, m_ConjugateDirection, m_HessTimesConjugateDirection);
        Real curvature = m_ConjugateDirection->dot(*m_HessTimesConjugateDirection);
        if(dotk::DOTk_SteihaugTointSolver::invalidCurvatureDetected(curvature) == true)
        {
            // compute scaled inexact trial step
            Real scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->axpy(scaling, *m_ConjugateDirection);
            break;
        }
        Real rayleigh_quotient = current_tau / curvature;
        m_Residual->axpy(-rayleigh_quotient, *m_HessTimesConjugateDirection);
        norm_residual = m_Residual->norm();
        m_NewtonStep->axpy(rayleigh_quotient, *m_ConjugateDirection);
        if(itr == 1)
        {
            m_CauchyStep->copy(*m_NewtonStep);
        }
        Real norm_newton_step = m_NewtonStep->norm();
        if(norm_newton_step > current_trust_region_radius)
        {
            // compute scaled inexact trial step
            Real scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->axpy(scaling, *m_ConjugateDirection);
            dotk::DOTk_SteihaugTointSolver::setStoppingCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        previous_tau = current_tau;
        if(dotk::DOTk_SteihaugTointSolver::toleranceSatisfied(norm_residual) == true)
        {
            break;
        }
        ++itr;
    }
    dotk::DOTk_SteihaugTointSolver::setNumItrDone(itr);
}

void DOTk_ProjectedSteihaugTointPcg::initialize()
{
    m_InactiveSet->fill(1);
}

Real DOTk_ProjectedSteihaugTointPcg::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                          const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_)
{
    this->applyVectorToPreconditioner(mng_, preconditioner_, m_NewtonStep, m_PrecTimesNewtonStep);
    this->applyVectorToPreconditioner(mng_, preconditioner_, m_ConjugateDirection, m_PrecTimesConjugateDirection);
    Real scale_factor = dotk::DOTk_SteihaugTointSolver::computeSteihaugTointStep(m_NewtonStep,
                                                                                 m_ConjugateDirection,
                                                                                 m_PrecTimesNewtonStep,
                                                                                 m_PrecTimesConjugateDirection);
    return (scale_factor);
}

void DOTk_ProjectedSteihaugTointPcg::applyVectorToHessian(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                          const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                          const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                                          std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    m_ActiveVector->copy(*vector_);
    m_ActiveVector->cwiseProd(*m_ActiveSet);
    m_InactiveVector->copy(*vector_);
    m_InactiveVector->cwiseProd(*m_InactiveSet);

    output_->fill(0);
    linear_operator_->apply(mng_, m_InactiveVector, output_);

    output_->cwiseProd(*m_InactiveSet);
    output_->axpy(static_cast<Real>(1.), *m_ActiveVector);
}

void DOTk_ProjectedSteihaugTointPcg::applyVectorToPreconditioner(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                                 const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                                                 const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                                                 std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    m_ActiveVector->copy(*vector_);
    m_ActiveVector->cwiseProd(*m_ActiveSet);
    m_InactiveVector->copy(*vector_);
    m_InactiveVector->cwiseProd(*m_InactiveSet);

    output_->fill(0);
    preconditioner_->applyPreconditioner(mng_, m_InactiveVector, output_);

    output_->cwiseProd(*m_InactiveSet);
    output_->axpy(static_cast<Real>(1.), *m_ActiveVector);
}

void DOTk_ProjectedSteihaugTointPcg::applyVectorToInvPreconditioner(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                                    const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                                                                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    m_ActiveVector->copy(*vector_);
    m_ActiveVector->cwiseProd(*m_ActiveSet);
    m_InactiveVector->copy(*vector_);
    m_InactiveVector->cwiseProd(*m_InactiveSet);

    output_->fill(0);
    preconditioner_->applyInvPreconditioner(mng_, m_InactiveVector, output_);

    output_->cwiseProd(*m_InactiveSet);
    output_->axpy(static_cast<Real>(1.), *m_ActiveVector);
}

}
