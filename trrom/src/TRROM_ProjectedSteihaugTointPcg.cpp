/*
 * TRROM_ProjectedSteihaugTointPcg.cpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_Preconditioner.hpp"
#include "TRROM_LinearOperator.hpp"
#include "TRROM_OptimizationDataMng.hpp"
#include "TRROM_ProjectedSteihaugTointPcg.hpp"

namespace trrom
{

ProjectedSteihaugTointPcg::ProjectedSteihaugTointPcg(const std::tr1::shared_ptr<trrom::Data> & data_) :
        trrom::SteihaugTointSolver(),
        m_Residual(data_->control()->create()),
        m_ActiveSet(data_->control()->create()),
        m_NewtonStep(data_->control()->create()),
        m_CauchyStep(data_->control()->create()),
        m_WorkVector(data_->control()->create()),
        m_InactiveSet(data_->control()->create()),
        m_ActiveVector(data_->control()->create()),
        m_InactiveVector(data_->control()->create()),
        m_ConjugateDirection(data_->control()->create()),
        m_PrecTimesNewtonStep(data_->control()->create()),
        m_InvPrecTimesResidual(data_->control()->create()),
        m_PrecTimesConjugateDirection(data_->control()->create()),
        m_HessTimesConjugateDirection(data_->control()->create())
{
    this->initialize();
}

ProjectedSteihaugTointPcg::~ProjectedSteihaugTointPcg()
{
}

const std::tr1::shared_ptr<trrom::Vector<double> > & ProjectedSteihaugTointPcg::getActiveSet() const
{
    return (m_ActiveSet);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & ProjectedSteihaugTointPcg::getInactiveSet() const
{
    return (m_InactiveSet);
}

void ProjectedSteihaugTointPcg::solve(const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                                      const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                                      const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_)
{
    m_NewtonStep->fill(0.);
    m_ConjugateDirection->fill(0);

    m_Residual->copy(*mng_->getNewGradient());
    m_Residual->elementWiseMultiplication(*m_InactiveSet);
    m_Residual->scale(static_cast<double>(-1.));

    this->iterate(preconditioner_, linear_operator_, mng_);
    if(trrom::SteihaugTointSolver::getNumItrDone() == 1)
    {
        trrom::SteihaugTointSolver::setStoppingCriterion(trrom::types::SOLVER_TOLERANCE_SATISFIED);
    }

    double norm_newton_step = m_NewtonStep->norm();
    if(norm_newton_step <= static_cast<double>(0.))
    {
        m_NewtonStep->copy(*mng_->getNewGradient());
        m_NewtonStep->scale(static_cast<double>(-1.));
        m_NewtonStep->elementWiseMultiplication(*m_InactiveSet);
    }
    mng_->getTrialStep()->copy(*m_NewtonStep);
}

void ProjectedSteihaugTointPcg::iterate(const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                                        const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                                        const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_)
{
    double previous_tau = 0;
    double norm_residual = m_Residual->norm();
    trrom::SteihaugTointSolver::setResidualNorm(norm_residual);
    double tolerance = trrom::SteihaugTointSolver::getSolverTolerance();
    double current_trust_region_radius = trrom::SteihaugTointSolver::getTrustRegionRadius();

    int itr = 1;
    int max_num_itr = trrom::SteihaugTointSolver::getMaxNumItr();
    while(norm_residual > tolerance)
    {
        if(itr > max_num_itr)
        {
            trrom::SteihaugTointSolver::setStoppingCriterion(trrom::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        this->applyVectorToInvPreconditioner(mng_, preconditioner_, m_Residual, m_InvPrecTimesResidual);
        //compute scaling
        double current_tau = m_Residual->dot(*m_InvPrecTimesResidual);
        if(itr > 1)
        {
            double beta = current_tau / previous_tau;
            m_ConjugateDirection->scale(beta);
            m_ConjugateDirection->axpy(static_cast<double>(1.), *m_InvPrecTimesResidual);
        }
        else
        {
            m_ConjugateDirection->copy(*m_InvPrecTimesResidual);
        }
        this->applyVectorToHessian(mng_, linear_operator_, m_ConjugateDirection, m_HessTimesConjugateDirection);
        double curvature = m_ConjugateDirection->dot(*m_HessTimesConjugateDirection);
        if(trrom::SteihaugTointSolver::invalidCurvatureDetected(curvature) == true)
        {
            // compute scaled inexact trial step
            double scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->axpy(scaling, *m_ConjugateDirection);
            break;
        }
        double rayleigh_quotient = current_tau / curvature;
        m_Residual->axpy(-rayleigh_quotient, *m_HessTimesConjugateDirection);
        norm_residual = m_Residual->norm();
        m_NewtonStep->axpy(rayleigh_quotient, *m_ConjugateDirection);
        if(itr == 1)
        {
            m_CauchyStep->copy(*m_NewtonStep);
        }
        double norm_newton_step = m_NewtonStep->norm();
        if(norm_newton_step > current_trust_region_radius)
        {
            // compute scaled inexact trial step
            double scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->axpy(scaling, *m_ConjugateDirection);
            trrom::SteihaugTointSolver::setStoppingCriterion(trrom::types::TRUST_REGION_VIOLATED);
            break;
        }
        previous_tau = current_tau;
        if(trrom::SteihaugTointSolver::toleranceSatisfied(norm_residual) == true)
        {
            break;
        }
        ++itr;
    }
    trrom::SteihaugTointSolver::setNumItrDone(itr);
}

void ProjectedSteihaugTointPcg::initialize()
{
    m_InactiveSet->fill(1);
}

double ProjectedSteihaugTointPcg::step(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                       const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_)
{
    this->applyVectorToPreconditioner(mng_, preconditioner_, m_NewtonStep, m_PrecTimesNewtonStep);
    this->applyVectorToPreconditioner(mng_, preconditioner_, m_ConjugateDirection, m_PrecTimesConjugateDirection);
    double scale_factor = trrom::SteihaugTointSolver::computeSteihaugTointStep(m_NewtonStep,
                                                                               m_ConjugateDirection,
                                                                               m_PrecTimesNewtonStep,
                                                                               m_PrecTimesConjugateDirection);
    return (scale_factor);
}

void ProjectedSteihaugTointPcg::applyVectorToHessian(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                                     const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                                                     const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                                     std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    m_ActiveVector->copy(*vector_);
    m_ActiveVector->elementWiseMultiplication(*m_ActiveSet);
    m_InactiveVector->copy(*vector_);
    m_InactiveVector->elementWiseMultiplication(*m_InactiveSet);

    output_->fill(0);
    linear_operator_->apply(mng_, m_InactiveVector, output_);

    output_->elementWiseMultiplication(*m_InactiveSet);
    output_->axpy(static_cast<double>(1.), *m_ActiveVector);
}

void ProjectedSteihaugTointPcg::applyVectorToPreconditioner(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                                            const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                                                            const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                                            std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    m_ActiveVector->copy(*vector_);
    m_ActiveVector->elementWiseMultiplication(*m_ActiveSet);
    m_InactiveVector->copy(*vector_);
    m_InactiveVector->elementWiseMultiplication(*m_InactiveSet);

    output_->fill(0);
    preconditioner_->applyPreconditioner(mng_, m_InactiveVector, output_);

    output_->elementWiseMultiplication(*m_InactiveSet);
    output_->axpy(static_cast<double>(1.), *m_ActiveVector);
}

void ProjectedSteihaugTointPcg::applyVectorToInvPreconditioner(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                                               const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_,
                                                               const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                                               std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    m_ActiveVector->copy(*vector_);
    m_ActiveVector->elementWiseMultiplication(*m_ActiveSet);
    m_InactiveVector->copy(*vector_);
    m_InactiveVector->elementWiseMultiplication(*m_InactiveSet);

    output_->fill(0);
    preconditioner_->applyInvPreconditioner(mng_, m_InactiveVector, output_);

    output_->elementWiseMultiplication(*m_InactiveSet);
    output_->axpy(static_cast<double>(1.), *m_ActiveVector);
}

}
