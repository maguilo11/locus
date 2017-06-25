/*
 * TRROM_SteihaugTointPcg.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_Preconditioner.hpp"
#include "TRROM_LinearOperator.hpp"
#include "TRROM_SteihaugTointPcg.hpp"
#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

SteihaugTointPcg::SteihaugTointPcg(const std::shared_ptr<trrom::Data> & data_) :
        trrom::SteihaugTointSolver(),
        m_NewtonStep(data_->control()->create()),
        m_CauchyStep(data_->control()->create()),
        m_ConjugateDirection(data_->control()->create()),
        m_NewDescentDirection(data_->control()->create()),
        m_OldDescentDirection(data_->control()->create()),
        m_PrecTimesNewtonStep(data_->control()->create()),
        m_PrecTimesConjugateDirection(data_->control()->create()),
        m_HessTimesConjugateDirection(data_->control()->create()),
        m_NewInvPrecTimesDescentDirection(data_->control()->create()),
        m_OldInvPrecTimesDescentDirection(data_->control()->create())
{
}

SteihaugTointPcg::~SteihaugTointPcg()
{
}

void SteihaugTointPcg::solve(const std::shared_ptr<trrom::Preconditioner> & preconditioner_,
                             const std::shared_ptr<trrom::LinearOperator> & linear_operator_,
                             const std::shared_ptr<trrom::OptimizationDataMng> & mng_)
{
    m_NewtonStep->fill(0.);
    m_OldDescentDirection->update(1., *mng_->getNewGradient(), 0.);
    preconditioner_->applyInvPreconditioner(mng_, m_OldDescentDirection, m_OldInvPrecTimesDescentDirection);
    m_ConjugateDirection->update(1., *m_OldInvPrecTimesDescentDirection, 0.);
    m_ConjugateDirection->scale(-1.);

    this->computeStoppingTolerance(m_OldDescentDirection);
    double current_trust_region_radius = this->getTrustRegionRadius();

    int itr = 1;
    int max_num_itr = trrom::SteihaugTointSolver::getMaxNumItr();
    while(1)
    {
        if(itr > max_num_itr)
        {
            this->setStoppingCriterion(trrom::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        linear_operator_->apply(mng_, m_ConjugateDirection, m_HessTimesConjugateDirection);
        double curvature = m_ConjugateDirection->dot(*m_HessTimesConjugateDirection);
        if(this->invalidCurvatureDetected(curvature) == true)
        {
            double scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->update(scaling, *m_ConjugateDirection, 1.);
            break;
        }
        double alpha = m_OldDescentDirection->dot(*m_OldInvPrecTimesDescentDirection) / curvature;
        m_NewtonStep->update(alpha, *m_ConjugateDirection, 1.);
        if(itr == 1)
        {
            m_CauchyStep->update(1., *m_NewtonStep, 0.);
        }
        double norm_newton_step = m_NewtonStep->norm();
        if(norm_newton_step >= current_trust_region_radius)
        {
            double scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->update(scaling, *m_ConjugateDirection, 1.);
            this->setStoppingCriterion(trrom::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_NewDescentDirection->update(1., *m_OldDescentDirection, 0.);
        m_NewDescentDirection->update(alpha, *m_HessTimesConjugateDirection, 1.);
        double norm_new_descent_dir = m_NewDescentDirection->norm();
        if(trrom::SteihaugTointSolver::toleranceSatisfied(norm_new_descent_dir) == true)
        {
            this->setResidualNorm(norm_new_descent_dir);
            break;
        }
        preconditioner_->applyInvPreconditioner(mng_, m_NewDescentDirection, m_NewInvPrecTimesDescentDirection);
        double numerator = m_NewDescentDirection->dot(*m_NewInvPrecTimesDescentDirection);
        double denominator = m_OldDescentDirection->dot(*m_OldInvPrecTimesDescentDirection);
        double rayleigh_quotient = numerator / denominator;
        m_ConjugateDirection->scale(rayleigh_quotient);
        m_ConjugateDirection->update(-1., *m_NewInvPrecTimesDescentDirection, 1.);
        ++ itr;
        m_OldDescentDirection->update(1., *m_NewDescentDirection, 0.);
        m_OldInvPrecTimesDescentDirection->update(1., *m_NewInvPrecTimesDescentDirection, 0.);
    }
    this->setNumItrDone(itr);
    mng_->getTrialStep()->update(1., *m_NewtonStep, 0.);
}

void SteihaugTointPcg::computeStoppingTolerance(const std::shared_ptr<trrom::Vector<double> > & gradient_)
{
    double relative_tolerance = this->getRelativeTolerance();
    double relative_tolerance_exponential = this->getRelativeToleranceExponential();
    double norm_current_gradient = gradient_->norm();
    double tolerance_condition = std::pow(norm_current_gradient, relative_tolerance_exponential);
    relative_tolerance = std::min(relative_tolerance, tolerance_condition);
    double stopping_tolerance = norm_current_gradient * relative_tolerance;
    this->setSolverTolerance(stopping_tolerance);
}

double SteihaugTointPcg::step(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                              const std::shared_ptr<trrom::Preconditioner> & preconditioner_)
{
    preconditioner_->applyPreconditioner(mng_, m_NewtonStep, m_PrecTimesNewtonStep);
    preconditioner_->applyPreconditioner(mng_, m_ConjugateDirection, m_PrecTimesConjugateDirection);
    double scale_factor = trrom::SteihaugTointSolver::computeSteihaugTointStep(m_NewtonStep,
                                                                               m_ConjugateDirection,
                                                                               m_PrecTimesNewtonStep,
                                                                               m_PrecTimesConjugateDirection);
    return (scale_factor);
}

}
