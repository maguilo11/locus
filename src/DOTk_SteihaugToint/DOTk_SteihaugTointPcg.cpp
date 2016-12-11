/*
 * DOTk_SteihaugTointPcg.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Preconditioner.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_SteihaugTointPcg.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_SteihaugTointPcg::DOTk_SteihaugTointPcg(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_SteihaugTointSolver(),
        m_NewtonStep(primal_->control()->clone()),
        m_CauchyStep(primal_->control()->clone()),
        m_ConjugateDirection(primal_->control()->clone()),
        m_NewDescentDirection(primal_->control()->clone()),
        m_OldDescentDirection(primal_->control()->clone()),
        m_PrecTimesNewtonStep(primal_->control()->clone()),
        m_PrecTimesConjugateDirection(primal_->control()->clone()),
        m_HessTimesConjugateDirection(primal_->control()->clone()),
        m_NewInvPrecTimesDescentDirection(primal_->control()->clone()),
        m_OldInvPrecTimesDescentDirection(primal_->control()->clone())
{
}

DOTk_SteihaugTointPcg::~DOTk_SteihaugTointPcg()
{
}

void DOTk_SteihaugTointPcg::solve(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                                  const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                  const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_NewtonStep->fill(0.);
    m_OldDescentDirection->copy(*mng_->getNewGradient());
    preconditioner_->applyInvPreconditioner(mng_, m_OldDescentDirection, m_OldInvPrecTimesDescentDirection);
    m_ConjugateDirection->copy(*m_OldInvPrecTimesDescentDirection);
    m_ConjugateDirection->scale(-1.);

    this->computeStoppingTolerance(m_OldDescentDirection);
    Real current_trust_region_radius = this->getTrustRegionRadius();

    size_t itr = 1;
    size_t max_num_itr = dotk::DOTk_SteihaugTointSolver::getMaxNumItr();
    while(1)
    {
        if(itr > max_num_itr)
        {
            this->setStoppingCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        linear_operator_->apply(mng_, m_ConjugateDirection, m_HessTimesConjugateDirection);
        Real curvature = m_ConjugateDirection->dot(*m_HessTimesConjugateDirection);
        if(this->invalidCurvatureDetected(curvature) == true)
        {
            Real scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->axpy(scaling, *m_ConjugateDirection);
            break;
        }
        Real alpha = m_OldDescentDirection->dot(*m_OldInvPrecTimesDescentDirection) / curvature;
        m_NewtonStep->axpy(alpha, *m_ConjugateDirection);
        if(itr == 1)
        {
            m_CauchyStep->copy(*m_NewtonStep);
        }
        Real norm_newton_step = m_NewtonStep->norm();
        if(norm_newton_step >= current_trust_region_radius)
        {
            Real scaling = this->step(mng_, preconditioner_);
            m_NewtonStep->axpy(scaling, *m_ConjugateDirection);
            this->setStoppingCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_NewDescentDirection->copy(*m_OldDescentDirection);
        m_NewDescentDirection->axpy(alpha, *m_HessTimesConjugateDirection);
        Real norm_new_descent_dir = m_NewDescentDirection->norm();
        if(dotk::DOTk_SteihaugTointSolver::toleranceSatisfied(norm_new_descent_dir) == true)
        {
            this->setResidualNorm(norm_new_descent_dir);
            break;
        }
        preconditioner_->applyInvPreconditioner(mng_, m_NewDescentDirection, m_NewInvPrecTimesDescentDirection);
        Real numerator = m_NewDescentDirection->dot(*m_NewInvPrecTimesDescentDirection);
        Real denominator = m_OldDescentDirection->dot(*m_OldInvPrecTimesDescentDirection);
        Real rayleigh_quotient = numerator / denominator;
        m_ConjugateDirection->scale(rayleigh_quotient);
        m_ConjugateDirection->axpy(-1., *m_NewInvPrecTimesDescentDirection);
        ++itr;
        m_OldDescentDirection->copy(*m_NewDescentDirection);
        m_OldInvPrecTimesDescentDirection->copy(*m_NewInvPrecTimesDescentDirection);
    }
    this->setNumItrDone(itr);
    mng_->getTrialStep()->copy(*m_NewtonStep);
}

void DOTk_SteihaugTointPcg::computeStoppingTolerance(const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    Real relative_tolerance = this->getRelativeTolerance();
    Real relative_tolerance_exponential = this->getRelativeToleranceExponential();
    Real norm_current_gradient = gradient_->norm();
    Real tolerance_condition = std::pow(norm_current_gradient, relative_tolerance_exponential);
    relative_tolerance = std::min(relative_tolerance, tolerance_condition);
    Real stopping_tolerance = norm_current_gradient * relative_tolerance;
    this->setSolverTolerance(stopping_tolerance);
}

Real DOTk_SteihaugTointPcg::step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_)
{
    preconditioner_->applyPreconditioner(mng_, m_NewtonStep, m_PrecTimesNewtonStep);
    preconditioner_->applyPreconditioner(mng_, m_ConjugateDirection, m_PrecTimesConjugateDirection);
    Real scale_factor = dotk::DOTk_SteihaugTointSolver::computeSteihaugTointStep(m_NewtonStep,
                                                                                 m_ConjugateDirection,
                                                                                 m_PrecTimesNewtonStep,
                                                                                 m_PrecTimesConjugateDirection);
    return (scale_factor);
}

}
