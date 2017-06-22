/*
 * DOTk_SteihaugTointSolver.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>
#include <math.h>
#include "vector.hpp"
#include "DOTk_SteihaugTointSolver.hpp"

namespace dotk
{

DOTk_SteihaugTointSolver::DOTk_SteihaugTointSolver() :
        m_MaxNumItr(200),
        m_NumItrDone(0),
        m_Tolerance(1e-8),
        m_ResidualNorm(0),
        m_TrustRegionRadius(0),
        m_RelativeTolerance(1e-1),
        m_RelativeToleranceExponential(0.5),
        m_StoppingCriterion(dotk::types::SOLVER_DID_NOT_CONVERGED)
{
}

DOTk_SteihaugTointSolver::~DOTk_SteihaugTointSolver()
{
}

void DOTk_SteihaugTointSolver::setMaxNumItr(size_t input_)
{
    m_MaxNumItr = input_;
}

size_t DOTk_SteihaugTointSolver::getMaxNumItr() const
{
    return (m_MaxNumItr);
}

void DOTk_SteihaugTointSolver::setNumItrDone(size_t input_)
{
    m_NumItrDone = input_;
}

size_t DOTk_SteihaugTointSolver::getNumItrDone() const
{
    return (m_NumItrDone);
}

void DOTk_SteihaugTointSolver::setSolverTolerance(Real input_)
{
    m_Tolerance = input_;
}

Real DOTk_SteihaugTointSolver::getSolverTolerance() const
{
    return (m_Tolerance);
}

void DOTk_SteihaugTointSolver::setTrustRegionRadius(Real input_)
{
    m_TrustRegionRadius = input_;
}

Real DOTk_SteihaugTointSolver::getTrustRegionRadius() const
{
    return (m_TrustRegionRadius);
}

void DOTk_SteihaugTointSolver::setResidualNorm(Real input_)
{
    m_ResidualNorm = input_;
}

Real DOTk_SteihaugTointSolver::getResidualNorm() const
{
    return (m_ResidualNorm);
}

void DOTk_SteihaugTointSolver::setRelativeTolerance(Real input_)
{
    m_RelativeTolerance = input_;
}

Real DOTk_SteihaugTointSolver::getRelativeTolerance() const
{
    return (m_RelativeTolerance);
}

void DOTk_SteihaugTointSolver::setRelativeToleranceExponential(Real input_)
{
    m_RelativeToleranceExponential = input_;
}

Real DOTk_SteihaugTointSolver::getRelativeToleranceExponential() const
{
    return (m_RelativeToleranceExponential);
}

void DOTk_SteihaugTointSolver::setStoppingCriterion(dotk::types::solver_stop_criterion_t input_)
{
    m_StoppingCriterion = input_;
}

dotk::types::solver_stop_criterion_t DOTk_SteihaugTointSolver::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

Real DOTk_SteihaugTointSolver::computeSteihaugTointStep(const std::shared_ptr<dotk::Vector<Real> > & newton_step_,
                                                        const std::shared_ptr<dotk::Vector<Real> > & conjugate_dir_,
                                                        const std::shared_ptr<dotk::Vector<Real> > & prec_times_newton_step_,
                                                        const std::shared_ptr<dotk::Vector<Real> > & prec_times_conjugate_dir_)
{
    // Dogleg trust region step
    Real newton_step_dot_prec_times_conjugate_dir = newton_step_->dot(*prec_times_conjugate_dir_);
    Real conjugate_dir_dot_prec_times_conjugate_dir = conjugate_dir_->dot(*prec_times_conjugate_dir_);

    Real newton_step_dot_prec_times_newton_step = newton_step_->dot(*prec_times_newton_step_);

    Real trust_region_radius = this->getTrustRegionRadius();
    Real a = newton_step_dot_prec_times_conjugate_dir * newton_step_dot_prec_times_conjugate_dir;
    Real b = conjugate_dir_dot_prec_times_conjugate_dir
            * (trust_region_radius * trust_region_radius - newton_step_dot_prec_times_newton_step);
    Real value = a + b;
    Real numerator = -newton_step_dot_prec_times_conjugate_dir + std::pow(value, static_cast<Real>(0.5));
    Real step = numerator / conjugate_dir_dot_prec_times_conjugate_dir;

    return (step);
}

bool DOTk_SteihaugTointSolver::invalidCurvatureDetected(const Real & curvature_)
{
    bool curvature_metric_violated = false;

    if(curvature_ < static_cast<Real>(0.))
    {
        this->setStoppingCriterion(dotk::types::NEGATIVE_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }
    else if(std::abs(curvature_) <= std::numeric_limits<Real>::min())
    {
        this->setStoppingCriterion(dotk::types::ZERO_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }
    else if(std::isinf(curvature_))
    {
        this->setStoppingCriterion(dotk::types::INF_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }
    else if(std::isnan(curvature_))
    {
        this->setStoppingCriterion(dotk::types::NaN_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }

    return (curvature_metric_violated);
}

bool DOTk_SteihaugTointSolver::toleranceSatisfied(const Real & norm_descent_direction_)
{
    this->setResidualNorm(norm_descent_direction_);
    Real stopping_tolerance = this->getSolverTolerance();

    bool tolerance_criterion_satisfied = false;
    if(norm_descent_direction_ < stopping_tolerance)
    {
        this->setStoppingCriterion(dotk::types::SOLVER_TOLERANCE_SATISFIED);
        tolerance_criterion_satisfied = true;
    }
    else if(std::isinf(norm_descent_direction_))
    {
        this->setStoppingCriterion(dotk::types::INF_RESIDUAL_NORM);
        tolerance_criterion_satisfied = true;
    }
    else if(std::isnan(norm_descent_direction_))
    {
        this->setStoppingCriterion(dotk::types::NaN_RESIDUAL_NORM);
        tolerance_criterion_satisfied = true;
    }

    return (tolerance_criterion_satisfied);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_SteihaugTointSolver::getActiveSet() const
{
    std::perror("\n**** Error in DOTk_SteihaugTointSolver::getActiveSet. Parent class function not defined. ABORT. ****\n");
    std::abort();
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_SteihaugTointSolver::getInactiveSet() const
{
    std::perror("\n**** Error in DOTk_SteihaugTointSolver::getInactiveSet. Parent class function not defined. ABORT. ****\n");
    std::abort();
}

}
