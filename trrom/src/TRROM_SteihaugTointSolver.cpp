/*
 * TRROM_SteihaugTointSolver.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>
#include <math.h>
#include "TRROM_Vector.hpp"
#include "TRROM_SteihaugTointSolver.hpp"

namespace trrom
{

SteihaugTointSolver::SteihaugTointSolver() :
        m_MaxNumItr(200),
        m_NumItrDone(0),
        m_Tolerance(1e-8),
        m_ResidualNorm(0),
        m_TrustRegionRadius(0),
        m_RelativeTolerance(1e-1),
        m_RelativeToleranceExponential(0.5),
        m_StoppingCriterion(trrom::types::SOLVER_DID_NOT_CONVERGED)
{
}

SteihaugTointSolver::~SteihaugTointSolver()
{
}

void SteihaugTointSolver::setMaxNumItr(int input_)
{
    m_MaxNumItr = input_;
}

int SteihaugTointSolver::getMaxNumItr() const
{
    return (m_MaxNumItr);
}

void SteihaugTointSolver::setNumItrDone(int input_)
{
    m_NumItrDone = input_;
}

int SteihaugTointSolver::getNumItrDone() const
{
    return (m_NumItrDone);
}

void SteihaugTointSolver::setSolverTolerance(double input_)
{
    m_Tolerance = input_;
}

double SteihaugTointSolver::getSolverTolerance() const
{
    return (m_Tolerance);
}

void SteihaugTointSolver::setTrustRegionRadius(double input_)
{
    m_TrustRegionRadius = input_;
}

double SteihaugTointSolver::getTrustRegionRadius() const
{
    return (m_TrustRegionRadius);
}

void SteihaugTointSolver::setResidualNorm(double input_)
{
    m_ResidualNorm = input_;
}

double SteihaugTointSolver::getResidualNorm() const
{
    return (m_ResidualNorm);
}

void SteihaugTointSolver::setRelativeTolerance(double input_)
{
    m_RelativeTolerance = input_;
}

double SteihaugTointSolver::getRelativeTolerance() const
{
    return (m_RelativeTolerance);
}

void SteihaugTointSolver::setRelativeToleranceExponential(double input_)
{
    m_RelativeToleranceExponential = input_;
}

double SteihaugTointSolver::getRelativeToleranceExponential() const
{
    return (m_RelativeToleranceExponential);
}

void SteihaugTointSolver::setStoppingCriterion(trrom::types::solver_stop_criterion_t input_)
{
    m_StoppingCriterion = input_;
}

trrom::types::solver_stop_criterion_t SteihaugTointSolver::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

double SteihaugTointSolver::computeSteihaugTointStep(const std::tr1::shared_ptr<trrom::Vector<double> > & newton_step_,
                                                     const std::tr1::shared_ptr<trrom::Vector<double> > & conjugate_dir_,
                                                     const std::tr1::shared_ptr<trrom::Vector<double> > & prec_times_newton_step_,
                                                     const std::tr1::shared_ptr<trrom::Vector<double> > & prec_times_conjugate_dir_)
{
    // Dogleg trust region step
    double newton_step_dot_prec_times_conjugate_dir = newton_step_->dot(*prec_times_conjugate_dir_);
    double conjugate_dir_dot_prec_times_conjugate_dir = conjugate_dir_->dot(*prec_times_conjugate_dir_);

    double newton_step_dot_prec_times_newton_step = newton_step_->dot(*prec_times_newton_step_);

    double trust_region_radius = this->getTrustRegionRadius();
    double a = newton_step_dot_prec_times_conjugate_dir * newton_step_dot_prec_times_conjugate_dir;
    double b = conjugate_dir_dot_prec_times_conjugate_dir
            * (trust_region_radius * trust_region_radius - newton_step_dot_prec_times_newton_step);
    double value = a + b;
    double numerator = -newton_step_dot_prec_times_conjugate_dir + std::pow(value, static_cast<double>(0.5));
    double step = numerator / conjugate_dir_dot_prec_times_conjugate_dir;

    return (step);
}

bool SteihaugTointSolver::invalidCurvatureDetected(const double & curvature_)
{
    bool curvature_metric_violated = false;

    if(curvature_ < static_cast<double>(0.))
    {
        this->setStoppingCriterion(trrom::types::NEGATIVE_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }
    else if(std::abs(curvature_) <= std::numeric_limits<double>::min())
    {
        this->setStoppingCriterion(trrom::types::ZERO_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }
    else if(std::isinf(curvature_))
    {
        this->setStoppingCriterion(trrom::types::INF_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }
    else if(std::isnan(curvature_))
    {
        this->setStoppingCriterion(trrom::types::NaN_CURVATURE_DETECTED);
        curvature_metric_violated = true;
    }

    return (curvature_metric_violated);
}

bool SteihaugTointSolver::toleranceSatisfied(const double & norm_descent_direction_)
{
    this->setResidualNorm(norm_descent_direction_);
    double stopping_tolerance = this->getSolverTolerance();

    bool tolerance_criterion_satisfied = false;
    if(norm_descent_direction_ < stopping_tolerance)
    {
        this->setStoppingCriterion(trrom::types::SOLVER_TOLERANCE_SATISFIED);
        tolerance_criterion_satisfied = true;
    }
    else if(std::isinf(norm_descent_direction_))
    {
        this->setStoppingCriterion(trrom::types::INF_RESIDUAL_NORM);
        tolerance_criterion_satisfied = true;
    }
    else if(std::isnan(norm_descent_direction_))
    {
        this->setStoppingCriterion(trrom::types::NaN_RESIDUAL_NORM);
        tolerance_criterion_satisfied = true;
    }

    return (tolerance_criterion_satisfied);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & SteihaugTointSolver::getActiveSet() const
{
    std::perror("\n**** Error in SteihaugTointSolver::getActiveSet. Parent class function not defined. ABORT. ****\n");
    std::abort();
}

const std::tr1::shared_ptr<trrom::Vector<double> > & SteihaugTointSolver::getInactiveSet() const
{
    std::perror("\n**** Error in SteihaugTointSolver::getInactiveSet. Parent class function not defined. ABORT. ****\n");
    std::abort();
}

}
