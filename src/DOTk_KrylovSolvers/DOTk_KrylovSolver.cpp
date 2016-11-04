/*
 * DOTk_KrylovSolver.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "vector.hpp"
#include "DOTk_KrylovSolver.hpp"

namespace dotk
{

DOTk_KrylovSolver::DOTk_KrylovSolver(dotk::types::krylov_solver_t type_) :
        m_NumSolverItrDone(0),
        m_InvalidCurvatureDetected(false),
        m_TrustRegionRadiusViolation(false),
        m_TrustRegionRadius(std::numeric_limits<Real>::max()),
        m_SolverResidualNorm(0.),
        m_SolverStoppingTolerance(0.),
        m_SolverType(type_),
        m_SolverStopCriterion(dotk::types::SOLVER_DID_NOT_CONVERGED)
{
}

DOTk_KrylovSolver::~DOTk_KrylovSolver()
{
}

void DOTk_KrylovSolver::setNumSolverItrDone(size_t itr_)
{
    m_NumSolverItrDone = itr_;
}

size_t DOTk_KrylovSolver::getNumSolverItrDone() const
{
    return (m_NumSolverItrDone);
}

void DOTk_KrylovSolver::trustRegionViolation(bool trust_region_violation_)
{
    m_TrustRegionRadiusViolation = trust_region_violation_;
}

bool DOTk_KrylovSolver::trustRegionViolationDetected() const
{
    return (m_TrustRegionRadiusViolation);
}

void DOTk_KrylovSolver::invalidCurvatureDetected(bool invalid_curvature_)
{
    m_InvalidCurvatureDetected = invalid_curvature_;
}

bool DOTk_KrylovSolver::invalidCurvatureWasDetected() const
{
    return (m_InvalidCurvatureDetected);
}

void DOTk_KrylovSolver::setTrustRegionRadius(Real trust_region_radius_)
{
    m_TrustRegionRadius = trust_region_radius_;
}

Real DOTk_KrylovSolver::getTrustRegionRadius() const
{
    return (m_TrustRegionRadius);
}

void DOTk_KrylovSolver::setSolverResidualNorm(Real norm_)
{
    m_SolverResidualNorm = norm_;
}

Real DOTk_KrylovSolver::getSolverResidualNorm() const
{
    return (m_SolverResidualNorm);
}

void DOTk_KrylovSolver::setInitialStoppingTolerance(Real tol_)
{
    m_SolverStoppingTolerance = tol_;
}

Real DOTk_KrylovSolver::getInitialStoppingTolerance() const
{
    return (m_SolverStoppingTolerance);
}

void DOTk_KrylovSolver::setSolverType(dotk::types::krylov_solver_t type_)
{
    m_SolverType = type_;
}

dotk::types::krylov_solver_t DOTk_KrylovSolver::getSolverType() const
{
    return (m_SolverType);
}

void DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::solver_stop_criterion_t flag_)
{
    m_SolverStopCriterion = flag_;
}

dotk::types::solver_stop_criterion_t DOTk_KrylovSolver::getSolverStopCriterion() const
{
    return (m_SolverStopCriterion);
}

bool DOTk_KrylovSolver::checkCurvature(Real curvature_)
{
    this->invalidCurvatureDetected(false);
    if(curvature_ < std::numeric_limits<Real>::min())
    {
        this->setSolverStopCriterion(dotk::types::NEGATIVE_CURVATURE_DETECTED);
        this->invalidCurvatureDetected(true);
    }
    else if(std::abs(curvature_) <= std::numeric_limits<Real>::min())
    {
        this->setSolverStopCriterion(dotk::types::ZERO_CURVATURE_DETECTED);
        this->invalidCurvatureDetected(true);
    }
    else if(std::isinf(curvature_))
    {
        this->setSolverStopCriterion(dotk::types::INF_CURVATURE_DETECTED);
        this->invalidCurvatureDetected(true);
    }
    else if(std::isnan(curvature_))
    {
        this->setSolverStopCriterion(dotk::types::NaN_CURVATURE_DETECTED);
        this->invalidCurvatureDetected(true);
    }
    return (this->invalidCurvatureWasDetected());
}

bool DOTk_KrylovSolver::checkResidualNorm(Real norm_, Real stopping_tolerance_)
{
    bool residual_norm_criterion_met = false;
    if(norm_ < stopping_tolerance_)
    {
        this->setSolverStopCriterion(dotk::types::SOLVER_TOLERANCE_SATISFIED);
        residual_norm_criterion_met = true;
    }
    else if(std::isinf(norm_))
    {
        this->setSolverStopCriterion(dotk::types::INF_RESIDUAL_NORM);
        residual_norm_criterion_met = true;
    }
    else if(std::isnan(norm_))
    {
        this->setSolverStopCriterion(dotk::types::NaN_RESIDUAL_NORM);
        residual_norm_criterion_met = true;
    }
    return (residual_norm_criterion_met);
}

}
