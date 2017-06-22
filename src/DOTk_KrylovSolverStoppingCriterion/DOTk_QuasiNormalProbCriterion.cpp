/*
 * DOTk_QuasiNormalProbCriterion.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"
#include "DOTk_QuasiNormalProbCriterion.hpp"

namespace dotk
{

DOTk_QuasiNormalProbCriterion::DOTk_QuasiNormalProbCriterion::DOTk_QuasiNormalProbCriterion() :
        dotk::DOTk_KrylovSolverStoppingCriterion(dotk::types::QUASI_NORMAL_PROBLEM_CRITERION)
{
    this->initialize();
}

DOTk_QuasiNormalProbCriterion::~DOTk_QuasiNormalProbCriterion()
{
}

Real DOTk_QuasiNormalProbCriterion::getStoppingTolerance() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::QUASI_NORMAL_STOPPING_TOL));
}

void DOTk_QuasiNormalProbCriterion::setStoppingTolerance(Real tolerance_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::QUASI_NORMAL_STOPPING_TOL, tolerance_);
}

Real DOTk_QuasiNormalProbCriterion::getRelativeTolerance() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::RELATIVE_TOLERANCE));
}

void DOTk_QuasiNormalProbCriterion::setRelativeTolerance(Real tolerance_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::RELATIVE_TOLERANCE, tolerance_);
}

Real DOTk_QuasiNormalProbCriterion::getTrustRegionRadiusPenaltyParameter() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::TRUST_REGION_RADIUS_PENALTY));
}

void DOTk_QuasiNormalProbCriterion::setTrustRegionRadiusPenaltyParameter(Real penalty_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::TRUST_REGION_RADIUS_PENALTY, penalty_);
}

Real DOTk_QuasiNormalProbCriterion::evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                                             const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    Real stopping_tolerance = this->getStoppingTolerance();
    return (stopping_tolerance);
}

void DOTk_QuasiNormalProbCriterion::initialize()
{
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::RELATIVE_TOLERANCE, 1e-4);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::QUASI_NORMAL_STOPPING_TOL, 1e-8);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::TRUST_REGION_RADIUS_PENALTY, 0.8);
}

}
