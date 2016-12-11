/*
 * DOTk_TangentialSubProblemCriterion.cpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_TangentialSubProblemCriterion.hpp"

namespace dotk
{

DOTk_TangentialSubProblemCriterion::DOTk_TangentialSubProblemCriterion(Real projected_gradient_tolerance_) :
        dotk::DOTk_KrylovSolverStoppingCriterion(dotk::types::TANGENTIAL_SUBPROBLEM_CRITERION)
{
    this->initialize(projected_gradient_tolerance_);
}

DOTk_TangentialSubProblemCriterion::~DOTk_TangentialSubProblemCriterion()
{
}

Real DOTk_TangentialSubProblemCriterion::evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    Real stopping_tolerance = 0.;
    size_t current_krylov_itr = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::CURRENT_KRYLOV_SOLVER_ITR);

    if(current_krylov_itr == 0)
    {
        Real trust_region_radius = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::TRUST_REGION_RADIUS);
        stopping_tolerance = this->computeInitialStoppingTolerance(trust_region_radius, kernel_vector_);
    }
    else
    {
        Real norm_residual = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::NORM_RESIDUAL);
        stopping_tolerance = this->computeStoppingTolerance(norm_residual, kernel_vector_);
    }

    return (stopping_tolerance);
}

Real DOTk_TangentialSubProblemCriterion::computeStoppingTolerance(Real norm_residual_,
                                                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    Real norm_kernel_vector = kernel_vector_->norm();
    Real projected_gradient_tol =
            dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::PROJECTED_GRADIENT_TOLERANCE);
    Real stopping_tolerance = projected_gradient_tol * std::min(norm_kernel_vector, norm_residual_);

    if(stopping_tolerance < std::numeric_limits<Real>::epsilon())
    {
        stopping_tolerance = std::numeric_limits<Real>::epsilon();
    }

    return (stopping_tolerance);
}

Real DOTk_TangentialSubProblemCriterion::computeInitialStoppingTolerance(Real trust_region_radius_,
                                                                         const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    Real norm_kernel_vector = kernel_vector_->norm();
    Real projected_gradient_tol =
            dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::PROJECTED_GRADIENT_TOLERANCE);
    Real norm_gradient = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::NORM_GRADIENT);
    Real stopping_tolerance = projected_gradient_tol
            * std::min(std::min(norm_kernel_vector, trust_region_radius_), norm_gradient);

    if(stopping_tolerance < std::numeric_limits<Real>::epsilon())
    {
        stopping_tolerance = std::numeric_limits<Real>::epsilon();
    }

    return (stopping_tolerance);
}

void DOTk_TangentialSubProblemCriterion::initialize(Real value_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::NORM_GRADIENT);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::NORM_RESIDUAL);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::TRUST_REGION_RADIUS);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::CURRENT_KRYLOV_SOLVER_ITR);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::PROJECTED_GRADIENT_TOLERANCE);
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::PROJECTED_GRADIENT_TOLERANCE, value_);
}

}
