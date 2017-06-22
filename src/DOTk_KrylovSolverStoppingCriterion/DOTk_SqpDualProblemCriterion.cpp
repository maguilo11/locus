/*
 * DOTk_SqpDualProblemCriterion.cpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DirectSolver.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_RightPreconditioner.hpp"
#include "DOTk_OrthogonalProjection.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_SqpDualProblemCriterion.hpp"

namespace dotk
{

DOTk_SqpDualProblemCriterion::DOTk_SqpDualProblemCriterion() :
        dotk::DOTk_KrylovSolverStoppingCriterion(dotk::types::SQP_DUAL_PROBLEM_CRITERION),
        mStoppingTolerance(0.)
{
    this->initialize();
}

DOTk_SqpDualProblemCriterion::~DOTk_SqpDualProblemCriterion()
{
}

Real DOTk_SqpDualProblemCriterion::evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                                            const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    size_t num_solver_itrs_done = solver_->getNumSolverItrDone();
    if(num_solver_itrs_done == 0)
    {
        mStoppingTolerance = this->computeStoppingTolerance();
        return (mStoppingTolerance);
    }
    else
    {
        return (mStoppingTolerance);
    }
}

void DOTk_SqpDualProblemCriterion::setDualTolerance(Real tolerance_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::DUAL_TOLERANCE, tolerance_);
}

Real DOTk_SqpDualProblemCriterion::getDualTolerance() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::DUAL_TOLERANCE));
}

void DOTk_SqpDualProblemCriterion::setDualDotGradientTolerance(Real tolerance_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::DUAL_DOT_GRAD_TOLERANCE, tolerance_);
}

Real DOTk_SqpDualProblemCriterion::getDualDotGradientTolerance() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::DUAL_DOT_GRAD_TOLERANCE));
}

Real DOTk_SqpDualProblemCriterion::computeStoppingTolerance()
{
    Real dual_tol = this->getDualTolerance();
    Real dual_dot_gradient_tol = this->getDualDotGradientTolerance();
    Real norm_gradient = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::NORM_GRADIENT);
    Real stopping_tolerance = std::min(dual_dot_gradient_tol, dual_tol * norm_gradient);

    return (stopping_tolerance);
}

void DOTk_SqpDualProblemCriterion::initialize()
{
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::NORM_GRADIENT);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::DUAL_TOLERANCE, 1e-4);
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::DUAL_DOT_GRAD_TOLERANCE, 1e4);
}

}
