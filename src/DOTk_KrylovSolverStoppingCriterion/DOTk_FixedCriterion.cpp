/*
 * DOTk_FixedCriterion.cpp
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
#include "DOTk_FixedCriterion.hpp"

namespace dotk
{

DOTk_FixedCriterion::DOTk_FixedCriterion(Real tolerance_) :
        dotk::DOTk_KrylovSolverStoppingCriterion(dotk::types::FIX_CRITERION)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::FIX_TOLERANCE, tolerance_);
}

DOTk_FixedCriterion::~DOTk_FixedCriterion()
{
}

Real DOTk_FixedCriterion::evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    Real tolerance = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::FIX_TOLERANCE);
    return (tolerance);
}

}
