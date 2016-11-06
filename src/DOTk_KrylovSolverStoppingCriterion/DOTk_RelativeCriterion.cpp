/*
 * DOTk_RelativeCriterion.cpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_RelativeCriterion.hpp"

namespace dotk
{

DOTk_RelativeCriterion::DOTk_RelativeCriterion(Real relative_tolerance_) :
        dotk::DOTk_KrylovSolverStoppingCriterion(dotk::types::RELATIVE_CRITERION),
        m_StoppingTolerance(0)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::RELATIVE_TOLERANCE, relative_tolerance_);
}

DOTk_RelativeCriterion::~DOTk_RelativeCriterion()
{
}

Real DOTk_RelativeCriterion::evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                                      const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_)
{
    if(solver_->getNumSolverItrDone() == 0)
    {
        Real initial_residual_norm = solver_->getSolverResidualNorm();
        m_StoppingTolerance = this->computeStoppingTolerance(initial_residual_norm);
        return (m_StoppingTolerance);
    }
    else
    {
        return (m_StoppingTolerance);
    }
}

Real DOTk_RelativeCriterion::computeStoppingTolerance(Real intial_norm_residual_)
{
    Real relative_tolerance = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::RELATIVE_TOLERANCE);
    Real stopping_tolerance = std::min(intial_norm_residual_ * intial_norm_residual_,
                                       relative_tolerance * intial_norm_residual_);
    if(stopping_tolerance < std::numeric_limits<Real>::epsilon())
    {
        stopping_tolerance = std::numeric_limits<Real>::epsilon();
    }

    return (stopping_tolerance);
}

}
