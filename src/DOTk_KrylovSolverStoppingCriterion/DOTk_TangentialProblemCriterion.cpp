/*
 * DOTk_TangentialProblemCriterion.cpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <algorithm>

#include "vector.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_TangentialProblemCriterion.hpp"

namespace dotk
{

DOTk_TangentialProblemCriterion::DOTk_TangentialProblemCriterion(const std::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_KrylovSolverStoppingCriterion(dotk::types::TANGENTIAL_PROBLEM_CRITERION),
        m_StoppingCriterionOptions(4, 0.),
        m_WorkVector(vector_->clone())
{
    this->initialize();
}

DOTk_TangentialProblemCriterion::~DOTk_TangentialProblemCriterion()
{
}

void DOTk_TangentialProblemCriterion::setTangentialTolerance(Real tolerance_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::TANGENTIAL_TOLERANCE, tolerance_);
}

Real DOTk_TangentialProblemCriterion::getTangentialTolerance() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::TANGENTIAL_TOLERANCE));
}

void DOTk_TangentialProblemCriterion::setTangentialToleranceContractionFactor(Real factor_)
{
    dotk::DOTk_KrylovSolverStoppingCriterion::set(dotk::types::TANGENTIAL_TOL_CONTRACTION_FACTOR, factor_);
}

Real DOTk_TangentialProblemCriterion::getTangentialToleranceContractionFactor() const
{
    return (dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::TANGENTIAL_TOL_CONTRACTION_FACTOR));
}

void DOTk_TangentialProblemCriterion::setCurrentTrialStep(const std::shared_ptr<dotk::Vector<Real> > & trial_step_)
{
    m_WorkVector->update(1., *trial_step_, 0.);
}

Real DOTk_TangentialProblemCriterion::evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                                               const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    Real stopping_tolerance = this->computeStoppingTolerance(kernel_vector_);
    return (stopping_tolerance);
}

void DOTk_TangentialProblemCriterion::initialize()
{
    this->insert(dotk::types::TANGENTIAL_TOLERANCE, 1e-4);
    this->insert(dotk::types::TANGENTIAL_TOL_CONTRACTION_FACTOR, 1e-3);
    this->insert(dotk::types::TRUST_REGION_RADIUS, std::numeric_limits<Real>::max());
    this->insert(dotk::types::NORM_TANGENTIAL_STEP_RESIDUAL);
    this->insert(dotk::types::NORM_PROJECTED_TANGENTIAL_STEP);
}

Real DOTk_TangentialProblemCriterion::computeStoppingTolerance(const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_)
{
    m_WorkVector->update(static_cast<Real>(1.0), *kernel_vector_, 1.);

    Real trust_region_radius = dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::TRUST_REGION_RADIUS);
    m_StoppingCriterionOptions[0] = trust_region_radius;
    m_StoppingCriterionOptions[1] = m_WorkVector->norm();

    Real tangential_tolerance_ = this->getTangentialTolerance();
    Real norm_projected_tangential_step =
            dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::NORM_PROJECTED_TANGENTIAL_STEP);
    m_StoppingCriterionOptions[2] = (tangential_tolerance_ * norm_projected_tangential_step) / trust_region_radius;

    Real norm_tangential_problem_residual =
            dotk::DOTk_KrylovSolverStoppingCriterion::get(dotk::types::NORM_TANGENTIAL_STEP_RESIDUAL);
    m_StoppingCriterionOptions[3] = (tangential_tolerance_ * norm_tangential_problem_residual) / trust_region_radius;

    Real minimum_value =
            std::min_element(m_StoppingCriterionOptions.begin(), m_StoppingCriterionOptions.end()).operator *();
    Real stopping_tolerance = trust_region_radius * minimum_value;

    if(stopping_tolerance < std::numeric_limits<Real>::epsilon())
    {
        stopping_tolerance = std::numeric_limits<Real>::epsilon();
    }

    return (stopping_tolerance);
}

}
