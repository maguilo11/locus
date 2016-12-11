/*
 * DOTk_KelleySachsStepMng.cpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_Preconditioner.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_KelleySachsStepMng.hpp"
#include "DOTk_SteihaugTointSolver.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SteihaugTointNewtonIO.hpp"

namespace dotk
{

DOTk_KelleySachsStepMng::DOTk_KelleySachsStepMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                 const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        m_Eta(0),
        m_Epsilon(0),
        m_StationarityMeasure(0),
        m_NormInactiveGradient(0),
        m_MidObjectiveFunctionValue(0),
        m_TrustRegionRadiusFlag(false),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(new dotk::DOTk_Preconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_BoundConstraint(new dotk::DOTk_BoundConstraints),
        m_MidPrimal(primal_->control()->clone()),
        m_LowerBound(primal_->control()->clone()),
        m_UpperBound(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_LowerBoundLimit(primal_->control()->clone()),
        m_UpperBoundLimit(primal_->control()->clone()),
        m_InactiveGradient(primal_->control()->clone()),
        m_ProjectedTrialStep(primal_->control()->clone()),
        m_ProjectedCauchyStep(primal_->control()->clone()),
        m_ActiveProjectedTrialStep(primal_->control()->clone()),
        m_InactiveProjectedTrialStep(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_KelleySachsStepMng::DOTk_KelleySachsStepMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                 const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                 const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_) :
        m_Eta(0),
        m_Epsilon(0),
        m_StationarityMeasure(0),
        m_NormInactiveGradient(0),
        m_MidObjectiveFunctionValue(0),
        m_TrustRegionRadiusFlag(false),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(preconditioner_),
        m_BoundConstraint(new dotk::DOTk_BoundConstraints),
        m_MidPrimal(primal_->control()->clone()),
        m_LowerBound(primal_->control()->clone()),
        m_UpperBound(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_LowerBoundLimit(primal_->control()->clone()),
        m_UpperBoundLimit(primal_->control()->clone()),
        m_InactiveGradient(primal_->control()->clone()),
        m_ProjectedTrialStep(primal_->control()->clone()),
        m_ProjectedCauchyStep(primal_->control()->clone()),
        m_ActiveProjectedTrialStep(primal_->control()->clone()),
        m_InactiveProjectedTrialStep(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_KelleySachsStepMng::~DOTk_KelleySachsStepMng()
{
}

Real DOTk_KelleySachsStepMng::getEta() const
{
    // Adaptive constants eta to ensure superlinear convergence
    return (m_Eta);
}

void DOTk_KelleySachsStepMng::setEta(Real input_)
{
    // Adaptive constants eta to ensure superlinear convergence
    m_Eta = input_;
}

Real DOTk_KelleySachsStepMng::getEpsilon() const
{
    // Adaptive constants epsilon to ensure superlinear convergence
    return (m_Epsilon);
}

void DOTk_KelleySachsStepMng::setEpsilon(Real input_)
{
    // Adaptive constants epsilon to ensure superlinear convergence
    m_Epsilon = input_;
}

Real DOTk_KelleySachsStepMng::getStationarityMeasure() const
{
    return (m_StationarityMeasure);
}

Real DOTk_KelleySachsStepMng::getMidObejectiveFunctionValue() const
{
    return (m_MidObjectiveFunctionValue);
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_KelleySachsStepMng::getMidPrimal() const
{
    return (m_MidPrimal);
}

void DOTk_KelleySachsStepMng::setNumOptimizationItrDone(const size_t & input_)
{
    m_LinearOperator->setNumOtimizationItrDone(input_);
}

void DOTk_KelleySachsStepMng::solveSubProblem(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                              const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                              const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_)
{
    m_TrustRegionRadiusFlag = false;
    this->setNumTrustRegionSubProblemItrDone(1);
    Real min_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getMinTrustRegionRadius();
    if(dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius() < min_trust_region_radius)
    {
        dotk::DOTk_TrustRegionStepMng::setTrustRegionRadius(min_trust_region_radius);
    }

    size_t max_num_itr = this->getMaxNumTrustRegionSubProblemItr();
    while(this->getNumTrustRegionSubProblemItrDone() <= max_num_itr)
    {
        // Compute active and inactive sets
        this->computeActiveAndInactiveSet(mng_, solver_);

        // Set solver tolerance
        m_InactiveGradient->update(1., *mng_->getNewGradient(), 0.);
        m_InactiveGradient->elementWiseMultiplication(*solver_->getInactiveSet());
        m_NormInactiveGradient = m_InactiveGradient->norm();
        Real solver_stopping_tolerance = this->getEta() * m_NormInactiveGradient;

        // Compute descent direction
        solver_->setSolverTolerance(solver_stopping_tolerance);
        solver_->setTrustRegionRadius(dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius());
        solver_->solve(m_Preconditioner, m_LinearOperator, mng_);

        // Project trial control
        m_MidPrimal->update(1., *mng_->getNewPrimal(), 0.);
        m_MidPrimal->update(static_cast<Real>(1.), *mng_->getTrialStep(), 1.);
        m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_MidPrimal);
        // Compute projected trial step
        m_ProjectedTrialStep->update(1., *m_MidPrimal, 0.);
        m_ProjectedTrialStep->update(static_cast<Real>(-1), *mng_->getNewPrimal(), 1.);

        // Compute predicted reduction based on mid trial control
        this->applyProjectedTrialStepToHessian(mng_, solver_);
        Real proj_trial_step_dot_inactive_gradient = m_ProjectedTrialStep->dot(*m_InactiveGradient);
        Real proj_trial_step_dot_hess_times_proj_trial_step = m_ProjectedTrialStep->dot(*mng_->getMatrixTimesVector());
        Real predicted_reduction = proj_trial_step_dot_inactive_gradient
                + (static_cast<Real>(0.5) * proj_trial_step_dot_hess_times_proj_trial_step);
        dotk::DOTk_TrustRegionStepMng::setPredictedReduction(predicted_reduction);
        // Update adaptive objective function inexactness tolerance
        dotk::DOTk_TrustRegionStepMng::updateAdaptiveObjectiveInexactnessTolerance();
        // Evaluate current mid objective function
        m_MidObjectiveFunctionValue = mng_->evaluateObjective(m_MidPrimal);

        // Compute actual reduction based on mid trial control
        Real actual_reduction = m_MidObjectiveFunctionValue - mng_->getNewObjectiveFunctionValue();
        dotk::DOTk_TrustRegionStepMng::setActualReduction(actual_reduction);

        // Compute actual over predicted reduction ratio
        Real actual_over_pred_red = actual_reduction /
                (predicted_reduction + std::numeric_limits<Real>::epsilon());
        dotk::DOTk_TrustRegionStepMng::setActualOverPredictedReduction(actual_over_pred_red);

        // Update trust region radius
        io_->printTrustRegionSubProblemDiagnostics(mng_, solver_, this);
        if(this->updateTrustRegionRadius(mng_) == true)
        {
            break;
        }
        dotk::DOTk_TrustRegionStepMng::updateNumTrustRegionSubProblemItrDone();
    }
    m_LinearOperator->updateLimitedMemoryStorage(true);
}

void DOTk_KelleySachsStepMng::bounds(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    bool control_bounds_active = (primal_->getControlLowerBound().use_count() > 0)
            && (primal_->getControlUpperBound().use_count() > 0);
    if(control_bounds_active == false)
    {
        std::perror("\n**** Error in DOTk_KelleySachsStepMng::bounds. User did not define bound constraints. ABORT. ****\n");
        std::abort();
    }

    m_LowerBound->update(1., *primal_->getControlLowerBound(), 0.);
    m_UpperBound->update(1., *primal_->getControlUpperBound(), 0.);
}

void DOTk_KelleySachsStepMng::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->control().use_count() > 0)
    {
        this->bounds(primal_);
    }
    else
    {
        std::perror("\n**** Error in DOTk_KelleySachsStepMng::initialize. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

bool DOTk_KelleySachsStepMng::updateTrustRegionRadius(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real actual_reduction = dotk::DOTk_TrustRegionStepMng::getActualReduction();
    Real actual_over_pred_red = dotk::DOTk_TrustRegionStepMng::getActualOverPredictedReduction();
    Real current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius();
    Real actual_over_pred_mid_bound = dotk::DOTk_TrustRegionStepMng::getActualOverPredictedReductionMidBound();
    Real actual_over_pred_lower_bound = dotk::DOTk_TrustRegionStepMng::getActualOverPredictedReductionLowerBound();
    Real actual_over_pred_upper_bound = dotk::DOTk_TrustRegionStepMng::getActualOverPredictedReductionUpperBound();

    bool stop_trust_region_sub_problem = false;
    Real actual_reduction_lower_bound = this->computeActualReductionLowerBound(mng_);
    if(actual_reduction >= actual_reduction_lower_bound)
    {
        current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionReduction()
                * current_trust_region_radius;
        m_TrustRegionRadiusFlag = true;
    }
    else if(actual_over_pred_red < actual_over_pred_lower_bound)
    {
        current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionReduction()
                * current_trust_region_radius;
        m_TrustRegionRadiusFlag = true;
    }
    else if(actual_over_pred_red >= actual_over_pred_lower_bound
            && actual_over_pred_red < actual_over_pred_mid_bound)
    {
        stop_trust_region_sub_problem = true;
    }
    else if(actual_over_pred_red >= actual_over_pred_mid_bound
            && actual_over_pred_red < actual_over_pred_upper_bound)
    {
        current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionExpansion()
                * current_trust_region_radius;
        stop_trust_region_sub_problem = true;
    }
    else if(actual_over_pred_red > actual_over_pred_upper_bound && m_TrustRegionRadiusFlag == true)
    {
        current_trust_region_radius = static_cast<Real>(2.)
                * dotk::DOTk_TrustRegionStepMng::getTrustRegionExpansion() * current_trust_region_radius;
        stop_trust_region_sub_problem = true;
    }
    else
    {
        Real max_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getMaxTrustRegionRadius();
        current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionExpansion()
                * current_trust_region_radius;
        current_trust_region_radius = std::min(max_trust_region_radius, current_trust_region_radius);
    }
    dotk::DOTk_TrustRegionStepMng::setTrustRegionRadius(current_trust_region_radius);

    return (stop_trust_region_sub_problem);
}

void DOTk_KelleySachsStepMng::applyProjectedTrialStepToHessian(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                               const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_)
{
    // Compute active and inactive projected trial step
    m_ActiveProjectedTrialStep->update(1., *m_ProjectedTrialStep, 0.);
    m_ActiveProjectedTrialStep->elementWiseMultiplication(*solver_->getActiveSet());
    m_InactiveProjectedTrialStep->update(1., *m_ProjectedTrialStep, 0.);
    m_InactiveProjectedTrialStep->elementWiseMultiplication(*solver_->getInactiveSet());

    // Apply inactive projected trial step to Hessian
    mng_->getMatrixTimesVector()->fill(0);
    m_LinearOperator->apply(mng_, m_InactiveProjectedTrialStep, mng_->getMatrixTimesVector());

    // Compute Hessian times projected trial step, i.e. ( ActiveSet + (InactiveSet' * Hess * InactiveSet) ) * vector
    mng_->getMatrixTimesVector()->elementWiseMultiplication(*solver_->getInactiveSet());
    mng_->getMatrixTimesVector()->update(static_cast<Real>(1.), *m_ActiveProjectedTrialStep, 1.);
}

Real DOTk_KelleySachsStepMng::computeActualReductionLowerBound(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real condition_one = dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius() /
            (m_NormInactiveGradient + std::numeric_limits<Real>::epsilon());
    Real lambda = std::min(condition_one, static_cast<Real>(1.));
    m_WorkVector->update(1., *mng_->getNewPrimal(), 0.);
    m_WorkVector->update(-lambda, *m_InactiveGradient, 1.);
    m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_WorkVector);
    m_ProjectedCauchyStep->update(1., *mng_->getNewPrimal(), 0.);
    m_ProjectedCauchyStep->update(static_cast<Real>(-1.), *m_WorkVector, 1.);

    const Real SLOPE_CONSTANT = 1e-4;
    Real norm_proj_cauchy_step = m_ProjectedCauchyStep->norm();
    Real lower_bound = -this->getStationarityMeasure() * SLOPE_CONSTANT * norm_proj_cauchy_step;

    return (lower_bound);
}

void DOTk_KelleySachsStepMng::computeActiveAndInactiveSet(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                          const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_)
{
    m_WorkVector->update(1., *mng_->getNewGradient(), 0.);
    m_WorkVector->elementWiseMultiplication(*solver_->getInactiveSet());

    Real condition = 0;
    Real norm_current_proj_gradient = m_WorkVector->norm();
    Real current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius();
    if(norm_current_proj_gradient > 0)
    {
        condition = current_trust_region_radius / norm_current_proj_gradient;
    }
    else
    {
        condition = current_trust_region_radius / mng_->getNormNewGradient();
    }
    Real lambda = std::min(condition, static_cast<Real>(1.));
    // Compute current lower bound limit
    m_WorkVector->fill(this->getEpsilon());
    m_LowerBoundLimit->update(1., *m_LowerBound, 0.);
    m_LowerBoundLimit->update(static_cast<Real>(-1.), *m_WorkVector, 1.);
    // Compute current upper bound limit
    m_UpperBoundLimit->update(1., *m_UpperBound, 0.);
    m_UpperBoundLimit->update(static_cast<Real>(1.), *m_WorkVector, 1.);
    // Compute active and inactive sets
    m_WorkVector->update(1., *mng_->getNewPrimal(), 0.);
    m_WorkVector->update(-lambda, *mng_->getNewGradient(), 1.);
    m_BoundConstraint->computeActiveAndInactiveSets(*m_WorkVector,
                                                    *m_LowerBoundLimit,
                                                    *m_UpperBoundLimit,
                                                    *solver_->getActiveSet(),
                                                    *solver_->getInactiveSet());
}

}
