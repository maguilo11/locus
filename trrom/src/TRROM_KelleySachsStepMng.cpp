/*
 * TRROM_KelleySachsStepMng.cpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_LinearOperator.hpp"
#include "TRROM_Preconditioner.hpp"
#include "TRROM_BoundConstraints.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_SteihaugTointSolver.hpp"
#include "TRROM_OptimizationDataMng.hpp"
#include "TRROM_SteihaugTointNewtonIO.hpp"

namespace trrom
{

KelleySachsStepMng::KelleySachsStepMng(const std::tr1::shared_ptr<trrom::Data> & data_,
                                       const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_) :
        trrom::TrustRegionStepMng(),
        m_Eta(0),
        m_Epsilon(0),
        m_StationarityMeasure(0),
        m_NormInactiveGradient(0),
        m_MidObjectiveFunctionValue(0),
        m_TrustRegionRadiusFlag(false),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(new trrom::Preconditioner(trrom::types::LEFT_PRECONDITIONER_DISABLED)),
        m_BoundConstraint(new trrom::BoundConstraints),
        m_MidPrimal(data_->control()->create()),
        m_LowerBound(data_->control()->create()),
        m_UpperBound(data_->control()->create()),
        m_WorkVector(data_->control()->create()),
        m_LowerBoundLimit(data_->control()->create()),
        m_UpperBoundLimit(data_->control()->create()),
        m_InactiveGradient(data_->control()->create()),
        m_ProjectedTrialStep(data_->control()->create()),
        m_ProjectedCauchyStep(data_->control()->create()),
        m_ActiveProjectedTrialStep(data_->control()->create()),
        m_InactiveProjectedTrialStep(data_->control()->create())
{
    this->initialize(data_);
}

KelleySachsStepMng::KelleySachsStepMng(const std::tr1::shared_ptr<trrom::Data> & data_,
                                       const std::tr1::shared_ptr<trrom::LinearOperator> & linear_operator_,
                                       const std::tr1::shared_ptr<trrom::Preconditioner> & preconditioner_) :
        trrom::TrustRegionStepMng(),
        m_Eta(0),
        m_Epsilon(0),
        m_StationarityMeasure(0),
        m_NormInactiveGradient(0),
        m_MidObjectiveFunctionValue(0),
        m_TrustRegionRadiusFlag(false),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(preconditioner_),
        m_BoundConstraint(new trrom::BoundConstraints),
        m_MidPrimal(data_->control()->create()),
        m_LowerBound(data_->control()->create()),
        m_UpperBound(data_->control()->create()),
        m_WorkVector(data_->control()->create()),
        m_LowerBoundLimit(data_->control()->create()),
        m_UpperBoundLimit(data_->control()->create()),
        m_InactiveGradient(data_->control()->create()),
        m_ProjectedTrialStep(data_->control()->create()),
        m_ProjectedCauchyStep(data_->control()->create()),
        m_ActiveProjectedTrialStep(data_->control()->create()),
        m_InactiveProjectedTrialStep(data_->control()->create())
{
    this->initialize(data_);
}

KelleySachsStepMng::~KelleySachsStepMng()
{
}

double KelleySachsStepMng::getEta() const
{
    // Adaptive constants eta to ensure superlinear convergence
    return (m_Eta);
}

void KelleySachsStepMng::setEta(double input_)
{
    // Adaptive constants eta to ensure superlinear convergence
    m_Eta = input_;
}

double KelleySachsStepMng::getEpsilon() const
{
    // Adaptive constants epsilon to ensure superlinear convergence
    return (m_Epsilon);
}

void KelleySachsStepMng::setEpsilon(double input_)
{
    // Adaptive constants epsilon to ensure superlinear convergence
    m_Epsilon = input_;
}

double KelleySachsStepMng::getStationarityMeasure() const
{
    return (m_StationarityMeasure);
}

double KelleySachsStepMng::getMidObejectiveFunctionValue() const
{
    return (m_MidObjectiveFunctionValue);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & KelleySachsStepMng::getMidPrimal() const
{
    return (m_MidPrimal);
}

bool KelleySachsStepMng::solveSubProblem(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                         const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                         const std::tr1::shared_ptr<trrom::SteihaugTointNewtonIO> & io_)
{
    m_TrustRegionRadiusFlag = false;
    bool trial_control_accepted = true;
    this->setNumTrustRegionSubProblemItrDone(1);
    double min_trust_region_radius = trrom::TrustRegionStepMng::getMinTrustRegionRadius();
    if(trrom::TrustRegionStepMng::getTrustRegionRadius() < min_trust_region_radius)
    {
        trrom::TrustRegionStepMng::setTrustRegionRadius(min_trust_region_radius);
    }

    int max_num_itr = this->getMaxNumTrustRegionSubProblemItr();
    while(this->getNumTrustRegionSubProblemItrDone() <= max_num_itr)
    {
        // Compute active and inactive sets
        this->computeActiveAndInactiveSet(mng_, solver_);

        // Set solver tolerance
        m_InactiveGradient->copy(*mng_->getNewGradient());
        m_InactiveGradient->elementWiseMultiplication(*solver_->getInactiveSet());
        m_NormInactiveGradient = m_InactiveGradient->norm();
        double solver_stopping_tolerance = this->getEta() * m_NormInactiveGradient;

        // Compute descent direction
        solver_->setSolverTolerance(solver_stopping_tolerance);
        solver_->setTrustRegionRadius(trrom::TrustRegionStepMng::getTrustRegionRadius());
        solver_->solve(m_Preconditioner, m_LinearOperator, mng_);

        // Project trial control
        m_MidPrimal->copy(*mng_->getNewPrimal());
        m_MidPrimal->axpy(static_cast<double>(1.), *mng_->getTrialStep());
        m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_MidPrimal);
        // Compute projected trial step
        m_ProjectedTrialStep->copy(*m_MidPrimal);
        m_ProjectedTrialStep->axpy(static_cast<double>(-1), *mng_->getNewPrimal());

        // Compute predicted reduction based on mid trial control
        this->applyProjectedTrialStepToHessian(mng_, solver_);
        double proj_trial_step_dot_inactive_gradient = m_ProjectedTrialStep->dot(*m_InactiveGradient);
        double proj_trial_step_dot_hess_times_proj_trial_step =
                m_ProjectedTrialStep->dot(*mng_->getMatrixTimesVector());
        double predicted_reduction = proj_trial_step_dot_inactive_gradient
                + (static_cast<double>(0.5) * proj_trial_step_dot_hess_times_proj_trial_step);
        trrom::TrustRegionStepMng::setPredictedReduction(predicted_reduction);
        // Update objective function inexactness tolerance (bound)
        trrom::TrustRegionStepMng::updateObjectiveInexactnessTolerance(predicted_reduction);
        double tolerance = trrom::TrustRegionStepMng::getObjectiveInexactnessTolerance();
        mng_->setObjectiveInexactnessTolerance(tolerance);
        if(mng_->isObjectiveInexactnessToleranceExceeded() == true)
        {
            trial_control_accepted = false;
            break;
        }
        // Evaluate current mid objective function
        m_MidObjectiveFunctionValue = mng_->evaluateObjective(m_MidPrimal);
        // Compute actual reduction based on mid trial control
        double actual_reduction = m_MidObjectiveFunctionValue - mng_->getNewObjectiveFunctionValue();
        trrom::TrustRegionStepMng::setActualReduction(actual_reduction);
        // Compute actual over predicted reduction ratio
        double actual_over_pred_red = actual_reduction / (predicted_reduction + std::numeric_limits<double>::epsilon());
        trrom::TrustRegionStepMng::setActualOverPredictedReduction(actual_over_pred_red);
        // Update trust region radius
        io_->printTrustRegionSubProblemDiagnostics(mng_, solver_, this);
        if(this->updateTrustRegionRadius(mng_) == true)
        {
            break;
        }
        trrom::TrustRegionStepMng::updateNumTrustRegionSubProblemItrDone();
    }
    return (trial_control_accepted);
}

void KelleySachsStepMng::bounds(const std::tr1::shared_ptr<trrom::Data> & data_)
{
    bool control_bounds_active = (data_->getControlLowerBound().use_count() > 0)
            && (data_->getControlUpperBound().use_count() > 0);
    if(control_bounds_active == false)
    {
        std::perror("\n**** Error in KelleySachsStepMng::bounds. User did not define bound constraints. ABORT. ****\n");
        std::abort();
    }

    m_LowerBound->copy(*data_->getControlLowerBound());
    m_UpperBound->copy(*data_->getControlUpperBound());
}

void KelleySachsStepMng::initialize(const std::tr1::shared_ptr<trrom::Data> & data_)
{
    if(data_->control().use_count() > 0)
    {
        this->bounds(data_);
    }
    else
    {
        std::perror("\n**** Error in KelleySachsStepMng::initialize. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

bool KelleySachsStepMng::updateTrustRegionRadius(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_)
{
    double actual_reduction = trrom::TrustRegionStepMng::getActualReduction();
    double actual_over_pred_red = trrom::TrustRegionStepMng::getActualOverPredictedReduction();
    double current_trust_region_radius = trrom::TrustRegionStepMng::getTrustRegionRadius();
    double actual_over_pred_mid_bound = trrom::TrustRegionStepMng::getActualOverPredictedReductionMidBound();
    double actual_over_pred_lower_bound = trrom::TrustRegionStepMng::getActualOverPredictedReductionLowerBound();
    double actual_over_pred_upper_bound = trrom::TrustRegionStepMng::getActualOverPredictedReductionUpperBound();

    bool stop_trust_region_sub_problem = false;
    double actual_reduction_lower_bound = this->computeActualReductionLowerBound(mng_);
    if(actual_reduction >= actual_reduction_lower_bound)
    {
        current_trust_region_radius = trrom::TrustRegionStepMng::getTrustRegionReduction()
                * current_trust_region_radius;
        m_TrustRegionRadiusFlag = true;
    }
    else if(actual_over_pred_red < actual_over_pred_lower_bound)
    {
        current_trust_region_radius = trrom::TrustRegionStepMng::getTrustRegionReduction()
                * current_trust_region_radius;
        m_TrustRegionRadiusFlag = true;
    }
    else if(actual_over_pred_red >= actual_over_pred_lower_bound && actual_over_pred_red < actual_over_pred_mid_bound)
    {
        stop_trust_region_sub_problem = true;
    }
    else if(actual_over_pred_red >= actual_over_pred_mid_bound && actual_over_pred_red < actual_over_pred_upper_bound)
    {
        current_trust_region_radius = trrom::TrustRegionStepMng::getTrustRegionExpansion()
                * current_trust_region_radius;
        stop_trust_region_sub_problem = true;
    }
    else if(actual_over_pred_red > actual_over_pred_upper_bound && m_TrustRegionRadiusFlag == true)
    {
        current_trust_region_radius = static_cast<double>(2.) * trrom::TrustRegionStepMng::getTrustRegionExpansion()
                * current_trust_region_radius;
        stop_trust_region_sub_problem = true;
    }
    else
    {
        double max_trust_region_radius = trrom::TrustRegionStepMng::getMaxTrustRegionRadius();
        current_trust_region_radius = trrom::TrustRegionStepMng::getTrustRegionExpansion()
                * current_trust_region_radius;
        current_trust_region_radius = std::min(max_trust_region_radius, current_trust_region_radius);
    }
    trrom::TrustRegionStepMng::setTrustRegionRadius(current_trust_region_radius);

    return (stop_trust_region_sub_problem);
}

void KelleySachsStepMng::applyProjectedTrialStepToHessian(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                                          const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_)
{
    // Compute active and inactive projected trial step
    m_ActiveProjectedTrialStep->copy(*m_ProjectedTrialStep);
    m_ActiveProjectedTrialStep->elementWiseMultiplication(*solver_->getActiveSet());
    m_InactiveProjectedTrialStep->copy(*m_ProjectedTrialStep);
    m_InactiveProjectedTrialStep->elementWiseMultiplication(*solver_->getInactiveSet());

    // Apply inactive projected trial step to Hessian
    mng_->getMatrixTimesVector()->fill(0);
    m_LinearOperator->apply(mng_, m_InactiveProjectedTrialStep, mng_->getMatrixTimesVector());

    // Compute Hessian times projected trial step, i.e. ( ActiveSet + (InactiveSet' * Hess * InactiveSet) ) * Vector
    mng_->getMatrixTimesVector()->elementWiseMultiplication(*solver_->getInactiveSet());
    mng_->getMatrixTimesVector()->axpy(static_cast<double>(1.), *m_ActiveProjectedTrialStep);
}

double KelleySachsStepMng::computeActualReductionLowerBound(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_)
{
    double condition_one = trrom::TrustRegionStepMng::getTrustRegionRadius()
            / (m_NormInactiveGradient + std::numeric_limits<double>::epsilon());
    double lambda = std::min(condition_one, static_cast<double>(1.));
    m_WorkVector->copy(*mng_->getNewPrimal());
    m_WorkVector->axpy(-lambda, *m_InactiveGradient);
    m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_WorkVector);
    m_ProjectedCauchyStep->copy(*mng_->getNewPrimal());
    m_ProjectedCauchyStep->axpy(static_cast<double>(-1.), *m_WorkVector);

    const double SLOPE_CONSTANT = 1e-4;
    double norm_proj_cauchy_step = m_ProjectedCauchyStep->norm();
    double lower_bound = -this->getStationarityMeasure() * SLOPE_CONSTANT * norm_proj_cauchy_step;

    return (lower_bound);
}

void KelleySachsStepMng::computeActiveAndInactiveSet(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                                     const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_)
{
    m_WorkVector->copy(*mng_->getNewGradient());
    m_WorkVector->elementWiseMultiplication(*solver_->getInactiveSet());

    double condition = 0;
    double norm_current_proj_gradient = m_WorkVector->norm();
    double current_trust_region_radius = trrom::TrustRegionStepMng::getTrustRegionRadius();
    if(norm_current_proj_gradient > 0)
    {
        condition = current_trust_region_radius / norm_current_proj_gradient;
    }
    else
    {
        condition = current_trust_region_radius / mng_->getNormNewGradient();
    }
    double lambda = std::min(condition, static_cast<double>(1.));
    // Compute current lower bound limit
    m_WorkVector->fill(this->getEpsilon());
    m_LowerBoundLimit->copy(*m_LowerBound);
    m_LowerBoundLimit->axpy(static_cast<double>(-1.), *m_WorkVector);
    // Compute current upper bound limit
    m_UpperBoundLimit->copy(*m_UpperBound);
    m_UpperBoundLimit->axpy(static_cast<double>(1.), *m_WorkVector);
    // Compute active and inactive sets
    m_WorkVector->copy(*mng_->getNewPrimal());
    m_WorkVector->axpy(-lambda, *mng_->getNewGradient());
    m_BoundConstraint->computeActiveAndInactiveSets(*m_WorkVector,
                                                    *m_LowerBoundLimit,
                                                    *m_UpperBoundLimit,
                                                    *solver_->getActiveSet(),
                                                    *solver_->getInactiveSet());
}

}
