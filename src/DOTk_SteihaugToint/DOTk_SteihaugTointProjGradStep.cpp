/*
 * DOTk_SteihaugTointProjGradStep.cpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_Preconditioner.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_SteihaugTointSolver.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SteihaugTointNewtonIO.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_SteihaugTointProjGradStep.hpp"

namespace dotk
{

DOTk_SteihaugTointProjGradStep::DOTk_SteihaugTointProjGradStep
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_TrustRegionStepMng(),
        m_MaxNumProjections(10),
        m_LineSearchContraction(0.5),
        m_ControlUpdateRoutineConstant(1e-2),
        m_PredictedReductionBasedOnCauchyStep(0),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(new dotk::DOTk_Preconditioner(dotk::types::LEFT_PRECONDITIONER_DISABLED)),
        m_BoundConstraint(new dotk::DOTk_BoundConstraints),
        m_ActiveSet(primal_->control()->clone()),
        m_LowerBound(primal_->control()->clone()),
        m_UpperBound(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_CurrentPrimal(primal_->control()->clone()),
        m_ProjectedTrialStep(primal_->control()->clone()),
        m_ProjectedCauchyStep(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_SteihaugTointProjGradStep::DOTk_SteihaugTointProjGradStep
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
 const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_) :
        dotk::DOTk_TrustRegionStepMng(),
        m_MaxNumProjections(10),
        m_LineSearchContraction(0.5),
        m_ControlUpdateRoutineConstant(1e-2),
        m_PredictedReductionBasedOnCauchyStep(0),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(preconditioner_),
        m_BoundConstraint(new dotk::DOTk_BoundConstraints),
        m_ActiveSet(primal_->control()->clone()),
        m_LowerBound(primal_->control()->clone()),
        m_UpperBound(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_CurrentPrimal(primal_->control()->clone()),
        m_ProjectedTrialStep(primal_->control()->clone()),
        m_ProjectedCauchyStep(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_SteihaugTointProjGradStep::~DOTk_SteihaugTointProjGradStep()
{
}

void DOTk_SteihaugTointProjGradStep::setMaxNumProjections(size_t value_)
{
    m_MaxNumProjections = value_;
}

size_t DOTk_SteihaugTointProjGradStep::getMaxNumProjections() const
{
    return (m_MaxNumProjections);
}

void DOTk_SteihaugTointProjGradStep::setLineSearchContraction(Real value_)
{
    m_LineSearchContraction = value_;
}

Real DOTk_SteihaugTointProjGradStep::getLineSearchContraction() const
{
    return (m_LineSearchContraction);
}

void DOTk_SteihaugTointProjGradStep::setControlUpdateRoutineConstant(Real value_)
{
    m_ControlUpdateRoutineConstant = value_;
}

Real DOTk_SteihaugTointProjGradStep::getControlUpdateRoutineConstant() const
{
    return (m_ControlUpdateRoutineConstant);
}

void DOTk_SteihaugTointProjGradStep::setNumOptimizationItrDone(const size_t & itr_)
{
    m_LinearOperator->setNumOtimizationItrDone(itr_);
}

void DOTk_SteihaugTointProjGradStep::solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                     const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                                     const std::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_)
{
    m_ActiveSet->fill(0);
    Real new_objective_value = 0.;
    this->setNumTrustRegionSubProblemItrDone(1);
    m_CurrentPrimal->update(1., *mng_->getNewPrimal(), 0.);
    Real angle_tolerance = this->getMinCosineAngleTolerance();
    Real current_objective_value = mng_->getNewObjectiveFunctionValue();
    Real min_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getMinTrustRegionRadius();
    if(dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius() < min_trust_region_radius)
    {
        dotk::DOTk_TrustRegionStepMng::setTrustRegionRadius(min_trust_region_radius);
    }

    size_t max_num_itr = this->getMaxNumTrustRegionSubProblemItr();
    while(this->getNumTrustRegionSubProblemItrDone() <= max_num_itr)
    {
        this->computeProjectedCauchyStep(mng_);

        Real current_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius();
        solver_->setTrustRegionRadius(current_trust_region_radius);
        solver_->solve(m_Preconditioner, m_LinearOperator, mng_);
        dotk::gtools::checkDescentDirection(mng_->getNewGradient(), mng_->getTrialStep(), angle_tolerance);

        this->updateControl(mng_);

        new_objective_value = mng_->evaluateObjective();
        Real actual_reduction = new_objective_value - current_objective_value;
        dotk::DOTk_TrustRegionStepMng::setActualReduction(actual_reduction);
        Real rho = actual_reduction / dotk::DOTk_TrustRegionStepMng::getPredictedReduction();
        dotk::DOTk_TrustRegionStepMng::setActualOverPredictedReduction(rho);

        io_->printTrustRegionSubProblemDiagnostics(mng_, solver_, this);
        if(dotk::DOTk_TrustRegionStepMng::updateTrustRegionRadius() == true)
        {
            break;
        }
        this->updateNumTrustRegionSubProblemItrDone();
        mng_->getNewPrimal()->update(1., *m_CurrentPrimal, 0.);
    }

    mng_->setOldObjectiveFunctionValue(current_objective_value);
    mng_->setNewObjectiveFunctionValue(new_objective_value);
    this->updateDataManager(mng_);
    m_LinearOperator->updateLimitedMemoryStorage(true);
}

void DOTk_SteihaugTointProjGradStep::initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->control().use_count() > 0)
    {
        this->bounds(primal_);
    }
    else
    {
        std::perror("\n**** Error in DOTk_SteihaugTointProjGradStep::initialize. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

void DOTk_SteihaugTointProjGradStep::bounds(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    bool control_bounds_active = (primal_->getControlLowerBound().use_count() > 0)
            && (primal_->getControlUpperBound().use_count() > 0);
    if(control_bounds_active == false)
    {
        std::perror("\n**** Error in DOTk_SteihaugTointProjGradStep::bounds. User did not define bound constraints. ABORT. ****\n");
        std::abort();
    }

    m_LowerBound->update(1., *primal_->getControlLowerBound(), 0.);
    m_UpperBound->update(1., *primal_->getControlUpperBound(), 0.);
}

void DOTk_SteihaugTointProjGradStep::updateDataManager(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    mng_->getOldPrimal()->update(1., *m_CurrentPrimal, 0.);
    mng_->getOldGradient()->update(1., *mng_->getNewGradient(), 0.);
    mng_->computeGradient();

    m_WorkVector->update(1., *mng_->getNewGradient(), 0.);
    m_BoundConstraint->pruneActive(*m_ActiveSet, *m_WorkVector);
    Real norm_projected_gradient = m_WorkVector->norm();
    mng_->setNormNewGradient(norm_projected_gradient);

    m_WorkVector->update(1., *mng_->getTrialStep(), 0.);
    m_BoundConstraint->pruneActive(*m_ActiveSet, *m_WorkVector);
    Real norm_projected_trial_step = m_WorkVector->norm();
    mng_->setNormTrialStep(norm_projected_trial_step);
}

void DOTk_SteihaugTointProjGradStep::updateControl(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real step = 1.;
    Real predicted_reduction = 0.;
    Real mu0 = this->getControlUpdateRoutineConstant();
    Real step_contraction = this->getLineSearchContraction();
    Real scaled_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionRadiusScaling()
            * dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius();

    size_t itr = 1;
    size_t max_num_projection = this->getMaxNumProjections();
    while(1)
    {
        mng_->getNewPrimal()->update(step, *mng_->getTrialStep(), 1.);
        m_BoundConstraint->projectActive(*m_LowerBound, *m_UpperBound, *mng_->getNewPrimal(), *m_ActiveSet);
        m_ProjectedTrialStep->update(1., *mng_->getNewPrimal(), 0.);
        m_ProjectedTrialStep->update(static_cast<Real>(-1.), *m_CurrentPrimal, 1.);
        m_LinearOperator->apply(mng_, m_ProjectedTrialStep, mng_->getMatrixTimesVector());
        Real old_grad_dot_proj_trial_step = mng_->getNewGradient()->dot(*m_ProjectedTrialStep);
        Real proj_trial_step_dot_hess_times_proj_trial_step = m_ProjectedTrialStep->dot(*mng_->getMatrixTimesVector());

        predicted_reduction = old_grad_dot_proj_trial_step
                + static_cast<Real>(0.5) * proj_trial_step_dot_hess_times_proj_trial_step;
        Real sufficient_decrease_condition = mu0 * m_PredictedReductionBasedOnCauchyStep;
        Real norm_proj_trial_step = m_ProjectedTrialStep->norm();

        if((predicted_reduction <= sufficient_decrease_condition)
                && (norm_proj_trial_step <= scaled_trust_region_radius))
        {
            break;
        }
        else if(itr > max_num_projection)
        {
            break;
        }
        mng_->getNewPrimal()->update(1., *m_CurrentPrimal, 0.);
        step = step * step_contraction;
        itr++;
    }
    dotk::DOTk_TrustRegionStepMng::setPredictedReduction(predicted_reduction);
}

void DOTk_SteihaugTointProjGradStep::computeProjectedCauchyStep
(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real step = -1.;
    Real mu0 = this->getControlUpdateRoutineConstant();
    Real step_contraction = this->getLineSearchContraction();
    Real scaled_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getTrustRegionRadiusScaling()
            * dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius();

    size_t itr = 1;
    size_t max_num_projection = this->getMaxNumProjections();
    while(1)
    {
        m_ProjectedCauchyStep->update(1., *m_CurrentPrimal, 0.);
        m_ProjectedCauchyStep->update(step, *mng_->getNewGradient(), 1.);
        m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_ProjectedCauchyStep);
        m_ProjectedCauchyStep->update(static_cast<Real>(-1.), *m_CurrentPrimal, 1.);
        m_LinearOperator->apply(mng_, m_ProjectedCauchyStep, mng_->getMatrixTimesVector());

        Real old_grad_dot_proj_cauchy_step = mng_->getNewGradient()->dot(*m_ProjectedCauchyStep);
        Real proj_cauchy_step_dot_hess_times_proj_cauchy_step =
                m_ProjectedCauchyStep->dot(*mng_->getMatrixTimesVector());
        m_PredictedReductionBasedOnCauchyStep = old_grad_dot_proj_cauchy_step
                + static_cast<Real>(0.5) * proj_cauchy_step_dot_hess_times_proj_cauchy_step;
        Real sufficient_decrease_condition = mu0 * old_grad_dot_proj_cauchy_step;
        Real norm_proj_cauchy_step = m_ProjectedCauchyStep->norm();

        if((m_PredictedReductionBasedOnCauchyStep <= sufficient_decrease_condition)
                && (norm_proj_cauchy_step <= scaled_trust_region_radius))
        {
            break;
        }
        else if(itr > max_num_projection)
        {
            break;
        }

        step = step * step_contraction;
        itr++;
    }
}

}
