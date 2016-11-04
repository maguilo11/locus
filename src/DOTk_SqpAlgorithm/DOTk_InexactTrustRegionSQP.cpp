/*
 * DOTk_InexactTrustRegionSQP.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_InexactTrustRegionSQP.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_InexactTrustRegionSqpIO.hpp"
#include "DOTk_InexactTrustRegionSqpSolverMng.hpp"

namespace dotk
{

DOTk_InexactTrustRegionSQP::DOTk_InexactTrustRegionSQP
(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
 const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & data_mng_,
 const std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_mng_) :
        dotk::DOTk_SequentialQuadraticProgramming(dotk::types::INEXACT_TRUST_REGION_SQP),
        m_ActualReduction(1.),
        m_PredictedReduction(1.),
        m_ActualOverPredictedReductionRatio(-1.),
        m_PredictedReductionParameter(1e-8),
        m_MeritFunctionPenaltyParameter(1.),
        m_ActualOverPredictedReductionTol(1e-8),
        m_MaxEffectiveTangentialOverTrialStepRatio(2),
        m_DualProbExitCriterion(dotk::types::SOLVER_DID_NOT_CONVERGED),
        m_NormalProbExitCriterion(dotk::types::SOLVER_DID_NOT_CONVERGED),
        m_TangentialProbExitCriterion(dotk::types::SOLVER_DID_NOT_CONVERGED),
        m_TangentialSubProbExitCriterion(dotk::types::SOLVER_DID_NOT_CONVERGED),
        m_Hessian(hessian_),
        m_IO(new dotk::DOTk_InexactTrustRegionSqpIO),
        m_SqpDataMng(data_mng_),
        m_SqpSolverMng(solver_mng_)
{
}

DOTk_InexactTrustRegionSQP::~DOTk_InexactTrustRegionSQP()
{
}

void DOTk_InexactTrustRegionSQP::setMaxTrustRegionRadius(Real radius_)
{
    m_SqpDataMng->setMaxTrustRegionRadius(radius_);
}

void DOTk_InexactTrustRegionSQP::setMinTrustRegionRadius(Real radius_)
{
    m_SqpDataMng->setMinTrustRegionRadius(radius_);
}

void DOTk_InexactTrustRegionSQP::setInitialTrustRegionRadius(Real radius_)
{
    m_SqpDataMng->setTrustRegionRadius(radius_);
}

void DOTk_InexactTrustRegionSQP::setTrustRegionExpansionFactor(Real factor_)
{
    m_SqpDataMng->setTrustRegionExpansionParameter(factor_);
}

void DOTk_InexactTrustRegionSQP::setTrustRegionContractionFactor(Real factor_)
{
    m_SqpDataMng->setTrustRegionContractionParameter(factor_);
}

void DOTk_InexactTrustRegionSQP::setMinActualOverPredictedReductionRatio(Real factor_)
{
    m_SqpDataMng->setMinActualOverPredictedReductionAllowed(factor_);
}

void DOTk_InexactTrustRegionSQP::setToleranceContractionFactor(Real factor_)
{
    m_SqpSolverMng->setToleranceContractionFactor(factor_);
}

void DOTk_InexactTrustRegionSQP::setTangentialTolerance(Real tangential_tolerance_)
{
    m_SqpSolverMng->setTangentialTolerance(tangential_tolerance_);
}

void DOTk_InexactTrustRegionSQP::setTangentialToleranceContractionFactor(Real factor_)
{
    m_SqpSolverMng->setTangentialToleranceContractionFactor(factor_);
}

void DOTk_InexactTrustRegionSQP::setQuasiNormalProblemRelativeTolerance(Real tolerance_)
{
    m_SqpSolverMng->setQuasiNormalProblemRelativeTolerance(tolerance_);
}

void DOTk_InexactTrustRegionSQP::setTangentialSubProbLeftPrecProjectionTolerance(Real tolerance_)
{
    m_SqpSolverMng->setTangentialSubProbLeftPrecProjectionTolerance(tolerance_);
}

void DOTk_InexactTrustRegionSQP::setDualProblemTolerance(Real tolerance_)
{
    m_SqpSolverMng->setDualProblemTolerance(tolerance_);
}

void DOTk_InexactTrustRegionSQP::setDualDotGradientTolerance(Real tolerance_)
{
    m_SqpSolverMng->setDualDotGradientTolerance(tolerance_);
}

void DOTk_InexactTrustRegionSQP::setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(Real parameter_)
{
    m_SqpSolverMng->setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(parameter_);
}

Real DOTk_InexactTrustRegionSQP::getActualReduction() const
{
    return (m_ActualReduction);
}

Real DOTk_InexactTrustRegionSQP::getPredictedReduction() const
{
    return (m_PredictedReduction);
}

Real DOTk_InexactTrustRegionSQP::getActualOverPredictedReductionRatio() const
{
    return (m_ActualOverPredictedReductionRatio);
}

void DOTk_InexactTrustRegionSQP::setPredictedReductionParameter(Real predicted_reduction_parameter_)
{
    m_PredictedReductionParameter = predicted_reduction_parameter_;
}

Real DOTk_InexactTrustRegionSQP::getPredictedReductionParameter() const
{
    return (m_PredictedReductionParameter);
}

void DOTk_InexactTrustRegionSQP::setMeritFunctionPenaltyParameter(Real merit_function_parameter_)
{
    m_MeritFunctionPenaltyParameter = merit_function_parameter_;
}

Real DOTk_InexactTrustRegionSQP::getMeritFunctionPenaltyParameter() const
{
    return (m_MeritFunctionPenaltyParameter);
}

void DOTk_InexactTrustRegionSQP::setActualOverPredictedReductionTolerance(Real tol_)
{
    m_ActualOverPredictedReductionTol = tol_;
}

Real DOTk_InexactTrustRegionSQP::getActualOverPredictedReductionTolerance() const
{
    return (m_ActualOverPredictedReductionTol);
}

void DOTk_InexactTrustRegionSQP::setMaxEffectiveTangentialOverTrialStepRatio(Real ratio_)
{
    m_MaxEffectiveTangentialOverTrialStepRatio = ratio_;
}

Real DOTk_InexactTrustRegionSQP::getMaxEffectiveTangentialOverTrialStepRatio() const
{
    return (m_MaxEffectiveTangentialOverTrialStepRatio);
}

void DOTk_InexactTrustRegionSQP::printDiagnosticsAndSolutionEveryItr()
{
    m_IO->display(dotk::types::ITERATION);
}

void DOTk_InexactTrustRegionSQP::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->display(dotk::types::FINAL);
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSQP::getDualProbExitCriterion() const
{
    return (m_DualProbExitCriterion);
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSQP::getNormalProbExitCriterion() const
{
    return (m_NormalProbExitCriterion);
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSQP::getTangentialProbExitCriterion() const
{
    return (m_TangentialProbExitCriterion);
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSQP::getTangentialSubProbExitCriterion() const
{
    return (m_TangentialSubProbExitCriterion);
}

void DOTk_InexactTrustRegionSQP::getMin()
{
    Real initial_objective_function_value = m_SqpDataMng->getRoutinesMng()->objective(m_SqpDataMng->getNewPrimal());
    m_SqpDataMng->setNewObjectiveFunctionValue(initial_objective_function_value);

    m_SqpDataMng->getRoutinesMng()->equalityConstraint(m_SqpDataMng->getNewPrimal(),
                                                       m_SqpDataMng->getNewEqualityConstraintResidual());

    m_SqpDataMng->getRoutinesMng()->gradient(m_SqpDataMng->getNewPrimal(),
                                             m_SqpDataMng->getNewDual(),
                                             m_SqpDataMng->getNewGradient());

    m_SqpSolverMng->solveDualProb(m_SqpDataMng);

    m_SqpDataMng->getRoutinesMng()->gradient(m_SqpDataMng->getNewPrimal(),
                                             m_SqpDataMng->getNewDual(),
                                             m_SqpDataMng->getNewGradient());

    m_IO->license();
    m_IO->openFile("DOTk_InexactTrustRegionSqpDiagnostics.out");
    size_t itr = 0;
    while(1)
    {
        dotk::DOTk_SequentialQuadraticProgramming::setNumItrDone(itr);
        m_IO->printDiagnosticsReport(this, m_SqpDataMng, m_SqpSolverMng);
        if(dotk::DOTk_SequentialQuadraticProgramming::checkStoppingCriteria(m_SqpDataMng) == true)
        {
            break;
        }
        dotk::DOTk_SequentialQuadraticProgramming::storePreviousSolution(m_SqpDataMng);
        this->solveTrustRegionSubProblem();
        m_SqpDataMng->getRoutinesMng()->gradient(m_SqpDataMng->getNewPrimal(),
                                                 m_SqpDataMng->getNewDual(),
                                                 m_SqpDataMng->getNewGradient());
        ++ itr;
    }

    if(m_IO->display() == dotk::types::FINAL)
    {
        dotk::printDual(m_SqpDataMng->getNewDual());
        dotk::printSolution(m_SqpDataMng->getNewPrimal());
    }

    m_IO->closeFile();
}

void DOTk_InexactTrustRegionSQP::solveTrustRegionSubProblem()
{
    size_t itr = 0;
    bool stop = false;
    Real original_tangential_tolerance = m_SqpSolverMng->getTangentialTolerance();
    while(1)
    {
        dotk::DOTk_SequentialQuadraticProgramming::setNumTrustRegionSubProblemItrDone(itr);
        if(this->checkSubProblemStoppingCriteria() == true || (stop == true))
        {
            break;
        }
        if((itr % 2) == 0)
        {
            // Odd iteration: Use tangential Cauchy step; Even iteration: Compute a new tangential step.
            dotk::types::solver_stop_criterion_t criterion = m_SqpSolverMng->solveQuasiNormalProb(m_SqpDataMng);
            this->setNormalProbExitCriterion(criterion);

            m_SqpDataMng->getRoutinesMng()->jacobian(m_SqpDataMng->getNewPrimal(),
                                                     m_SqpDataMng->m_NormalStep,
                                                     m_SqpDataMng->m_LinearizedEqConstraint);
            m_SqpDataMng->m_LinearizedEqConstraint->axpy(static_cast<Real>(1.),
                                                         *m_SqpDataMng->getNewEqualityConstraintResidual());
            m_Hessian->apply(m_SqpDataMng, m_SqpDataMng->m_NormalStep, m_SqpDataMng->m_HessTimesNormalStep);

            criterion = m_SqpSolverMng->solveTangentialSubProb(m_SqpDataMng);
            this->setTangentialSubProbExitCriterion(criterion);
        }
        m_Hessian->apply(m_SqpDataMng, m_SqpDataMng->m_ProjectedTangentialStep, m_SqpDataMng->getMatrixTimesVector());
        this->correctTrialStep();
        m_SqpSolverMng->setTangentialTolerance(original_tangential_tolerance);
        if((itr % 2) == 0)
        {
            /* If inexactness in the tangential step isn't acceptable, try the tangential
             Cauchy step before re-computing effective tangential and quasi-normal steps. */
            m_SqpDataMng->m_ProjectedTangentialStep->copy(*m_SqpDataMng->m_ProjectedTangentialCauchyStep);
        }
        else
        {
            /* If the tangential Cauchy point didn't work; then, tighten each augmented system
             solve tolerance and re-compute effective tangential and quasi-normal steps. */
            stop = m_SqpSolverMng->adjustSolversTolerance();
        }
        Real actual_reduction = this->computeActualReduction();
        Real actual_over_predicted_reduction = actual_reduction / this->getPredictedReduction();
        this->setActualOverPredictedReductionRatio(actual_over_predicted_reduction);
        stop = this->updateTrustRegionRadius(actual_over_predicted_reduction);
        if(stop == false)
        {
            this->resetState();
        }
        ++ itr;
    }
}

bool DOTk_InexactTrustRegionSQP::updateTrustRegionRadius(Real actual_over_predicted_reduction_)
{
    bool trust_region_updated = false;
    Real min_trust_region_radius = m_SqpDataMng->getMinTrustRegionRadius();
    Real max_trust_region_radius = m_SqpDataMng->getMaxTrustRegionRadius();
    Real trust_region_radius = m_SqpDataMng->getTrustRegionRadius();
    Real norm_trial_step = m_SqpDataMng->getTrialStep()->norm();

    if(actual_over_predicted_reduction_ > static_cast<Real>(0.9))
    {
        // min{ max{ 7*dot(trial_step) ,trust_region_radius_k, trust_region_radius_min }, trust_region_radius_max }
        trust_region_radius =
                trust_region_radius > min_trust_region_radius ? trust_region_radius: min_trust_region_radius;
        Real condition = static_cast<Real>(7.) * norm_trial_step;
        trust_region_radius = condition > trust_region_radius ? condition: trust_region_radius;
        trust_region_radius =
                trust_region_radius < max_trust_region_radius ? trust_region_radius: max_trust_region_radius;
        trust_region_updated = true;
    }
    else if(actual_over_predicted_reduction_ >= static_cast<Real>(0.8))
    {
        // min{ max{ ExpansionParam*dot(trial_step) ,trust_region_radius_k, trust_region_radius_min }, trust_region_radius_max }
        trust_region_radius =
                trust_region_radius > min_trust_region_radius ? trust_region_radius: min_trust_region_radius;
        Real condition = m_SqpDataMng->getTrustRegionExpansionParameter() * norm_trial_step;
        trust_region_radius = condition > trust_region_radius ? condition: trust_region_radius;
        trust_region_radius =
                trust_region_radius < max_trust_region_radius ? trust_region_radius: max_trust_region_radius;
        trust_region_updated = true;
    }
    else if(actual_over_predicted_reduction_ > m_SqpDataMng->getMinActualOverPredictedReductionAllowed())
    {
        trust_region_updated = true;
    }
    else if(std::isnan(actual_over_predicted_reduction_))
    {
        trust_region_radius *= m_SqpDataMng->getTrustRegionContractionParameter();
    }
    else
    {
        Real norm_quasi_normal_step = m_SqpDataMng->m_NormalStep->norm();
        Real norm_tangential_step = m_SqpDataMng->m_TangentialStep->norm();
        Real new_trust_region_radius = m_SqpDataMng->getTrustRegionContractionParameter()
                * (norm_quasi_normal_step > norm_tangential_step ? norm_quasi_normal_step: norm_tangential_step);
        trust_region_radius =
                new_trust_region_radius > max_trust_region_radius ?
                        m_SqpDataMng->getTrustRegionContractionParameter() * trust_region_radius: new_trust_region_radius;
    }
    m_SqpDataMng->setTrustRegionRadius(trust_region_radius);

    return (trust_region_updated);
}

Real DOTk_InexactTrustRegionSQP::computeActualReduction()
{
    Real penalty_param = this->getMeritFunctionPenaltyParameter();
    Real objective_function_value = m_SqpDataMng->getRoutinesMng()->objective(m_SqpDataMng->getNewPrimal());
    m_SqpDataMng->setNewObjectiveFunctionValue(objective_function_value);
    m_SqpDataMng->getRoutinesMng()->equalityConstraint(m_SqpDataMng->getNewPrimal(),
                                                       m_SqpDataMng->getNewEqualityConstraintResidual());
    Real change_in_objective_function = m_SqpDataMng->getOldObjectiveFunctionValue()
            - m_SqpDataMng->getNewObjectiveFunctionValue();

    if(std::abs(change_in_objective_function / m_SqpDataMng->getOldObjectiveFunctionValue())
            < std::numeric_limits<Real>::epsilon())
    {
        change_in_objective_function = static_cast<Real>(0.);
    }

    Real old_dual_dot_old_eq_constraint =
            m_SqpDataMng->getOldDual()->dot(*m_SqpDataMng->getOldEqualityConstraintResidual());
    Real new_dual_dot_new_eq_constraint =
            m_SqpDataMng->getNewDual()->dot(*m_SqpDataMng->getNewEqualityConstraintResidual());
    Real old_eq_constraint_dot_old_eq_constraint =
            m_SqpDataMng->getOldEqualityConstraintResidual()->dot(*m_SqpDataMng->getOldEqualityConstraintResidual());
    Real new_eq_constraint_dot_new_eq_constraint =
            m_SqpDataMng->getNewEqualityConstraintResidual()->dot(*m_SqpDataMng->getNewEqualityConstraintResidual());

    Real actual_reduction = change_in_objective_function
            + (old_dual_dot_old_eq_constraint - new_dual_dot_new_eq_constraint)
            + (penalty_param * (old_eq_constraint_dot_old_eq_constraint - new_eq_constraint_dot_new_eq_constraint));
    this->setActualReduction(actual_reduction);

    return (actual_reduction);
}

bool DOTk_InexactTrustRegionSQP::checkSubProblemStoppingCriteria()
{
    Real trial_step_dot_trial_step = m_SqpDataMng->getTrialStep()->dot(*m_SqpDataMng->getTrialStep());
    Real proj_tang_step_dot_proj_tang_step =
            m_SqpDataMng->m_ProjectedTangentialStep->dot(*m_SqpDataMng->m_ProjectedTangentialStep);

    Real trust_region_radius = m_SqpDataMng->getTrustRegionRadius();
    Real trial_step_tol = dotk::DOTk_SequentialQuadraticProgramming::getTrialStepTolerance();
    Real ratio = this->getMaxEffectiveTangentialOverTrialStepRatio();

    bool stop = false;
    size_t itr = dotk::DOTk_SequentialQuadraticProgramming::getNumTrustRegionSubProblemItrDone();
    if(trust_region_radius < trial_step_tol)
    {
        stop = true;
    }
    else if((proj_tang_step_dot_proj_tang_step <= (ratio * ratio * trial_step_dot_trial_step)) && (itr > 0))
    {
        stop = true;
    }

    return (stop);
}

void DOTk_InexactTrustRegionSQP::correctTrialStep()
{
    Real tolerance = this->getActualOverPredictedReductionTolerance();
    Real stopping_criterion = static_cast<Real>(0.5) * (static_cast<Real>(1.0) - tolerance);
    Real tangential_tolerance = m_SqpSolverMng->getTangentialTolerance();
    Real tangential_tol_contraction_factor = m_SqpSolverMng->getTangentialToleranceContractionFactor();
    Real prediceted_reduction_residual_over_prediceted_reduction = std::numeric_limits<Real>::max();

    this->setPredictedReduction(static_cast<Real>(1.));

    m_SqpDataMng->getTrialStep()->copy(*m_SqpDataMng->m_NormalStep);
    m_SqpDataMng->getTrialStep()->axpy(static_cast<Real>(1.), *m_SqpDataMng->m_ProjectedTangentialStep);
    while(prediceted_reduction_residual_over_prediceted_reduction > stopping_criterion)
    {
        m_SqpDataMng->getNewPrimal()->copy(*m_SqpDataMng->getOldPrimal());
        dotk::types::solver_stop_criterion_t criterion = m_SqpSolverMng->solveTangentialProb(m_SqpDataMng);
        this->setTangentialProbExitCriterion(criterion);

        m_SqpDataMng->getRoutinesMng()->jacobian(m_SqpDataMng->getNewPrimal(),
                                                 m_SqpDataMng->m_TangentialStep,
                                                 m_SqpDataMng->m_JacobianTimesTangentialStep);
        m_SqpDataMng->getTrialStep()->copy(*m_SqpDataMng->m_NormalStep);
        m_SqpDataMng->getTrialStep()->axpy(static_cast<Real>(1.), *m_SqpDataMng->m_TangentialStep);
        m_SqpDataMng->getNewPrimal()->axpy(static_cast<Real>(1.), *m_SqpDataMng->getTrialStep());
        m_SqpDataMng->getRoutinesMng()->gradient(m_SqpDataMng->getNewPrimal(),
                                                 m_SqpDataMng->getOldDual(),
                                                 m_SqpDataMng->getNewGradient());

        criterion = m_SqpSolverMng->solveDualProb(m_SqpDataMng);
        this->setDualProbExitCriterion(criterion);

        Real partial_predicted_reduction = this->computePartialPredictedReduction();
        this->updateMeritFunctionPenaltyParameter(partial_predicted_reduction);
        Real predicted_reduction_residual = this->computePredictedReductionResidual();
        Real predicted_reduction = this->computePredictedReduction(partial_predicted_reduction);
        prediceted_reduction_residual_over_prediceted_reduction = std::abs(predicted_reduction_residual)
                / predicted_reduction;

        tangential_tolerance = tangential_tolerance * tangential_tol_contraction_factor;
        m_SqpSolverMng->setTangentialTolerance(tangential_tolerance);
        if(tangential_tolerance < std::numeric_limits<Real>::epsilon())
        {
            break;
        }
    }
}

void DOTk_InexactTrustRegionSQP::updateMeritFunctionPenaltyParameter(Real partial_predicted_reduction_)
{
    Real predicted_reduction_param = this->getPredictedReductionParameter();
    Real current_merit_function_penalty_param = this->getMeritFunctionPenaltyParameter();
    Real linearized_eq_constraint_dot_linearized_eq_constraint =
            m_SqpDataMng->m_LinearizedEqConstraint->dot(*m_SqpDataMng->m_LinearizedEqConstraint);
    Real eq_constraint_dot_eq_constraint =
            m_SqpDataMng->getNewEqualityConstraintResidual()->dot(*m_SqpDataMng->getNewEqualityConstraintResidual());

    Real update_criterion = static_cast<Real>(-0.5) * current_merit_function_penalty_param
            * (eq_constraint_dot_eq_constraint - linearized_eq_constraint_dot_linearized_eq_constraint);
    if(partial_predicted_reduction_ < update_criterion)
    {
        current_merit_function_penalty_param = ((static_cast<Real>(-2.) * partial_predicted_reduction_)
                / (eq_constraint_dot_eq_constraint - linearized_eq_constraint_dot_linearized_eq_constraint))
                + predicted_reduction_param;
    }
    this->setMeritFunctionPenaltyParameter(current_merit_function_penalty_param);
}

Real DOTk_InexactTrustRegionSQP::computePartialPredictedReduction()
{
    Real partial_predicted_reduction = 0.;

    partial_predicted_reduction -= m_SqpDataMng->m_ProjectedGradient->dot(*m_SqpDataMng->m_ProjectedTangentialStep);

    partial_predicted_reduction -= m_SqpDataMng->getOldGradient()->dot(*m_SqpDataMng->m_NormalStep);

    partial_predicted_reduction -= static_cast<Real>(0.5)
            * m_SqpDataMng->m_NormalStep->dot(*m_SqpDataMng->m_HessTimesNormalStep);

    partial_predicted_reduction -= static_cast<Real>(0.5)
            * m_SqpDataMng->m_ProjectedTangentialStep->dot(*m_SqpDataMng->getMatrixTimesVector());

    partial_predicted_reduction -= m_SqpDataMng->m_DeltaDual->dot(*m_SqpDataMng->m_LinearizedEqConstraint);

    return (partial_predicted_reduction);
}

Real DOTk_InexactTrustRegionSQP::computePredictedReductionResidual()
{
    Real predicted_reduction_residual = -m_SqpDataMng->m_DeltaDual->dot(*m_SqpDataMng->m_JacobianTimesTangentialStep);

    Real merit_function_penalty_param = this->getMeritFunctionPenaltyParameter();
    Real jacobian_times_tang_step_dot_jacobian_times_tang_step =
            m_SqpDataMng->m_JacobianTimesTangentialStep->dot(*m_SqpDataMng->m_JacobianTimesTangentialStep);
    predicted_reduction_residual -= merit_function_penalty_param
            * jacobian_times_tang_step_dot_jacobian_times_tang_step;

    Real jacobian_times_tang_step_dot_linearized_eq_constraint =
            m_SqpDataMng->m_JacobianTimesTangentialStep->dot(*m_SqpDataMng->m_LinearizedEqConstraint);
    predicted_reduction_residual -= static_cast<Real>(2.) * merit_function_penalty_param
            * jacobian_times_tang_step_dot_linearized_eq_constraint;

    return (predicted_reduction_residual);
}

Real DOTk_InexactTrustRegionSQP::computePredictedReduction(Real partial_predicted_reduction_)
{
    Real merit_function_penalty_param = this->getMeritFunctionPenaltyParameter();
    Real eq_constraint_residual_dot_eq_constraint_residual =
            m_SqpDataMng->getNewEqualityConstraintResidual()->dot(*m_SqpDataMng->getNewEqualityConstraintResidual());
    Real linearized_eq_constraint_dot_linearized_eq_constraint =
            m_SqpDataMng->m_LinearizedEqConstraint->dot(*m_SqpDataMng->m_LinearizedEqConstraint);

    Real predicted_reduction = partial_predicted_reduction_
            + merit_function_penalty_param
                    * (eq_constraint_residual_dot_eq_constraint_residual
                            - linearized_eq_constraint_dot_linearized_eq_constraint);
    this->setPredictedReduction(predicted_reduction);

    return (predicted_reduction);
}

void DOTk_InexactTrustRegionSQP::resetState()
{
    m_SqpDataMng->setNewObjectiveFunctionValue(m_SqpDataMng->getOldObjectiveFunctionValue());
    m_SqpDataMng->getNewPrimal()->copy(*m_SqpDataMng->getOldPrimal());
    m_SqpDataMng->getNewEqualityConstraintResidual()->copy(*m_SqpDataMng->getOldEqualityConstraintResidual());
    m_SqpDataMng->getNewGradient()->copy(*m_SqpDataMng->getOldGradient());
    m_SqpDataMng->getNewDual()->copy(*m_SqpDataMng->getOldDual());
}

void DOTk_InexactTrustRegionSQP::setActualReduction(Real actual_reduction_)
{
    m_ActualReduction = actual_reduction_;
}

void DOTk_InexactTrustRegionSQP::setPredictedReduction(Real predicted_reduction_)
{
    m_PredictedReduction = predicted_reduction_;
}

void DOTk_InexactTrustRegionSQP::setActualOverPredictedReductionRatio(Real actual_over_predicted_reduction_)
{
    m_ActualOverPredictedReductionRatio = actual_over_predicted_reduction_;
}

void DOTk_InexactTrustRegionSQP::setDualProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_)
{
    m_DualProbExitCriterion = criterion_;
}

void DOTk_InexactTrustRegionSQP::setNormalProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_)
{
    m_NormalProbExitCriterion = criterion_;
}

void DOTk_InexactTrustRegionSQP::setTangentialProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_)
{
    m_TangentialProbExitCriterion = criterion_;
}

void DOTk_InexactTrustRegionSQP::setTangentialSubProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_)
{
    m_TangentialSubProbExitCriterion = criterion_;
}

}
