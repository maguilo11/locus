/*
 * DOTk_SubProblemGCMMA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>
#include <cstdlib>
#include <algorithm>

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_SubProblemGCMMA.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_DualObjectiveFunctionMMA.hpp"

namespace dotk
{

DOTk_SubProblemGCMMA::DOTk_SubProblemGCMMA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_) :
        dotk::DOTk_SubProblemCCSA(dotk::ccsa::subproblem_t::GCMMA),
        m_ObjectiveFunctionRho(1),
        m_ObjectiveFunctionMinRho(1e-5),
        m_NewTrialObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_OldTrialObjectiveFunctionValue(std::numeric_limits<Real>::min()),
        m_DeltaControl(data_mng_->m_CurrentControl->clone()),
        m_TrialControl(data_mng_->m_CurrentControl->clone()),
        m_InequalityConstraintRho(data_mng_->m_Dual->clone()),
        m_TrialFeasibilityMeasures(data_mng_->m_Dual->clone()),
        m_TrialInequalityResiduals(data_mng_->m_Dual->clone()),
        m_InequalityConstraintMinRho(data_mng_->m_Dual->clone()),
        m_Bounds(new dotk::DOTk_BoundConstraints),
        m_DualSolver(new dotk::DOTk_DualSolverNLCG(data_mng_->m_Primal)),
        m_DualObjectiveFunction(new dotk::DOTk_DualObjectiveFunctionMMA(data_mng_))
{
    this->initialize();
}

DOTk_SubProblemGCMMA::DOTk_SubProblemGCMMA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                     const std::tr1::shared_ptr<dotk::DOTk_DualSolverCCSA> & dual_solver_) :
        dotk::DOTk_SubProblemCCSA(dotk::ccsa::subproblem_t::GCMMA),
        m_ObjectiveFunctionRho(1),
        m_ObjectiveFunctionMinRho(1e-5),
        m_NewTrialObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_OldTrialObjectiveFunctionValue(std::numeric_limits<Real>::min()),
        m_DeltaControl(data_mng_->m_CurrentControl->clone()),
        m_TrialControl(data_mng_->m_CurrentControl->clone()),
        m_InequalityConstraintRho(data_mng_->m_Dual->clone()),
        m_TrialFeasibilityMeasures(data_mng_->m_Dual->clone()),
        m_TrialInequalityResiduals(data_mng_->m_Dual->clone()),
        m_InequalityConstraintMinRho(data_mng_->m_Dual->clone()),
        m_Bounds(new dotk::DOTk_BoundConstraints),
        m_DualSolver(dual_solver_),
        m_DualObjectiveFunction(new dotk::DOTk_DualObjectiveFunctionMMA(data_mng_))
{
    this->initialize();
}

DOTk_SubProblemGCMMA::~DOTk_SubProblemGCMMA()
{
}

dotk::ccsa::stopping_criterion_t DOTk_SubProblemGCMMA::getDualSolverStoppingCriterion() const
{
    return (m_DualSolver->getStoppingCriterion());
}

void DOTk_SubProblemGCMMA::setDualObjectiveEpsilonParameter(Real input_)
{
    m_DualObjectiveFunction->setEpsilon(input_);
}

void DOTk_SubProblemGCMMA::solve(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    Real scale = dotk::DOTk_SubProblemCCSA::getDualObjectiveTrialControlBoundScaling();
    m_DualObjectiveFunction->updateMovingAsymptotes(data_mng_->m_CurrentControl, data_mng_->m_CurrentSigma);
    m_DualObjectiveFunction->updateTrialControlBounds(scale, data_mng_->m_CurrentControl, data_mng_->m_CurrentSigma);
    m_DualObjectiveFunction->setCurrentObjectiveFunctionValue(data_mng_->m_CurrentObjectiveFunctionValue);
    m_DualObjectiveFunction->setCurrentInequalityConstraintResiduals(data_mng_->m_CurrentInequalityResiduals);

    dotk::DOTk_SubProblemCCSA::resetIterationCount();
    size_t max_num_iterations = dotk::DOTk_SubProblemCCSA::getMaxNumIterations();
    while(dotk::DOTk_SubProblemCCSA::getIterationCount() < max_num_iterations)
    {
        m_DualObjectiveFunction->updateObjectiveCoefficientVectors(m_ObjectiveFunctionRho,
                                                                   data_mng_->m_CurrentSigma,
                                                                   data_mng_->m_CurrentObjectiveGradient);
        m_DualObjectiveFunction->updateInequalityCoefficientVectors(m_InequalityConstraintRho,
                                                                    data_mng_->m_CurrentSigma,
                                                                    data_mng_->m_CurrentInequalityGradients);

        m_DualSolver->solve(m_DualObjectiveFunction, data_mng_->m_Dual);
        m_DualObjectiveFunction->gatherTrialControl(m_TrialControl);
        m_Bounds->projectActive(*data_mng_->m_ControlLowerBound,
                                *data_mng_->m_ControlUpperBound,
                                *m_TrialControl,
                                *data_mng_->m_ActiveSet);
        m_NewTrialObjectiveFunctionValue = data_mng_->evaluateObjectiveFunction(m_TrialControl);
        data_mng_->evaluateInequalityConstraints(m_TrialControl, m_TrialInequalityResiduals, m_TrialFeasibilityMeasures);

        m_DeltaControl->update(1., *m_TrialControl, 0.);
        m_DeltaControl->update(-1., *data_mng_->m_CurrentControl, 1.);
        this->updateObjectiveGlobalizationScalingParameters(data_mng_);
        this->updateInequalityGlobalizationScalingParameters(data_mng_);

        dotk::DOTk_SubProblemCCSA::updateIterationCount();
        if(this->stoppingCriteriaSatisfied(data_mng_))
        {
            break;
        }
        m_DualSolver->reset();
        m_OldTrialObjectiveFunctionValue = m_NewTrialObjectiveFunctionValue;
    }

    this->updateState(data_mng_);
    if(dotk::DOTk_SubProblemCCSA::getIterationCount() >= max_num_iterations)
    {
        dotk::DOTk_SubProblemCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::MAX_NUMBER_ITERATIONS);
    }
}

void DOTk_SubProblemGCMMA::initialize()
{
    m_InequalityConstraintRho->fill(1.);
    m_InequalityConstraintMinRho->fill(1e-5);
}

void DOTk_SubProblemGCMMA::checkGlobalizationScalingParameters()
{
    size_t number_inequalitites = m_InequalityConstraintRho->size();
    m_ObjectiveFunctionRho = std::max(static_cast<Real>(0.1) * m_ObjectiveFunctionRho, m_ObjectiveFunctionMinRho);
    for(size_t index = 0; index < number_inequalitites; ++ index)
    {
        (*m_InequalityConstraintRho)[index] = std::max(static_cast<Real>(0.1) * (*m_InequalityConstraintRho)[index],
                                                       (*m_InequalityConstraintMinRho)[index]);
    }
}

void DOTk_SubProblemGCMMA::updateState(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    this->checkGlobalizationScalingParameters();

    data_mng_->m_CurrentControl->update(1., *m_TrialControl, 0.);
    data_mng_->m_CurrentInequalityResiduals->update(1., *m_TrialInequalityResiduals, 0.);
    data_mng_->m_CurrentFeasibilityMeasures->update(1., *m_TrialFeasibilityMeasures, 0.);
    data_mng_->m_CurrentObjectiveFunctionValue = m_NewTrialObjectiveFunctionValue;
}

bool DOTk_SubProblemGCMMA::stoppingCriteriaSatisfied(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    bool criteria_satisfied = false;
    Real residual_norm = dotk::ccsa::computeResidualNorm(m_TrialControl, data_mng_->m_Dual, data_mng_);
    Real objective_stagnation_measure =
            std::abs(m_NewTrialObjectiveFunctionValue - m_OldTrialObjectiveFunctionValue);

    if(residual_norm < dotk::DOTk_SubProblemCCSA::getResidualTolerance())
    {
        criteria_satisfied = true;
        dotk::DOTk_SubProblemCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE);
    }
    else if(objective_stagnation_measure < dotk::DOTk_SubProblemCCSA::getStagnationTolerance())
    {
        criteria_satisfied = true;
        dotk::DOTk_SubProblemCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::OBJECTIVE_STAGNATION);
    }

    return (criteria_satisfied);
}

void DOTk_SubProblemGCMMA::updateObjectiveGlobalizationScalingParameters(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    Real w_function_evaluation = 0;
    Real v_function_evaluation = 0;
    size_t number_primals = data_mng_->m_CurrentControl->size();
    for(size_t index = 0; index < number_primals; ++index)
    {
        Real numerator = (*m_DeltaControl)[index] * (*m_DeltaControl)[index];
        Real denominator = ((*data_mng_->m_CurrentSigma)[index] * (*data_mng_->m_CurrentSigma)[index])
                - ((*m_DeltaControl)[index] * (*m_DeltaControl)[index]);
        w_function_evaluation += numerator / denominator;

        numerator = (((*data_mng_->m_CurrentSigma)[index] * (*data_mng_->m_CurrentSigma)[index])
                * (*data_mng_->m_CurrentObjectiveGradient)[index] * (*m_DeltaControl)[index])
                + ((*data_mng_->m_CurrentSigma)[index] * std::abs((*data_mng_->m_CurrentObjectiveGradient)[index])
                        * ((*m_DeltaControl)[index] * (*m_DeltaControl)[index]));
        v_function_evaluation += numerator / denominator;
    }
    w_function_evaluation = static_cast<Real>(0.5)*w_function_evaluation;
    v_function_evaluation = data_mng_->m_CurrentObjectiveFunctionValue + v_function_evaluation;

    Real ccsa_function_value = v_function_evaluation + m_ObjectiveFunctionRho*w_function_evaluation;

    Real actual_over_predicted_reduction = (m_NewTrialObjectiveFunctionValue - ccsa_function_value)
            / w_function_evaluation;
    if(actual_over_predicted_reduction > 0)
    {
        Real value_one = static_cast<Real>(10) * m_ObjectiveFunctionRho;
        Real value_two = static_cast<Real>(1.1) * (m_ObjectiveFunctionRho + actual_over_predicted_reduction);
        m_ObjectiveFunctionRho = std::min(value_one, value_two);
    }
}

void DOTk_SubProblemGCMMA::updateInequalityGlobalizationScalingParameters(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    size_t number_primals = data_mng_->m_CurrentControl->size();
    for(size_t index_i = 0; index_i < data_mng_->getNumberInequalityConstraints(); ++index_i)
    {
        Real w_function_evaluation = 0;
        Real v_function_evaluation = 0;
        for(size_t index_j = 0; index_j < number_primals; ++index_j)
        {
            Real numerator = (*m_DeltaControl)[index_j] * (*m_DeltaControl)[index_j];
            Real denominator = ((*data_mng_->m_CurrentSigma)[index_j] * (*data_mng_->m_CurrentSigma)[index_j])
                    - ((*m_DeltaControl)[index_j] * (*m_DeltaControl)[index_j]);
            w_function_evaluation += numerator / denominator;

            numerator = (((*data_mng_->m_CurrentSigma)[index_j] * (*data_mng_->m_CurrentSigma)[index_j])
                    * (*data_mng_->m_CurrentInequalityGradients->basis(index_i))[index_j]
                    * (*m_DeltaControl)[index_j])
                    + ((*data_mng_->m_CurrentSigma)[index_j]
                            * std::abs((*data_mng_->m_CurrentInequalityGradients->basis(index_i))[index_j])
                            * ((*m_DeltaControl)[index_j] * (*m_DeltaControl)[index_j]));
            v_function_evaluation += numerator / denominator;
        }
        w_function_evaluation = static_cast<Real>(0.5) * w_function_evaluation;
        v_function_evaluation = (*data_mng_->m_CurrentInequalityResiduals)[index_i] + v_function_evaluation;

        Real ccsa_function_value = v_function_evaluation
                + (*m_InequalityConstraintRho)[index_i] * w_function_evaluation;

        Real actual_over_predicted_reduction = ((*m_TrialInequalityResiduals)[index_i] - ccsa_function_value)
                / w_function_evaluation;
        if(actual_over_predicted_reduction > static_cast<Real>(0.))
        {
            Real value_one = static_cast<Real>(10) * (*m_InequalityConstraintRho)[index_i];
            Real value_two = static_cast<Real>(1.1)
                    * ((*m_InequalityConstraintRho)[index_i] + actual_over_predicted_reduction);
            (*m_InequalityConstraintRho)[index_i] = std::min(value_one, value_two);
        }
    }
}

}
