/*
 * DOTk_AlgorithmCCSA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>
#include <algorithm>

#include "vector.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTK_MethodCcsaIO.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_SubProblemMMA.hpp"
#include "DOTk_DualSolverCCSA.hpp"
#include "DOTk_SubProblemGCMMA.hpp"
#include "DOTk_BoundConstraints.hpp"

namespace dotk
{

DOTk_AlgorithmCCSA::DOTk_AlgorithmCCSA(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                                       const std::shared_ptr<dotk::DOTk_SubProblemCCSA> & sub_problem_) :
        m_StoppingCriterion(dotk::ccsa::stopping_criterion_t::NOT_CONVERGED),
        m_IterationCount(0),
        m_MaxNumIterations(100),
        m_ResidualTolerance(1e-4),
        m_GradientTolerance(1e-3),
        m_CurrentMaxResidual(std::numeric_limits<Real>::max()),
        m_CurrentResidualNorm(std::numeric_limits<Real>::max()),
        m_FeasibilityTolerance(1e-4),
        m_ControlStagnationTolerance(1e-3),
        m_CurrentMaxFeasibilityMeasure(std::numeric_limits<Real>::max()),
        m_CurrentObjectiveGradientNorm(std::numeric_limits<Real>::max()),
        m_InitialNormObjectiveGradient(std::numeric_limits<Real>::max()),
        m_MovingAsymptoteUpperBoundScale(10),
        m_MovingAsymptoteLowerBoundScale(0.01),
        m_CurrentControlStagnationMeasure(std::numeric_limits<Real>::max()),
        m_MovingAsymptoteExpansionParameter(1.2),
        m_MovingAsymptoteContractionParameter(0.4),
        m_OldSigma(data_mng_->m_CurrentControl->clone()),
        m_AuxiliaryZcandidates(data_mng_->m_Dual->clone()),
        m_ControlAtIterationIminusTwo(data_mng_->m_CurrentControl->clone()),
        m_IO(std::make_shared<dotk::DOTK_MethodCcsaIO>()),
        m_DataMng(data_mng_),
        m_Bounds(std::make_shared<dotk::DOTk_BoundConstraints>()),
        m_SubProblem(sub_problem_)
{
}

DOTk_AlgorithmCCSA::~DOTk_AlgorithmCCSA()
{
}

dotk::ccsa::stopping_criterion_t DOTk_AlgorithmCCSA::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

void DOTk_AlgorithmCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t type_)
{
    m_StoppingCriterion = type_;
}

size_t DOTk_AlgorithmCCSA::getIterationCount() const
{
    return (m_IterationCount);
}

size_t DOTk_AlgorithmCCSA::getMaxNumIterations() const
{
    return (m_MaxNumIterations);
}

void DOTk_AlgorithmCCSA::setMaxNumIterations(size_t max_num_iterations_)
{
    m_MaxNumIterations = max_num_iterations_;
}

Real DOTk_AlgorithmCCSA::getResidualTolerance() const
{
    return (m_ResidualTolerance);
}

void DOTk_AlgorithmCCSA::setResidualTolerance(Real tolerance_)
{
    m_ResidualTolerance = tolerance_;
}

Real DOTk_AlgorithmCCSA::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

void DOTk_AlgorithmCCSA::setGradientTolerance(Real tolerance_)
{
    m_GradientTolerance = tolerance_;
}

Real DOTk_AlgorithmCCSA::getCurrentMaxResidual() const
{
    return (m_CurrentMaxResidual);
}

void DOTk_AlgorithmCCSA::setCurrentMaxResidual(Real input_)
{
    m_CurrentMaxResidual = input_;
}

Real DOTk_AlgorithmCCSA::getCurrentResidualNorm() const
{
    return (m_CurrentResidualNorm);
}

void DOTk_AlgorithmCCSA::setCurrentResidualNorm(Real input_)
{
    m_CurrentResidualNorm = input_;
}

Real DOTk_AlgorithmCCSA::getFeasibilityTolerance() const
{
    return (m_FeasibilityTolerance);
}

void DOTk_AlgorithmCCSA::setFeasibilityTolerance(Real tolerance_)
{
    m_FeasibilityTolerance = tolerance_;
}

Real DOTk_AlgorithmCCSA::getControlStagnationTolerance() const
{
    return (m_ControlStagnationTolerance);
}

void DOTk_AlgorithmCCSA::setControlStagnationTolerance(Real tolerance_)
{
    m_ControlStagnationTolerance = tolerance_;
}

Real DOTk_AlgorithmCCSA::getCurrentMaxFeasibilityMeasure() const
{
    return (m_CurrentMaxFeasibilityMeasure);
}

void DOTk_AlgorithmCCSA::setCurrentMaxFeasibilityMeasure(Real tolerance_)
{
    m_CurrentMaxFeasibilityMeasure = tolerance_;
}

Real DOTk_AlgorithmCCSA::getCurrentObjectiveGradientNorm() const
{
    return (m_CurrentObjectiveGradientNorm);
}

void DOTk_AlgorithmCCSA::setCurrentObjectiveGradientNorm(Real tolerance_)
{
    m_CurrentObjectiveGradientNorm = tolerance_;
}

Real DOTk_AlgorithmCCSA::getCurrentControlStagnationMeasure() const
{
    return (m_CurrentControlStagnationMeasure);
}

void DOTk_AlgorithmCCSA::setCurrentControlStagnationMeasure(Real input_)
{
    m_CurrentControlStagnationMeasure = input_;
}

Real DOTk_AlgorithmCCSA::getMovingAsymptoteUpperBoundScale() const
{
    return (m_MovingAsymptoteUpperBoundScale);
}

void DOTk_AlgorithmCCSA::setMovingAsymptoteUpperBoundScale(Real input_)
{
    m_MovingAsymptoteUpperBoundScale = input_;
}

Real DOTk_AlgorithmCCSA::getMovingAsymptoteLowerBoundScale() const
{
    return (m_MovingAsymptoteLowerBoundScale);
}

void DOTk_AlgorithmCCSA::setMovingAsymptoteLowerBoundScale(Real input_)
{
    m_MovingAsymptoteLowerBoundScale = input_;
}

Real DOTk_AlgorithmCCSA::getMovingAsymptoteExpansionParameter() const
{
    return (m_MovingAsymptoteExpansionParameter);
}

void DOTk_AlgorithmCCSA::setMovingAsymptoteExpansionParameter(Real input_)
{
    m_MovingAsymptoteExpansionParameter = input_;
}

Real DOTk_AlgorithmCCSA::getMovingAsymptoteContractionParameter() const
{
    return (m_MovingAsymptoteContractionParameter);
}

void DOTk_AlgorithmCCSA::setMovingAsymptoteContractionParameter(Real input_)
{
    m_MovingAsymptoteContractionParameter = input_;
}

void DOTk_AlgorithmCCSA::setDualObjectiveEpsilonParameter(Real input_)
{
    m_SubProblem->setDualObjectiveEpsilonParameter(input_);
}

void DOTk_AlgorithmCCSA::setDualObjectiveTrialControlBoundScaling(Real input_)
{
    m_SubProblem->setDualObjectiveTrialControlBoundScaling(input_);
}

void DOTk_AlgorithmCCSA::printDiagnosticsAndSolutionAtEveryItr()
{
    m_IO->printSolutionAtEachIteration();
}

void DOTk_AlgorithmCCSA::printDiagnosticsAtEveryItrAndSolutionAtTheEnd()
{
    m_IO->printSolutionAtFinalIteration();
}

void DOTk_AlgorithmCCSA::getMin()
{
    m_IO->openFile("DOTk_CcsaMethodDiagnostics.out");
    m_DataMng->initializeAuxiliaryVariables();
    m_DataMng->evaluateObjectiveFunction();
    m_DataMng->evaluateInequalityConstraintResiduals();

    size_t max_num_iterations = this->getMaxNumIterations();
    while(dotk::DOTk_AlgorithmCCSA::getIterationCount() < max_num_iterations)
    {
        m_DataMng->computeFunctionGradients();
        bool stopping_criterion_satisfied = this->stoppingCriteriaSatisfied();
        m_IO->print(this, m_DataMng);

        if(stopping_criterion_satisfied == true)
        {
            break;
        }

        this->updateSigmaParameters();

        m_ControlAtIterationIminusTwo->update(1., *m_DataMng->m_PreviousControl, 0.);
        m_DataMng->m_PreviousControl->update(1., *m_DataMng->m_CurrentControl, 0.);

        m_SubProblem->solve(m_DataMng);

        this->updateIterationCount();
    }
    if(dotk::DOTk_AlgorithmCCSA::getIterationCount() >= max_num_iterations)
    {
        dotk::DOTk_AlgorithmCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::MAX_NUMBER_ITERATIONS);
    }

    m_IO->printSolution(m_DataMng);
    m_IO->closeFile();
}

void DOTk_AlgorithmCCSA::updateIterationCount()
{
    m_IterationCount++;
}

void DOTk_AlgorithmCCSA::updateSigmaParameters()
{
    size_t number_controls = m_DataMng->m_CurrentControl->size();
    m_OldSigma->update(1., *m_DataMng->m_CurrentSigma, 0.);

    if(this->getIterationCount() < static_cast<size_t>(2))
    {
        m_DataMng->m_CurrentSigma->update(1., *m_DataMng->m_ControlUpperBound, 0.);
        m_DataMng->m_CurrentSigma->update(static_cast<Real>(-1.), *m_DataMng->m_ControlLowerBound, 1.);
        m_DataMng->m_CurrentSigma->scale(static_cast<Real>(0.5));
    }
    else
    {
        Real expansion_parameter = this->getMovingAsymptoteExpansionParameter();
        Real contraction_parameter = this->getMovingAsymptoteContractionParameter();
        for(size_t index = 0; index < number_controls; ++index)
        {
            Real value = ((*m_DataMng->m_CurrentControl)[index] - (*m_DataMng->m_PreviousControl)[index])
                           * ((*m_DataMng->m_PreviousControl)[index] - (*m_ControlAtIterationIminusTwo)[index]);
            if(value > 0)
            {
                (*m_DataMng->m_CurrentSigma)[index] = expansion_parameter * (*m_OldSigma)[index];
            }
            else if(value < 0)
            {
                (*m_DataMng->m_CurrentSigma)[index] = contraction_parameter * (*m_OldSigma)[index];
            }
            else
            {
                (*m_DataMng->m_CurrentSigma)[index] = (*m_OldSigma)[index];
            }
            // check that lower bound is satisfied
            Real lower_bound_scale = this->getMovingAsymptoteLowerBoundScale();
            value = lower_bound_scale
                    * ((*m_DataMng->m_ControlUpperBound)[index] - (*m_DataMng->m_ControlLowerBound)[index]);
            (*m_DataMng->m_CurrentSigma)[index] = std::max(value, (*m_DataMng->m_CurrentSigma)[index]);
            // check that upper bound is satisfied
            Real upper_bound_scale = this->getMovingAsymptoteUpperBoundScale();
            value = upper_bound_scale
                    * ((*m_DataMng->m_ControlUpperBound)[index] - (*m_DataMng->m_ControlLowerBound)[index]);
            (*m_DataMng->m_CurrentSigma)[index] = std::min(value, (*m_DataMng->m_CurrentSigma)[index]);
        }
    }
}

bool DOTk_AlgorithmCCSA::stoppingCriteriaSatisfied()
{
    bool criteria_satisfied = false;

    Real residual_norm = dotk::ccsa::computeResidualNorm(m_DataMng->m_CurrentControl, m_DataMng->m_Dual, m_DataMng);
    this->setCurrentResidualNorm(residual_norm);

    m_DataMng->m_WorkVector->update(1., *m_DataMng->m_CurrentControl, 0.);
    m_DataMng->m_WorkVector->update(-1., *m_DataMng->m_PreviousControl, 1.);
    Real control_stagnation_measure = m_DataMng->m_WorkVector->norm();
    this->setCurrentControlStagnationMeasure(control_stagnation_measure);

    m_DataMng->m_WorkVector->update(1., *m_DataMng->m_CurrentObjectiveGradient, 0.);
    m_Bounds->pruneActive(*m_DataMng->m_ActiveSet, *m_DataMng->m_WorkVector);
    Real optimality_gradient_norm = m_DataMng->m_WorkVector->norm();
    Real feasibility_measure = m_DataMng->m_CurrentFeasibilityMeasures->max();
    this->setCurrentMaxFeasibilityMeasure(feasibility_measure);
    Real max_residual = m_DataMng->m_CurrentInequalityResiduals->max();
    this->setCurrentMaxResidual(max_residual);

    if(this->getIterationCount() < 1)
    {
        m_InitialNormObjectiveGradient = optimality_gradient_norm;
    }

    Real relative_optimality_gradient_norm = optimality_gradient_norm / m_InitialNormObjectiveGradient;
    this->setCurrentObjectiveGradientNorm(relative_optimality_gradient_norm);
    bool optimality_reached = relative_optimality_gradient_norm < dotk::DOTk_AlgorithmCCSA::getGradientTolerance();
    bool feasibility_reached = feasibility_measure < dotk::DOTk_AlgorithmCCSA::getFeasibilityTolerance();

    if(residual_norm < dotk::DOTk_AlgorithmCCSA::getResidualTolerance())
    {
        criteria_satisfied = true;
        dotk::DOTk_AlgorithmCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::RESIDUAL_TOLERANCE);
    }
    else if(control_stagnation_measure < dotk::DOTk_AlgorithmCCSA::getControlStagnationTolerance())
    {
        criteria_satisfied = true;
        dotk::DOTk_AlgorithmCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::CONTROL_STAGNATION);
    }
    else if(optimality_reached && feasibility_reached)
    {
        criteria_satisfied = true;
        dotk::DOTk_AlgorithmCCSA::setStoppingCriterion(dotk::ccsa::stopping_criterion_t::OPTIMALITY_AND_FEASIBILITY_MET);
    }
    return (criteria_satisfied);
}

}
