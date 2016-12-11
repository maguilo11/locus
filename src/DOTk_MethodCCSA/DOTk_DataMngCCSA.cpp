/*
 * DOTk_DataMngCCSA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cstdio>
#include <limits>
#include <cstdlib>
#include <algorithm>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_DataMngCCSA.hpp"

#include "DOTk_RoutinesTypeLP.hpp"
#include "DOTk_RoutinesTypeNP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

DOTk_DataMngCCSA::DOTk_DataMngCCSA(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                   const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_) :
        m_ObjectiveCoefficientsA(1),
        m_InitialAuxiliaryVariableZ(0),
        m_CurrentObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_Dual(primal_->dual()->clone()),
        m_MinRho(primal_->dual()->clone()),
        m_ActiveSet(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_CurrentSigma(primal_->control()->clone()),
        m_CurrentControl(primal_->control()->clone()),
        m_PreviousControl(primal_->control()->clone()),
        m_ControlLowerBound(primal_->control()->clone()),
        m_ControlUpperBound(primal_->control()->clone()),
        m_CurrentObjectiveGradient(primal_->control()->clone()),
        m_CurrentFeasibilityMeasures(primal_->dual()->clone()),
        m_CurrentInequalityResiduals(primal_->dual()->clone()),
        m_CurrentInequalityGradients(),
        m_InputAuxiliaryVariablesY(primal_->dual()->clone()),
        m_InputInequalityCoefficientsA(primal_->dual()->clone()),
        m_InputInequalityCoefficientsC(primal_->dual()->clone()),
        m_InputInequalityCoefficientsD(primal_->dual()->clone()),
        m_Primal(primal_),
        m_NumInequalityConstraints(inequality_.size()),
        m_AssemblyMng(new dotk::DOTk_RoutinesTypeLP(objective_, inequality_))
{
    this->initialize(primal_);
}

DOTk_DataMngCCSA::DOTk_DataMngCCSA(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                   const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                                   const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_) :
        m_ObjectiveCoefficientsA(1),
        m_InitialAuxiliaryVariableZ(0),
        m_CurrentObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_Dual(primal_->dual()->clone()),
        m_MinRho(primal_->dual()->clone()),
        m_ActiveSet(primal_->control()->clone()),
        m_WorkVector(primal_->control()->clone()),
        m_CurrentSigma(primal_->control()->clone()),
        m_CurrentControl(primal_->control()->clone()),
        m_PreviousControl(primal_->control()->clone()),
        m_ControlLowerBound(primal_->control()->clone()),
        m_ControlUpperBound(primal_->control()->clone()),
        m_CurrentObjectiveGradient(primal_->control()->clone()),
        m_CurrentFeasibilityMeasures(primal_->dual()->clone()),
        m_CurrentInequalityResiduals(primal_->dual()->clone()),
        m_CurrentInequalityGradients(),
        m_InputAuxiliaryVariablesY(primal_->dual()->clone()),
        m_InputInequalityCoefficientsA(primal_->dual()->clone()),
        m_InputInequalityCoefficientsC(primal_->dual()->clone()),
        m_InputInequalityCoefficientsD(primal_->dual()->clone()),
        m_Primal(primal_),
        m_NumInequalityConstraints(inequality_.size()),
        m_AssemblyMng(new dotk::DOTk_RoutinesTypeNP(primal_, objective_, equality_, inequality_))
{
    this->initialize(primal_);
}

DOTk_DataMngCCSA::~DOTk_DataMngCCSA()
{
}

size_t DOTk_DataMngCCSA::getNumberInequalityConstraints() const
{
    return (m_NumInequalityConstraints);
}

size_t DOTk_DataMngCCSA::getObjectiveFunctionEvaluationCounter() const
{
    return (m_AssemblyMng->getObjectiveFunctionEvaluationCounter());
}

size_t DOTk_DataMngCCSA::getGradientEvaluationCounter() const
{
    return (m_AssemblyMng->getGradientEvaluationCounter());
}

size_t DOTk_DataMngCCSA::getInequalityConstraintGradientCounter() const
{
    return (m_AssemblyMng->getInequalityConstraintGradientCounter());
}

void DOTk_DataMngCCSA::computeFunctionGradients()
{
    m_AssemblyMng->gradient(m_CurrentControl, m_CurrentObjectiveGradient);

    for(size_t index = 0; index < m_NumInequalityConstraints; ++index)
    {
        m_AssemblyMng->inequalityGradient(index, m_CurrentControl, m_CurrentInequalityGradients->basis(index));
    }
}

void DOTk_DataMngCCSA::evaluateFunctionValues()
{
    m_CurrentObjectiveFunctionValue = m_AssemblyMng->objective(m_CurrentControl);
    for(size_t index = 0; index < m_NumInequalityConstraints; ++index)
    {
        (*m_CurrentInequalityResiduals)[index] = m_AssemblyMng->inequalityValue(index, m_CurrentControl)
                - m_AssemblyMng->inequalityBound(index);
        (*m_CurrentFeasibilityMeasures)[index] = std::abs((*m_CurrentInequalityResiduals)[index])
                / m_AssemblyMng->inequalityBound(index);
    }
}

void DOTk_DataMngCCSA::evaluateInequalityConstraintResiduals()
{
    for(size_t index = 0; index < m_NumInequalityConstraints; ++index)
    {
        (*m_CurrentInequalityResiduals)[index] = m_AssemblyMng->inequalityValue(index, m_CurrentControl)
                - m_AssemblyMng->inequalityBound(index);
    }
}

void DOTk_DataMngCCSA::evaluateObjectiveFunction()
{
    m_CurrentObjectiveFunctionValue = m_AssemblyMng->objective(m_CurrentControl);
}

void DOTk_DataMngCCSA::initializeAuxiliaryVariables()
{
    Real max_a_coeff_value = m_InputInequalityCoefficientsA->max();
    if(max_a_coeff_value > static_cast<Real>(0))
    {
        std::tr1::shared_ptr<dotk::Vector<Real> > initial_candidates_for_auxiliary_vars_z =
                m_InputInequalityCoefficientsA->clone();
        for(size_t index = 0; index < m_NumInequalityConstraints; ++index)
        {
            if((*m_InputInequalityCoefficientsA)[index] > 0)
            {
                (*m_InputAuxiliaryVariablesY)[index] = 0;
                Real value = m_AssemblyMng->inequalityValue(index, m_CurrentControl)
                        - m_AssemblyMng->inequalityBound(index);
                (*initial_candidates_for_auxiliary_vars_z)[index] = std::max(0., value)
                        / (*m_InputInequalityCoefficientsA)[index];
            }
            else
            {
                Real value = m_AssemblyMng->inequalityValue(index, m_CurrentControl)
                        - m_AssemblyMng->inequalityBound(index);
                (*m_InputAuxiliaryVariablesY)[index] = std::max(0., value);
            }
        }
        m_InitialAuxiliaryVariableZ = initial_candidates_for_auxiliary_vars_z->max();
    }
    else
    {
        for(size_t index = 0; index < m_NumInequalityConstraints; ++index)
        {
            (*m_InputAuxiliaryVariablesY)[index] = m_AssemblyMng->inequalityValue(index, m_CurrentControl)
                    - m_AssemblyMng->inequalityBound(index);
        }
        m_InitialAuxiliaryVariableZ = 0.;
    }
}

Real DOTk_DataMngCCSA::evaluateObjectiveFunction(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_)
{
    Real value = m_AssemblyMng->objective(primal_);
    return (value);
}

void DOTk_DataMngCCSA::evaluateInequalityConstraints(const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                                                     const std::tr1::shared_ptr<dotk::Vector<Real> > & residual_,
                                                     const std::tr1::shared_ptr<dotk::Vector<Real> > & feasibility_measure_)
{
    for(size_t index = 0; index < m_NumInequalityConstraints; ++index)
    {
        (*residual_)[index] = m_AssemblyMng->inequalityValue(index, control_)
                - m_AssemblyMng->inequalityBound(index);
        (*feasibility_measure_)[index] = std::abs((*residual_)[index]) / m_AssemblyMng->inequalityBound(index);
    }
}

void DOTk_DataMngCCSA::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    this->checkInputs(primal_);

    m_Dual->update(1., *primal_->dual(), 0.);
    m_CurrentControl->update(1., *primal_->control(), 0.);
    m_ControlLowerBound->update(1., *primal_->getControlLowerBound(), 0.);
    m_ControlUpperBound->update(1., *primal_->getControlUpperBound(), 0.);

    m_MinRho->fill(1e-5);
    m_InputInequalityCoefficientsA->fill(0.);
    m_InputInequalityCoefficientsC->fill(1e3);
    m_InputInequalityCoefficientsD->fill(1.);
    m_CurrentFeasibilityMeasures->fill(std::numeric_limits<Real>::max());

    this->initializeAuxiliaryVariables();
    m_CurrentInequalityGradients.reset(new dotk::serial::DOTk_RowMatrix<Real>
        (*primal_->control(), m_NumInequalityConstraints));
}

void DOTk_DataMngCCSA::checkInitialAuxiliaryVariables()
{
    Real min_a_coeff_value = m_InputInequalityCoefficientsA->min();
    Real min_c_coeff_value = m_InputInequalityCoefficientsC->min();
    Real min_d_coeff_value = m_InputInequalityCoefficientsD->min();
    if(min_a_coeff_value < static_cast<Real>(0.))
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInitialAuxiliaryVariables -> Negative A coefficients detected. ABORT ****\n");
        std::abort();
    }
    else if(min_c_coeff_value < static_cast<Real>(0.))
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInitialAuxiliaryVariables -> Negative C coefficients detected. ABORT ****\n");
        std::abort();
    }
    else if(min_d_coeff_value < static_cast<Real>(0.))
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInitialAuxiliaryVariables -> Negative D coefficients detected. ABORT ****\n");
        std::abort();
    }
}

void DOTk_DataMngCCSA::checkInputs(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    assert(primal_->dual()->size() == m_NumInequalityConstraints);

    if(primal_->dual().use_count() < 1)
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInputs -> User did not define dual variables. ABORT ****\n");
        std::abort();
    }
    else if(primal_->control().use_count() < 1)
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInputs -> User did not define control variables. ABORT ****\n");
        std::abort();
    }
    else if(primal_->getControlLowerBound().use_count() < 1)
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInputs -> User did not define control lower bounds. ABORT ****\n");
        std::abort();
    }
    else if(primal_->getControlUpperBound().use_count() < 1)
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_DataMngCCSA::checkInputs -> User did not define control upper bounds. ABORT ****\n");
        std::abort();
    }
    this->checkInitialAuxiliaryVariables();
}

}
