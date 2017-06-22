/*
 * DOTk_DataMngCCSA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DATAMNGCCSA_HPP_
#define DOTK_DATAMNGCCSA_HPP_

#include <vector>
#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_AssemblyManager;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;
template<typename ScalarType>
class DOTk_InequalityConstraint;

class DOTk_DataMngCCSA
{
public:
    DOTk_DataMngCCSA(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                     const std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    DOTk_DataMngCCSA(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                     const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                     const std::vector<std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    virtual ~DOTk_DataMngCCSA();

    size_t getNumberInequalityConstraints() const;
    size_t getObjectiveFunctionEvaluationCounter() const;
    size_t getGradientEvaluationCounter() const;
    size_t getInequalityConstraintGradientCounter() const;

    void computeFunctionGradients();
    void evaluateFunctionValues();
    void evaluateInequalityConstraintResiduals();
    void evaluateObjectiveFunction();
    void initializeAuxiliaryVariables();

    Real evaluateObjectiveFunction(const std::shared_ptr<dotk::Vector<Real> > & primal_);
    void evaluateInequalityConstraints(const std::shared_ptr<dotk::Vector<Real> > & control_,
                                       const std::shared_ptr<dotk::Vector<Real> > & residual_,
                                       const std::shared_ptr<dotk::Vector<Real> > & feasibility_measure_);

public:
    Real m_ObjectiveCoefficientsA;
    Real m_InitialAuxiliaryVariableZ;
    Real m_CurrentObjectiveFunctionValue;

    std::shared_ptr<dotk::Vector<Real> > m_Dual;
    std::shared_ptr<dotk::Vector<Real> > m_MinRho;
    std::shared_ptr<dotk::Vector<Real> > m_ActiveSet;
    std::shared_ptr<dotk::Vector<Real> > m_WorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentSigma;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentControl;
    std::shared_ptr<dotk::Vector<Real> > m_PreviousControl;
    std::shared_ptr<dotk::Vector<Real> > m_ControlLowerBound;
    std::shared_ptr<dotk::Vector<Real> > m_ControlUpperBound;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentObjectiveGradient;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentFeasibilityMeasures;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentInequalityResiduals;
    std::shared_ptr<dotk::matrix<Real> > m_CurrentInequalityGradients;
    std::shared_ptr<dotk::Vector<Real> > m_InputAuxiliaryVariablesY;
    std::shared_ptr<dotk::Vector<Real> > m_InputInequalityCoefficientsA;
    std::shared_ptr<dotk::Vector<Real> > m_InputInequalityCoefficientsC;
    std::shared_ptr<dotk::Vector<Real> > m_InputInequalityCoefficientsD;

    std::shared_ptr<dotk::DOTk_Primal> m_Primal;

private:
    void initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void checkInitialAuxiliaryVariables();
    void checkInputs(const std::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    size_t m_NumInequalityConstraints;
    std::shared_ptr<dotk::DOTk_AssemblyManager> m_AssemblyMng;

private:
    DOTk_DataMngCCSA(const dotk::DOTk_DataMngCCSA &);
    dotk::DOTk_DataMngCCSA & operator=(const dotk::DOTk_DataMngCCSA & rhs_);
};

}

#endif /* DOTK_DATAMNGCCSA_HPP_ */
