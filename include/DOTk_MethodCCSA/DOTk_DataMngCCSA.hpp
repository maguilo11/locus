/*
 * DOTk_DataMngCCSA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DATAMNGCCSA_HPP_
#define DOTK_DATAMNGCCSA_HPP_

#include <vector>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_AssemblyManager;

template<typename Type>
class vector;
template<typename Type>
class matrix;
template<typename Type>
class DOTk_ObjectiveFunction;
template<typename Type>
class DOTk_EqualityConstraint;
template<typename Type>
class DOTk_InequalityConstraint;

class DOTk_DataMngCCSA
{
public:
    DOTk_DataMngCCSA(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                     const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    DOTk_DataMngCCSA(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                     const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                     const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
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

    Real evaluateObjectiveFunction(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);
    void evaluateInequalityConstraints(const std::tr1::shared_ptr<dotk::vector<Real> > & control_,
                                       const std::tr1::shared_ptr<dotk::vector<Real> > & residual_,
                                       const std::tr1::shared_ptr<dotk::vector<Real> > & feasibility_measure_);

public:
    Real m_ObjectiveCoefficientsA;
    Real m_InitialAuxiliaryVariableZ;
    Real m_CurrentObjectiveFunctionValue;

    std::tr1::shared_ptr<dotk::vector<Real> > m_Dual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_MinRho;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ActiveSet;
    std::tr1::shared_ptr<dotk::vector<Real> > m_WorkVector;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CurrentSigma;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CurrentControl;
    std::tr1::shared_ptr<dotk::vector<Real> > m_PreviousControl;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlLowerBound;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlUpperBound;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CurrentObjectiveGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CurrentFeasibilityMeasures;
    std::tr1::shared_ptr<dotk::vector<Real> > m_CurrentInequalityResiduals;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_CurrentInequalityGradients;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InputAuxiliaryVariablesY;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InputInequalityCoefficientsA;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InputInequalityCoefficientsC;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InputInequalityCoefficientsD;

    std::tr1::shared_ptr<dotk::DOTk_Primal> m_Primal;

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    void checkInitialAuxiliaryVariables();
    void checkInputs(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    size_t m_NumInequalityConstraints;
    std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> m_AssemblyMng;

private:
    DOTk_DataMngCCSA(const dotk::DOTk_DataMngCCSA &);
    dotk::DOTk_DataMngCCSA & operator=(const dotk::DOTk_DataMngCCSA & rhs_);
};

}

#endif /* DOTK_DATAMNGCCSA_HPP_ */
