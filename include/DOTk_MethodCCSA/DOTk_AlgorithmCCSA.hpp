/*
 * DOTk_AlgorithmCCSA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef INCLUDE_DOTK_METHODCCSA_DOTK_ALGORITHMCCSA_HPP_
#define INCLUDE_DOTK_METHODCCSA_DOTK_ALGORITHMCCSA_HPP_

#include <tr1/memory>

#include "DOTk_UtilsCCSA.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;
class DOTK_MethodCcsaIO;
class DOTk_SubProblemCCSA;
class DOTk_BoundConstraints;

template<class Type>
class vector;

class DOTk_AlgorithmCCSA
{
    // Conservative Convex Separable Approximations (CCSA) method main algorithmic driver
public:
    DOTk_AlgorithmCCSA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                       const std::tr1::shared_ptr<dotk::DOTk_SubProblemCCSA> & sub_problem_);
    ~DOTk_AlgorithmCCSA();

    dotk::ccsa::stopping_criterion_t getStoppingCriterion() const;
    void setStoppingCriterion(dotk::ccsa::stopping_criterion_t type_);

    size_t getIterationCount() const;
    size_t getMaxNumIterations() const;
    void setMaxNumIterations(size_t max_num_iterations_);

    Real getResidualTolerance() const;
    void setResidualTolerance(Real tolerance_);
    Real getGradientTolerance() const;
    void setGradientTolerance(Real tolerance_);
    Real getCurrentMaxResidual() const;
    void setCurrentMaxResidual(Real input_);
    Real getOptimalityTolerance() const;
    void setOptimalityTolerance(Real tolerance_);
    Real getCurrentResidualNorm() const;
    void setCurrentResidualNorm(Real input_);
    Real getFeasibilityTolerance() const;
    void setFeasibilityTolerance(Real tolerance_);
    Real getControlStagnationTolerance() const;
    void setControlStagnationTolerance(Real tolerance_);
    Real getCurrentMaxFeasibilityMeasure() const;
    void setCurrentMaxFeasibilityMeasure(Real input_);
    Real getCurrentObjectiveGradientNorm() const;
    void setCurrentObjectiveGradientNorm(Real input_);
    Real getMovingAsymptoteUpperBoundScale() const;
    void setMovingAsymptoteUpperBoundScale(Real input_);
    Real getMovingAsymptoteLowerBoundScale() const;
    void setMovingAsymptoteLowerBoundScale(Real input_);
    Real getCurrentControlStagnationMeasure() const;
    void setCurrentControlStagnationMeasure(Real input_);
    Real getMovingAsymptoteExpansionParameter() const;
    void setMovingAsymptoteExpansionParameter(Real input_);
    Real getMovingAsymptoteContractionParameter() const;
    void setMovingAsymptoteContractionParameter(Real input_);

    void setMaxNumberSubProblemIterations(size_t input_);
    void setDualObjectiveEpsilonParameter(Real input_);
    void setDualObjectiveTrialControlBoundScaling(Real input_);

    void printDiagnosticsAndSolutionAtEveryItr();
    void printDiagnosticsAtEveryItrAndSolutionAtTheEnd();

    void getMin();

private:
    void updateIterationCount();
    void updateSigmaParameters();
    bool stoppingCriteriaSatisfied();

private:
    dotk::ccsa::stopping_criterion_t m_StoppingCriterion;

    size_t m_IterationCount;
    size_t m_MaxNumIterations;

    Real m_ResidualTolerance;
    Real m_GradientTolerance;
    Real m_CurrentMaxResidual;
    Real m_OptimalityTolerance;
    Real m_CurrentResidualNorm;
    Real m_FeasibilityTolerance;
    Real m_ControlStagnationTolerance;
    Real m_CurrentMaxFeasibilityMeasure;
    Real m_CurrentObjectiveGradientNorm;
    Real m_InitialNormObjectiveGradient;
    Real m_MovingAsymptoteUpperBoundScale;
    Real m_MovingAsymptoteLowerBoundScale;
    Real m_CurrentControlStagnationMeasure;
    Real m_MovingAsymptoteExpansionParameter;
    Real m_MovingAsymptoteContractionParameter;

    std::tr1::shared_ptr<dotk::vector<Real> > m_OldSigma;
    std::tr1::shared_ptr<dotk::vector<Real> > m_AuxiliaryZcandidates;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlAtIterationIminusTwo;

    std::tr1::shared_ptr<dotk::DOTK_MethodCcsaIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> m_DataMng;
    std::tr1::shared_ptr<dotk::DOTk_BoundConstraints> m_Bounds;
    std::tr1::shared_ptr<dotk::DOTk_SubProblemCCSA> m_SubProblem;

private:
    DOTk_AlgorithmCCSA(const dotk::DOTk_AlgorithmCCSA &);
    dotk::DOTk_AlgorithmCCSA & operator=(const dotk::DOTk_AlgorithmCCSA & rhs_);
};

}

#endif /* INCLUDE_DOTK_METHODCCSA_DOTK_ALGORITHMCCSA_HPP_ */
