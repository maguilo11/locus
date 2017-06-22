/*
 * DOTk_MexMethodCCSA.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXMETHODCCSA_HPP_
#define DOTK_MEXMETHODCCSA_HPP_

#include <mex.h>
#include <memory>

#include "DOTk_Types.hpp"
#include "DOTk_UtilsCCSA.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;
class DOTk_AlgorithmCCSA;
class DOTk_DualSolverNLCG;

class DOTk_MexMethodCCSA
{
public:
    explicit DOTk_MexMethodCCSA(const mxArray* options_);
    virtual ~DOTk_MexMethodCCSA();

    dotk::types::problem_t getProblemType() const;
    dotk::ccsa::dual_solver_t getDualSolverType() const;
    dotk::types::nonlinearcg_t getNonlinearConjugateGradientType() const;

    void setPrimalSolverParameters(dotk::DOTk_AlgorithmCCSA & primal_solver_);
    void setDualSolverParameters(const std::shared_ptr<dotk::DOTk_DualSolverNLCG> & dual_solver_);
    void gatherOutputData(dotk::DOTk_AlgorithmCCSA & algorithm_,
                          const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                          mxArray* output_[]);

    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;

private:
    void initialize(const mxArray* options_);

private:
    size_t m_DualSolverMaxNumberIterations;
    size_t m_PrimalSolverMaxNumberIterations;
    size_t m_DualSolverMaxNumberLineSearchIterations;

    double m_GradientTolerance;
    double m_ResidualTolerance;
    double m_FeasibilityTolerance;
    double m_ControlStagnationTolerance;

    double m_MovingAsymptoteUpperBoundScale;
    double m_MovingAsymptoteLowerBoundScale;
    double m_MovingAsymptoteExpansionParameter;
    double m_MovingAsymptoteContractionParameter;

    double m_DualSolverGradientTolerance;
    double m_DualSolverTrialStepTolerance;
    double m_DualObjectiveEpsilonParameter;
    double m_DualSolverLineSearchStepLowerBound;
    double m_DualSolverLineSearchStepUpperBound;
    double m_DualObjectiveTrialControlBoundScaling;
    double m_DualSolverObjectiveStagnationTolerance;

    dotk::types::problem_t m_ProblemType;
    dotk::ccsa::dual_solver_t m_DualSolverType;
    dotk::types::nonlinearcg_t m_DualSolverTypeNLCG;

private:
    DOTk_MexMethodCCSA(const dotk::DOTk_MexMethodCCSA & rhs_);
    dotk::DOTk_MexMethodCCSA& operator=(const dotk::DOTk_MexMethodCCSA & rhs_);
};

}

#endif /* DOTK_MEXMETHODCCSA_HPP_ */
