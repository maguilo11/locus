/*
 * DOTk_MexAlgorithmTypeNewton.hpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXALGORITHMTYPENEWTON_HPP_
#define DOTK_MEXALGORITHMTYPENEWTON_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
class DOTk_InexactNewtonAlgorithms;

class DOTk_MexAlgorithmTypeNewton
{
public:
    explicit DOTk_MexAlgorithmTypeNewton(const mxArray* options_);
    virtual ~DOTk_MexAlgorithmTypeNewton();

    size_t getNumControls() const;
    size_t getMaxNumAlgorithmItr() const;
    double getGradientTolerance() const;
    double getTrialStepTolerance() const;
    double getObjectiveFunctionTolerance() const;
    double getKrylovSolverRelativeTolerance() const;
    dotk::types::problem_t getProblemType() const;

    void gatherOutputData(const dotk::DOTk_InexactNewtonAlgorithms & algorithm_,
                          const dotk::DOTk_OptimizationDataMng & mng_,
                          mxArray* output_[]);

    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;

private:
    void initialize(const mxArray* options_);

private:
    size_t m_NumControls;
    size_t m_MaxNumAlgorithmItr;
    double m_GradientTolerance;
    double m_TrialStepTolerance;
    double m_ObjectiveFunctionTolerance;
    double m_KrylovSolverRelativeTolerance;
    dotk::types::problem_t m_ProblemType;

private:
    DOTk_MexAlgorithmTypeNewton(const dotk::DOTk_MexAlgorithmTypeNewton&);
    dotk::DOTk_MexAlgorithmTypeNewton& operator=(const dotk::DOTk_MexAlgorithmTypeNewton&);
};

}

#endif /* DOTK_MEXALGORITHMTYPENEWTON_HPP_ */
