/*
 * DOTk_MexAlgorithmTypeSQP.hpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXALGORITHMTYPESQP_HPP_
#define DOTK_MEXALGORITHMTYPESQP_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_MexAlgorithmTypeSQP
{
public:
    explicit DOTk_MexAlgorithmTypeSQP(const mxArray* options_);
    virtual ~DOTk_MexAlgorithmTypeSQP();

    size_t getMaxNumAlgorithmItr() const;

    double getGradientTolerance() const;
    double getTrialStepTolerance() const;
    double getOptimalityTolerance() const;
    double getFeasibilityTolerance() const;

    dotk::types::problem_t getProblemType() const;
    dotk::types::hessian_t getHessianComputationMethod() const;

    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;

private:
    void initialize(const mxArray* options_);

private:
    size_t m_MaxNumAlgorithmItr;

    double m_GradientTolerance;
    double m_TrialStepTolerance;
    double m_FeasibilityTolerance;

    dotk::types::problem_t m_ProblemType;
    dotk::types::hessian_t m_HessianComputationMethod;

private:
    DOTk_MexAlgorithmTypeSQP(const dotk::DOTk_MexAlgorithmTypeSQP&);
    dotk::DOTk_MexAlgorithmTypeSQP& operator=(const dotk::DOTk_MexAlgorithmTypeSQP&);
};

}

#endif /* DOTK_MEXALGORITHMTYPESQP_HPP_ */
