/*
 * DOTk_MexAlgorithmTypeSQP.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cstddef>

#include "vector.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexAlgorithmTypeSQP.hpp"

namespace dotk
{

DOTk_MexAlgorithmTypeSQP::DOTk_MexAlgorithmTypeSQP(const mxArray* options_) :
        m_MaxNumAlgorithmItr(50),
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_FeasibilityTolerance(1e-12),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_HessianComputationMethod(dotk::types::HESSIAN_DISABLED)
{
    this->initialize(options_);
}

DOTk_MexAlgorithmTypeSQP::~DOTk_MexAlgorithmTypeSQP()
{
}

size_t DOTk_MexAlgorithmTypeSQP::getMaxNumAlgorithmItr() const
{
    return (m_MaxNumAlgorithmItr);
}

double DOTk_MexAlgorithmTypeSQP::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

double DOTk_MexAlgorithmTypeSQP::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

double DOTk_MexAlgorithmTypeSQP::getFeasibilityTolerance() const
{
    return (m_FeasibilityTolerance);
}

dotk::types::problem_t DOTk_MexAlgorithmTypeSQP::getProblemType() const
{
    return (m_ProblemType);
}

dotk::types::hessian_t DOTk_MexAlgorithmTypeSQP::getHessianComputationMethod() const
{
    return (m_HessianComputationMethod);
}

void DOTk_MexAlgorithmTypeSQP::initialize(const mxArray* options_)
{
    m_ProblemType = dotk::mex::parseProblemType(options_);
    m_TrialStepTolerance = dotk::mex::parseStepTolerance(options_);
    m_GradientTolerance = dotk::mex::parseGradientTolerance(options_);
    m_MaxNumAlgorithmItr = dotk::mex::parseMaxNumOuterIterations(options_);
    m_FeasibilityTolerance = dotk::mex::parseFeasibilityTolerance(options_);
    m_HessianComputationMethod = dotk::mex::parseHessianComputationMethod(options_);
}

}
