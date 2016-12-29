/*
 * DOTk_MexAlgorithmTypeNewton.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexVector.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexAlgorithmTypeNewton.hpp"

#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_InexactNewtonAlgorithms.hpp"

namespace dotk
{

DOTk_MexAlgorithmTypeNewton::DOTk_MexAlgorithmTypeNewton(const mxArray* options_) :
        m_NumControls(0),
        m_MaxNumAlgorithmItr(100),
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_ObjectiveFunctionTolerance(1e-10),
        m_KrylovSolverRelativeTolerance(1e-10),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED)
{
    this->initialize(options_);
}

DOTk_MexAlgorithmTypeNewton::~DOTk_MexAlgorithmTypeNewton()
{
}

size_t DOTk_MexAlgorithmTypeNewton::getNumControls() const
{
    return (m_NumControls);
}

size_t DOTk_MexAlgorithmTypeNewton::getMaxNumAlgorithmItr() const
{
    return (m_MaxNumAlgorithmItr);
}

double DOTk_MexAlgorithmTypeNewton::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

double DOTk_MexAlgorithmTypeNewton::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

double DOTk_MexAlgorithmTypeNewton::getObjectiveFunctionTolerance() const
{
    return (m_ObjectiveFunctionTolerance);
}

double DOTk_MexAlgorithmTypeNewton::getKrylovSolverRelativeTolerance() const
{
    return (m_KrylovSolverRelativeTolerance);
}

dotk::types::problem_t DOTk_MexAlgorithmTypeNewton::getProblemType() const
{
    return (m_ProblemType);
}

void DOTk_MexAlgorithmTypeNewton::initialize(const mxArray* options_)
{
    m_ProblemType = dotk::mex::parseProblemType(options_);
    m_NumControls = dotk::mex::parseNumberControls(options_);
    m_TrialStepTolerance = dotk::mex::parseStepTolerance(options_);
    m_GradientTolerance = dotk::mex::parseGradientTolerance(options_);
    m_MaxNumAlgorithmItr = dotk::mex::parseMaxNumOuterIterations(options_);
    m_ObjectiveFunctionTolerance = dotk::mex::parseObjectiveTolerance(options_);
    m_KrylovSolverRelativeTolerance = dotk::mex::parseKrylovSolverRelativeTolerance(options_);
}

void DOTk_MexAlgorithmTypeNewton::gatherOutputData(const dotk::DOTk_InexactNewtonAlgorithms & algorithm_,
                                                   const dotk::DOTk_OptimizationDataMng & mng_,
                                                   mxArray* outputs_[])
{
    // Create memory allocation for output struct
    const char *field_names[7] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "Gradient", "NormGradient", "Step", "NormStep" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    /* NOTE: mxSetField does not create a copy of the data allocated. Thus,
     * mxDestroyArray cannot be called. Furthermore, MEX array data (e.g.
     * control, gradient, etc.) should be duplicated since the data in the
     * manager will be deallocated at the end. */
    mxArray* mx_number_iterations = mxCreateNumericMatrix(1, 1, mxINDEX_CLASS, mxREAL);
    static_cast<size_t*>(mxGetData(mx_number_iterations))[0] = algorithm_.getNumItrDone();
    mxSetField(outputs_[0], 0, "Iterations", mx_number_iterations);

    double value = mng_.getNewObjectiveFunctionValue();
    mxArray* mx_objective_function_value = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "ObjectiveFunctionValue", mx_objective_function_value);

    dotk::MexVector & control = dynamic_cast<dotk::MexVector &>(*mng_.getNewPrimal());
    mxArray* mx_control = mxDuplicateArray(control.array());
    mxSetField(outputs_[0], 0, "Control", mx_control);

    dotk::MexVector & gradient = dynamic_cast<dotk::MexVector &>(*mng_.getNewGradient());
    mxArray* mx_gradient = mxDuplicateArray(gradient.array());
    mxSetField(outputs_[0], 0, "Gradient", mx_gradient);

    value = gradient.norm();
    mxArray* mx_norm_gradient = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormGradient", mx_norm_gradient);

    dotk::MexVector & step = dynamic_cast<dotk::MexVector &>(*mng_.getTrialStep());
    mxArray* mx_step = mxDuplicateArray(step.array());
    mxSetField(outputs_[0], 0, "Step", mx_step);

    value = step.norm();
    mxArray* mx_norm_step = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormStep", mx_norm_step);
}

}
