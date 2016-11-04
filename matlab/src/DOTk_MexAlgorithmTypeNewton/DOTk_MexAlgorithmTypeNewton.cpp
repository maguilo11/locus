/*
 * DOTk_MexAlgorithmTypeNewton.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexAlgorithmTypeNewton.hpp"
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
    dotk::mex::parseProblemType(options_, m_ProblemType);
    dotk::mex::parseNumberControls(options_, m_NumControls);
    dotk::mex::parseGradientTolerance(options_, m_GradientTolerance);
    dotk::mex::parseMaxNumAlgorithmItr(options_, m_MaxNumAlgorithmItr);
    dotk::mex::parseTrialStepTolerance(options_, m_TrialStepTolerance);
    dotk::mex::parseOptimalityTolerance(options_, m_ObjectiveFunctionTolerance);
    dotk::mex::parseKrylovSolverRelativeTolerance(options_, m_KrylovSolverRelativeTolerance);
}

void DOTk_MexAlgorithmTypeNewton::gatherOutputData(const dotk::DOTk_InexactNewtonAlgorithms & algorithm_,
                                                   const dotk::DOTk_OptimizationDataMng & mng_,
                                                   mxArray* output_[])
{
    // Create memory allocation for output struct
    const char *field_names[7] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "Gradient", "NormGradient", "TrialStep", "NormTrialStep" };
    output_[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    dotk::DOTk_MexArrayPtr itr(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(itr.get()))[0] = algorithm_.getNumItrDone();
    mxSetField(output_[0], 0, "Iterations", itr.get());

    dotk::DOTk_MexArrayPtr objective_function_value(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(objective_function_value.get())[0] = mng_.getNewObjectiveFunctionValue();
    mxSetField(output_[0], 0, "ObjectiveFunctionValue", objective_function_value.get());

    size_t num_controls = this->getNumControls();
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getNewPrimal()->gather(mxGetPr(primal.get()));
    mxSetField(output_[0], 0, "Control", primal.get());

    dotk::DOTk_MexArrayPtr gradient(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getNewGradient()->gather(mxGetPr(gradient.get()));
    mxSetField(output_[0], 0, "Gradient", gradient.get());

    dotk::DOTk_MexArrayPtr norm_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_gradient.get())[0] = mng_.getNewGradient()->norm();
    mxSetField(output_[0], 0, "NormGradient", norm_gradient.get());

    dotk::DOTk_MexArrayPtr trial_step(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getTrialStep()->gather(mxGetPr(trial_step.get()));
    mxSetField(output_[0], 0, "TrialStep", trial_step.get());

    dotk::DOTk_MexArrayPtr norm_trial_step(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_trial_step.get())[0] = mng_.getTrialStep()->norm();
    mxSetField(output_[0], 0, "NormTrialStep", norm_trial_step.get());

    itr.release();
    objective_function_value.release();
    primal.release();
    gradient.release();
    norm_gradient.release();
    trial_step.release();
    norm_trial_step.release();
}

}
