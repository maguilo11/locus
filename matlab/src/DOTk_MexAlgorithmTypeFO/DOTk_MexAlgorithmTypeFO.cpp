/*
 * DOTk_MexAlgorithmTypeFO.cpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexAlgorithmTypeFO.hpp"
#include "DOTk_FirstOrderAlgorithm.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_MexAlgorithmTypeFO::DOTk_MexAlgorithmTypeFO(const mxArray* options_) :
        m_NumDuals(0),
        m_NumControls(0),
        m_MaxNumAlgorithmItr(100),
        m_MaxNumLineSearchItr(10),
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_OptimalityTolerance(1e-10),
        m_LineSearchContractionFactor(0.5),
        m_LineSearchStagnationTolerance(1e-8),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_LineSearchMethod(dotk::types::LINE_SEARCH_DISABLED)
{
    this->initialize(options_);
}

DOTk_MexAlgorithmTypeFO::~DOTk_MexAlgorithmTypeFO()
{
}

void DOTk_MexAlgorithmTypeFO::setNumDuals(const mxArray* options_)
{
    dotk::mex::parseNumberDuals(options_, m_NumDuals);
}

size_t DOTk_MexAlgorithmTypeFO::getNumDuals() const
{
    return (m_NumDuals);
}

void DOTk_MexAlgorithmTypeFO::setNumControls(const mxArray* options_)
{
    dotk::mex::parseNumberControls(options_, m_NumControls);
}

size_t DOTk_MexAlgorithmTypeFO::getNumControls() const
{
    return (m_NumControls);
}

size_t DOTk_MexAlgorithmTypeFO::getMaxNumAlgorithmItr() const
{
    return (m_MaxNumAlgorithmItr);
}

double DOTk_MexAlgorithmTypeFO::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

double DOTk_MexAlgorithmTypeFO::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

double DOTk_MexAlgorithmTypeFO::getOptimalityTolerance() const
{
    return (m_OptimalityTolerance);
}

size_t DOTk_MexAlgorithmTypeFO::getMaxNumLineSearchItr() const
{
    return (m_MaxNumLineSearchItr);
}

double DOTk_MexAlgorithmTypeFO::getLineSearchContractionFactor() const
{
    return (m_LineSearchContractionFactor);
}

double DOTk_MexAlgorithmTypeFO::getLineSearchStagnationTolerance() const
{
    return (m_LineSearchStagnationTolerance);
}

dotk::types::problem_t DOTk_MexAlgorithmTypeFO::getProblemType() const
{
    return (m_ProblemType);
}

dotk::types::line_search_t DOTk_MexAlgorithmTypeFO::getLineSearchMethod() const
{
    return (m_LineSearchMethod);
}

void DOTk_MexAlgorithmTypeFO::setBoundConstraintMethod(const mxArray* options_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> & step_)
{
    dotk::types::constraint_method_t type = dotk::types::CONSTRAINT_METHOD_DISABLED;
    dotk::mex::parseBoundConstraintMethod(options_, type);

    switch(type)
    {
        case dotk::types::FEASIBLE_DIR:
        {
            size_t iterations = 0;
            dotk::mex::parseMaxNumFeasibleItr(options_, iterations);
            double factor = 0.;
            dotk::mex::parseBoundConstraintContractionFactor(options_, factor);
            step_->setFeasibleDirectionConstraint(primal_);
            step_->setMaxNumFeasibleItr(iterations);
            step_->setBoundConstraintMethodContractionStep(factor);
            break;
        }
        case dotk::types::PROJECTION_ALONG_FEASIBLE_DIR:
        {
            double factor = 0.;
            dotk::mex::parseBoundConstraintContractionFactor(options_, factor);
            step_->setBoundConstraintMethodContractionStep(factor);
            step_->setProjectionAlongFeasibleDirConstraint(primal_);
            break;
        }
        default:
        {
            step_->setProjectionAlongFeasibleDirConstraint(primal_);
            std::string msg(" DOTk/MEX WARNING: Invalid Bound Constraint Method. Default = PROJECTION ALONG FEASIBLE DIRECTION. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexAlgorithmTypeFO::setLineSearchMethodParameters(dotk::DOTk_LineSearchStepMng & mng_)
{
    size_t max_num_itr = this->getMaxNumLineSearchItr();
    mng_.setMaxNumIterations(max_num_itr);
    Real contraction_factor = this->getLineSearchContractionFactor();
    mng_.setContractionFactor(contraction_factor);
    Real stagnation_tolerance = this->getLineSearchStagnationTolerance();
    mng_.setStagnationTolerance(stagnation_tolerance);
}

void DOTk_MexAlgorithmTypeFO::gatherOutputData(const dotk::DOTk_FirstOrderAlgorithm & algorithm_,
                                               const dotk::DOTk_LineSearchAlgorithmsDataMng & mng_,
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

void DOTk_MexAlgorithmTypeFO::initialize(const mxArray* options_)
{
    dotk::mex::parseProblemType(options_, m_ProblemType);
    dotk::mex::parseLineSearchMethod(options_, m_LineSearchMethod);
    dotk::mex::parseGradientTolerance(options_, m_GradientTolerance);
    dotk::mex::parseTrialStepTolerance(options_, m_TrialStepTolerance);
    dotk::mex::parseMaxNumAlgorithmItr(options_, m_MaxNumAlgorithmItr);
    dotk::mex::parseOptimalityTolerance(options_, m_OptimalityTolerance);
    dotk::mex::parseMaxNumLineSearchItr(options_, m_MaxNumLineSearchItr);
    dotk::mex::parseLineSearchContractionFactor(options_, m_LineSearchContractionFactor);
    dotk::mex::parseLineSearchStagnationTolerance(options_, m_LineSearchStagnationTolerance);
}

}
