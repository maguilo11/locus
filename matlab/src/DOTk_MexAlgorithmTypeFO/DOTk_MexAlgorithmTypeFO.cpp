/*
 * DOTk_MexAlgorithmTypeFO.cpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <iostream>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexAlgorithmTypeFO.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchStepMng.hpp"
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
        m_ObjectiveTolerance(1e-10),
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
    m_NumDuals = dotk::mex::parseNumberDuals(options_);
}

size_t DOTk_MexAlgorithmTypeFO::getNumDuals() const
{
    return (m_NumDuals);
}

void DOTk_MexAlgorithmTypeFO::setNumControls(const mxArray* options_)
{
    m_NumControls = dotk::mex::parseNumberControls(options_);
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

double DOTk_MexAlgorithmTypeFO::getObjectiveTolerance() const
{
    return (m_ObjectiveTolerance);
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
    dotk::types::constraint_method_t type = dotk::mex::parseBoundConstraintMethod(options_);
    switch(type)
    {
        case dotk::types::FEASIBLE_DIR:
        {
            size_t iterations = dotk::mex::parseMaxNumFeasibleItr(options_);
            double factor = dotk::mex::parseFeasibleStepContractionFactor(options_);
            step_->setFeasibleDirectionConstraint(primal_);
            step_->setMaxNumFeasibleItr(iterations);
            step_->setBoundConstraintMethodContractionStep(factor);
            break;
        }
        case dotk::types::PROJECTION_ALONG_FEASIBLE_DIR:
        {
            double factor = dotk::mex::parseFeasibleStepContractionFactor(options_);
            step_->setBoundConstraintMethodContractionStep(factor);
            step_->setProjectionAlongFeasibleDirConstraint(primal_);
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> Bound constraint method keyword. Method set to PROJECTION ALONG FEASIBLE DIRECTION.\n";
            mexWarnMsgTxt(msg.str().c_str());
            step_->setProjectionAlongFeasibleDirConstraint(primal_);
            break;
        }
    }
}

void DOTk_MexAlgorithmTypeFO::setLineSearchStepMng(dotk::DOTk_LineSearchStepMng & mng_)
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

void DOTk_MexAlgorithmTypeFO::initialize(const mxArray* options_)
{
    m_ProblemType = dotk::mex::parseProblemType(options_);
    m_TrialStepTolerance = dotk::mex::parseStepTolerance(options_);
    m_LineSearchMethod = dotk::mex::parseLineSearchMethod(options_);
    m_GradientTolerance = dotk::mex::parseGradientTolerance(options_);
    m_ObjectiveTolerance = dotk::mex::parseObjectiveTolerance(options_);
    m_MaxNumLineSearchItr = dotk::mex::parseMaxNumLineSearchItr(options_);
    m_MaxNumAlgorithmItr = dotk::mex::parseMaxNumOuterIterations(options_);
    m_LineSearchContractionFactor = dotk::mex::parseLineSearchContractionFactor(options_);
    m_LineSearchStagnationTolerance = dotk::mex::parseLineSearchStagnationTolerance(options_);
}

}
