/*
 * DOTk_MexSteihaugTointNewton.cpp
 *
 *  Created on: Sep 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_TrustRegionStepMng.hpp"
#include "DOTk_SteihaugTointNewton.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_MexSteihaugTointNewton.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

namespace dotk
{

DOTk_MexSteihaugTointNewton::DOTk_MexSteihaugTointNewton(const mxArray* options_[]) :
        m_MaxNumOptItr(100),
        m_MaxNumSubProblemItr(20),
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_ObjectiveTolerance(1e-10),
        m_ActualReductionTolerance(1e-8),
        m_MaxTrustRegionRadius(1e4),
        m_InitialTrustRegionRadius(1e2),
        m_TrustRegionExpansionFactor(2),
        m_TrustRegionContractionFactor(0.5),
        m_ActualOverPredictedReductionLowerBound(0.1),
        m_ActualOverPredictedReductionUpperBound(0.75),
        m_ActualOverPredictedReductionMiddleBound(0.25),
        m_SetInitialTrustRegionRadiusToNormGrad(true)
{
    this->initialize(options_);
}

DOTk_MexSteihaugTointNewton::~DOTk_MexSteihaugTointNewton()
{
}

size_t DOTk_MexSteihaugTointNewton::getMaxNumOptimizationItr() const
{
    return (m_MaxNumOptItr);
}

size_t DOTk_MexSteihaugTointNewton::getMaxNumSubProblemItr() const
{
    return (m_MaxNumSubProblemItr);
}

double DOTk_MexSteihaugTointNewton::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

double DOTk_MexSteihaugTointNewton::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

double DOTk_MexSteihaugTointNewton::getObjectiveTolerance() const
{
    return (m_ObjectiveTolerance);
}

double DOTk_MexSteihaugTointNewton::getActualReductionTolerance() const
{
    return (m_ActualReductionTolerance);
}

double DOTk_MexSteihaugTointNewton::getActualOverPredictedReductionUpperBound() const
{
    return (m_ActualOverPredictedReductionUpperBound);
}

double DOTk_MexSteihaugTointNewton::getActualOverPredictedReductionLowerBound() const
{
    return (m_ActualOverPredictedReductionLowerBound);
}

double DOTk_MexSteihaugTointNewton::getActualOverPredictedReductionMiddleBound() const
{
    return (m_ActualOverPredictedReductionMiddleBound);
}

double DOTk_MexSteihaugTointNewton::getMaxTrustRegionRadius() const
{
    return (m_MaxTrustRegionRadius);
}

double DOTk_MexSteihaugTointNewton::getInitialTrustRegionRadius() const
{
    return (m_InitialTrustRegionRadius);
}

double DOTk_MexSteihaugTointNewton::getTrustRegionExpansionFactor() const
{
    return (m_TrustRegionExpansionFactor);
}

double DOTk_MexSteihaugTointNewton::getTrustRegionContractionFactor() const
{
    return (m_TrustRegionContractionFactor);
}

bool DOTk_MexSteihaugTointNewton::isInitialTrustRegionRadiusSetToNormGrad() const
{
    return (m_SetInitialTrustRegionRadiusToNormGrad);
}

void DOTk_MexSteihaugTointNewton::initialize(const mxArray* options_[])
{
    dotk::mex::parseMaxNumAlgorithmItr(options_[0], m_MaxNumOptItr);
    dotk::mex::parseGradientTolerance(options_[0], m_GradientTolerance);
    dotk::mex::parseTrialStepTolerance(options_[0], m_TrialStepTolerance);
    dotk::mex::parseOptimalityTolerance(options_[0], m_ObjectiveTolerance);
    dotk::mex::parseActualReductionTolerance(options_[0], m_ActualReductionTolerance);

    dotk::mex::parseMaxTrustRegionRadius(options_[0], m_MaxTrustRegionRadius);
    dotk::mex::parseInitialTrustRegionRadius(options_[0], m_InitialTrustRegionRadius);
    dotk::mex::parseTrustRegionExpansionFactor(options_[0], m_TrustRegionExpansionFactor);
    dotk::mex::parseTrustRegionContractionFactor(options_[0], m_TrustRegionContractionFactor);
    dotk::mex::parseMaxNumTrustRegionSubProblemItr(options_[0], m_MaxNumSubProblemItr);
    dotk::mex::parseMinActualOverPredictedReductionRatio(options_[0], m_ActualOverPredictedReductionLowerBound);
    dotk::mex::parseMaxActualOverPredictedReductionRatio(options_[0], m_ActualOverPredictedReductionUpperBound);
    dotk::mex::parseMidActualOverPredictedReductionRatio(options_[0], m_ActualOverPredictedReductionMiddleBound);
    dotk::mex::parseSetInitialTrustRegionRadiusToNormGradFlag(options_[0], m_SetInitialTrustRegionRadiusToNormGrad);
}

void DOTk_MexSteihaugTointNewton::setAlgorithmParameters(dotk::DOTk_SteihaugTointNewton & algorithm_)
{
    size_t max_num_itr = this->getMaxNumOptimizationItr();
    algorithm_.setMaxNumOptimizationItr(max_num_itr);
    double objective_tolerance = this->getObjectiveTolerance();
    algorithm_.setObjectiveTolerance(objective_tolerance);
    double gradient_tolerance = this->getGradientTolerance();
    algorithm_.setGradientTolerance(gradient_tolerance);
    double trial_step_tolerance = this->getTrialStepTolerance();
    algorithm_.setTrialStepTolerance(trial_step_tolerance);
}

void DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionStepMng> & mng_)
{
    size_t itr = this->getMaxNumSubProblemItr();
    mng_->setMaxNumTrustRegionSubProblemItr(itr);

    double value = this->getActualOverPredictedReductionUpperBound();
    mng_->setActualOverPredictedReductionUpperBound(value);
    value = this->getActualOverPredictedReductionLowerBound();
    mng_->setActualOverPredictedReductionLowerBound(value);
    value = this->getActualOverPredictedReductionMiddleBound();
    mng_->setActualOverPredictedReductionMidBound(value);

    value = this->getMaxTrustRegionRadius();
    mng_->setMaxTrustRegionRadius(value);
    value = this->getInitialTrustRegionRadius();
    mng_->setTrustRegionRadius(value);
    value = this->getTrustRegionExpansionFactor();
    mng_->setTrustRegionExpansion(value);
    value = this->getTrustRegionContractionFactor();
    mng_->setTrustRegionReduction(value);

    bool flag = this->isInitialTrustRegionRadiusSetToNormGrad();
    mng_->setInitialTrustRegionRadiusToGradNorm(flag);
}

void DOTk_MexSteihaugTointNewton::gatherOutputData(const dotk::DOTk_SteihaugTointNewton & algorithm_,
                                                   const dotk::DOTk_OptimizationDataMng & mng_,
                                                   const dotk::DOTk_TrustRegionStepMng & step_,
                                                   mxArray* output_[])
{
    // Create memory allocation for output struct
    const char *field_names[6] =
        { "Iterations", "ObjectiveFunctionValue", "ActualReduction", "NormGradient", "NormTrialStep", "Control"};
    output_[0] = mxCreateStructMatrix(1, 1, 6, field_names);

    dotk::DOTk_MexArrayPtr itr(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(itr.get()))[0] = algorithm_.getNumOptimizationItrDone();
    mxSetField(output_[0], 0, "Iterations", itr.get());

    dotk::DOTk_MexArrayPtr objective_function_value(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(objective_function_value.get())[0] = mng_.getNewObjectiveFunctionValue();
    mxSetField(output_[0], 0, "ObjectiveFunctionValue", objective_function_value.get());

    dotk::DOTk_MexArrayPtr actual_reduction(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(actual_reduction.get())[0] = std::abs(step_.getActualReduction());
    mxSetField(output_[0], 0, "ActualReduction", actual_reduction.get());

    dotk::DOTk_MexArrayPtr norm_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_gradient.get())[0] = mng_.getNormNewGradient();
    mxSetField(output_[0], 0, "NormGradient", norm_gradient.get());

    dotk::DOTk_MexArrayPtr norm_trial_step(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_trial_step.get())[0] = mng_.getNormTrialStep();
    mxSetField(output_[0], 0, "NormTrialStep", norm_trial_step.get());

    size_t num_controls = mng_.getNewPrimal()->size();
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getNewPrimal()->gather(mxGetPr(primal.get()));
    mxSetField(output_[0], 0, "Control", primal.get());

    itr.release();
    objective_function_value.release();
    actual_reduction.release();
    norm_gradient.release();
    norm_trial_step.release();
    primal.release();
}

}
