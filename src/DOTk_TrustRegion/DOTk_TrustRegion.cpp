/*
 * DOTk_TrustRegion.cpp
 *
 *  Created on: Oct 30, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>

#include "vector.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_TrustRegion::DOTk_TrustRegion() :
        mLineSearchIterationCount(-1),
        mMaxTrustRegionSubProblemIterations(50),
        mNumTrustRegionSubProblemItrDone(0),
        mInvalidCurvatureDetected(false),
        mNormGradient(0.),
        mLineSearchStep(1.0),
        mTrustRegionRadius(1e4),
        mMaxTrustRegionRadius(1e4),
        mMinTrustRegionRadius(1e-6),
        mActualReduction(0.),
        mPredictedReduction(0.),
        mContractionParameter(0.5),
        mExpansionParameter(2.0),
        mGradTimesMatrixTimesGrad(0.),
        mMinActualOverPredictedReductionAllowed(0.25),
        mTrustRegionType(dotk::types::TRUST_REGION_DISABLED)
{
}

DOTk_TrustRegion::DOTk_TrustRegion(dotk::types::trustregion_t type_) :
        mLineSearchIterationCount(-1),
        mMaxTrustRegionSubProblemIterations(50),
        mNumTrustRegionSubProblemItrDone(0),
        mInvalidCurvatureDetected(false),
        mNormGradient(0.),
        mLineSearchStep(1.0),
        mTrustRegionRadius(1e4),
        mMaxTrustRegionRadius(1e4),
        mMinTrustRegionRadius(1e-6),
        mActualReduction(0.),
        mPredictedReduction(0.),
        mContractionParameter(0.5),
        mExpansionParameter(2.0),
        mGradTimesMatrixTimesGrad(0.),
        mMinActualOverPredictedReductionAllowed(0.25),
        mTrustRegionType(type_)
{
}

DOTk_TrustRegion::~DOTk_TrustRegion()
{
}

void DOTk_TrustRegion::setLineSearchIterationCount(size_t value_)
{
    mLineSearchIterationCount = value_;
}

Int DOTk_TrustRegion::getLineSearchIterationCount() const
{
    return (mLineSearchIterationCount);
}

void DOTk_TrustRegion::setMaxTrustRegionSubProblemIterations(Int itr_)
{
    mMaxTrustRegionSubProblemIterations = itr_;
}

size_t DOTk_TrustRegion::getMaxTrustRegionSubProblemIterations() const
{
    return (mMaxTrustRegionSubProblemIterations);
}

void DOTk_TrustRegion::setNumTrustRegionSubProblemItrDone(Int itr_)
{
    mNumTrustRegionSubProblemItrDone = itr_;
}

size_t DOTk_TrustRegion::getNumTrustRegionSubProblemItrDone() const
{
    return (mNumTrustRegionSubProblemItrDone);
}

void DOTk_TrustRegion::invalidCurvatureDetected(bool invalid_curvature_)
{
    mInvalidCurvatureDetected = invalid_curvature_;
}

bool DOTk_TrustRegion::isCurvatureInvalid() const
{
    return (mInvalidCurvatureDetected);
}

void DOTk_TrustRegion::setLineSearchStep(Real value_)
{
    mLineSearchStep = value_;
}

void DOTk_TrustRegion::setNormGradient(Real value_)
{
    mNormGradient = value_;
}

Real DOTk_TrustRegion::getNormGradient() const
{
    return (mNormGradient);
}

Real DOTk_TrustRegion::getLineSearchStep() const
{
    return (mLineSearchStep);
}

void DOTk_TrustRegion::setTrustRegionRadius(Real value_)
{
    mTrustRegionRadius = value_;
}

Real DOTk_TrustRegion::getTrustRegionRadius() const
{
    return (mTrustRegionRadius);
}

void DOTk_TrustRegion::setMaxTrustRegionRadius(Real value_)
{
    mMaxTrustRegionRadius = value_;
}

Real DOTk_TrustRegion::getMaxTrustRegionRadius() const
{
    return (mMaxTrustRegionRadius);
}

void DOTk_TrustRegion::setMinTrustRegionRadius(Real value_)
{
    mMinTrustRegionRadius = value_;
}

Real DOTk_TrustRegion::getMinTrustRegionRadius() const
{
    return (mMinTrustRegionRadius);
}

void DOTk_TrustRegion::setContractionParameter(Real value_)
{
    mContractionParameter = value_;
}

Real DOTk_TrustRegion::getContractionParameter() const
{
    return (mContractionParameter);
}

Real DOTk_TrustRegion::getActualReduction() const
{
    return (mActualReduction);
}

Real DOTk_TrustRegion::getPredictedReduction() const
{
    return (mPredictedReduction);
}

void DOTk_TrustRegion::setExpansionParameter(Real value_)
{
    mExpansionParameter = value_;
}

Real DOTk_TrustRegion::getExpansionParameter() const
{
    return (mExpansionParameter);
}

void DOTk_TrustRegion::setGradTimesMatrixTimesGrad(Real value_)
{
    mGradTimesMatrixTimesGrad = value_;
}

Real DOTk_TrustRegion::getGradTimesMatrixTimesGrad() const
{
    return (mGradTimesMatrixTimesGrad);
}

void DOTk_TrustRegion::setMinActualOverPredictedReductionAllowed(Real value_)
{
    mMinActualOverPredictedReductionAllowed = value_;
}

Real DOTk_TrustRegion::getMinActualOverPredictedReductionAllowed() const
{
    return (mMinActualOverPredictedReductionAllowed);
}

dotk::types::trustregion_t DOTk_TrustRegion::getTrustRegionType() const
{
    return (mTrustRegionType);
}

void DOTk_TrustRegion::setTrustRegionType(dotk::types::trustregion_t type_)
{
    mTrustRegionType = type_;
}

void DOTk_TrustRegion::computeActualReduction(Real new_objective_func_val_, Real old_objective_func_val_)
{
    mActualReduction = old_objective_func_val_ - new_objective_func_val_;
}

void DOTk_TrustRegion::computePredictedReduction(const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                                                 const std::shared_ptr<dotk::Vector<Real> > & trial_step_,
                                                 const std::shared_ptr<dotk::Vector<Real> > & matrix_times_trial_step_)
{
    Real curvature = trial_step_->dot(*matrix_times_trial_step_);
    Real current_gradient_dot_trial_step = new_grad_->dot(*trial_step_);
    mPredictedReduction = static_cast<Real>(-1.) * (current_gradient_dot_trial_step + (static_cast<Real>(0.5) * curvature));
}

void DOTk_TrustRegion::computeCauchyPoint(const std::shared_ptr<dotk::Vector<Real> > & grad_,
                                          const std::shared_ptr<dotk::Vector<Real> > & matrix_times_grad_,
                                          const std::shared_ptr<dotk::Vector<Real> > & cauchy_point_)
{
    Real cauchy_pt_scale_factor = 1.;
    Real norm_grad = grad_->norm();
    this->setNormGradient(norm_grad);
    Real curvature = grad_->dot(*matrix_times_grad_);
    this->setGradTimesMatrixTimesGrad(curvature);
    if(curvature > std::numeric_limits<Real>::min())
    {
        cauchy_pt_scale_factor = (norm_grad * norm_grad * norm_grad) / curvature;
        cauchy_pt_scale_factor = std::min(cauchy_pt_scale_factor, 1.0);
        Real alpha = cauchy_pt_scale_factor / norm_grad;
        cauchy_point_->update(-alpha, *grad_, 0.);
    }
    else
    {
        Real alpha = cauchy_pt_scale_factor / norm_grad;
        cauchy_point_->update(-alpha, *grad_, 0.);
    }
}

Real DOTk_TrustRegion::computeDoglegRoot(const Real & trust_region_radius_,
                                         const std::shared_ptr<dotk::Vector<Real> > & vector1_,
                                         const std::shared_ptr<dotk::Vector<Real> > & vector2_)
{
    Real norm_vector1 = vector1_->norm();
    Real vector1_dot_vector1 = norm_vector1 * norm_vector1;
    Real vector2_dot_vector1 = vector2_->dot(*vector1_);
    Real norm_vector2 = vector2_->norm();
    Real vector2_dot_vector2 = norm_vector2 * norm_vector2;
    Real vector2_dot_vector2_minus_trust_region_square = vector2_dot_vector2
            - (trust_region_radius_ * trust_region_radius_);

    Real sqrt_root_term = std::sqrt((vector2_dot_vector1 * vector2_dot_vector1)
            - (vector1_dot_vector1 * vector2_dot_vector2_minus_trust_region_square));

    Real root = (-vector2_dot_vector1 + sqrt_root_term) / vector1_dot_vector1;

    if(this->isTrustRegionStepInvalid(root) == true)
    {
        root = this->computeAlternateStep(trust_region_radius_, vector2_);
    }

    return (root);
}

bool DOTk_TrustRegion::isTrustRegionStepInvalid(Real step_)
{
    bool invalid_step = false;
    if(std::isnan(step_))
    {
        invalid_step = true;
    }
    else if(std::isinf(step_))
    {
        invalid_step = true;
    }
    return (invalid_step);
}

bool DOTk_TrustRegion::acceptTrustRegionRadius(const std::shared_ptr<dotk::Vector<Real> > & trial_step_)
{
    bool accept_trust_region = false;
    if(this->actualOverPredictedReductionViolated() == true)
    {
        this->shrinkTrustRegionRadius();
    }
    else
    {
        this->expandTrustRegionRadius(trial_step_);
        accept_trust_region = true;
    }
    return (accept_trust_region);
}

Real DOTk_TrustRegion::computeAlternateStep(const Real & trust_region_radius_,
                                            const std::shared_ptr<dotk::Vector<Real> > & vector_)
{
    Real step = 0.;
    Real norm_vector = vector_->norm();
    if(norm_vector > trust_region_radius_)
    {
        step = trust_region_radius_;
    }
    else
    {
        step = static_cast<Real>(1.0);
    }
    return (step);
}

void DOTk_TrustRegion::step(const dotk::DOTk_OptimizationDataMng * const mng_,
                            const std::shared_ptr<dotk::Vector<Real> > & method_specific_required_data_,
                            const std::shared_ptr<dotk::Vector<Real> > & scaled_direction_)
{
    this->computeCauchyPoint(mng_->getNewGradient(), method_specific_required_data_, scaled_direction_);
    Real trust_region = this->getTrustRegionRadius();
    scaled_direction_->scale(trust_region);
}

void DOTk_TrustRegion::shrinkTrustRegionRadius()
{
    Real trust_region_radius = this->getTrustRegionRadius();
    trust_region_radius *= this->getContractionParameter();
    this->setTrustRegionRadius(trust_region_radius);
}

void DOTk_TrustRegion::expandTrustRegionRadius(const std::shared_ptr<dotk::Vector<Real> > & trial_step_)
{
    Real trust_region_radius = this->getTrustRegionRadius();
    Real norm_trial_step = trial_step_->norm();
    Real norm_trial_step_minus_trust_region_radius = norm_trial_step - trust_region_radius;
    Real trust_region_radius_boundary = static_cast<Real>(1.e-4) * trust_region_radius;
    if(std::abs(norm_trial_step_minus_trust_region_radius) < trust_region_radius_boundary)
    {
        trust_region_radius = std::min(this->getExpansionParameter() * trust_region_radius,
                                       this->getMaxTrustRegionRadius());
        this->setTrustRegionRadius(trust_region_radius);
    }
    else
    {
        this->setTrustRegionRadius(trust_region_radius);
    }
}

bool DOTk_TrustRegion::actualOverPredictedReductionViolated()
{
    Real actual_reduction = this->getActualReduction();
    Real actual_over_predicted_reduction = actual_reduction / this->getPredictedReduction();
    bool min_actual_over_predicted_reduction_violated = false;
    if(std::isnan(actual_over_predicted_reduction))
    {
        min_actual_over_predicted_reduction_violated = true;
    }
    else if(std::isinf(actual_over_predicted_reduction))
    {
        min_actual_over_predicted_reduction_violated = true;
    }
    else if(actual_reduction < std::numeric_limits<Real>::min())
    {
        min_actual_over_predicted_reduction_violated = true;
    }
    else if(actual_over_predicted_reduction < this->getMinActualOverPredictedReductionAllowed())
    {
        min_actual_over_predicted_reduction_violated = true;
    }
    return (min_actual_over_predicted_reduction_violated);
}

}
