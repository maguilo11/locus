/*
 * DOTk_DoubleDoglegTrustRegion.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DoubleDoglegTrustRegion.hpp"

namespace dotk
{

DOTk_DoubleDoglegTrustRegion::DOTk_DoubleDoglegTrustRegion(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_TrustRegion(dotk::types::trustregion_t::TRUST_REGION_DOUBLE_DOGLEG),
        mParamPromoteMonotonicallyDecreasingQuadraticModel(0.8),
        mCauchyPoint(vector_->clone()),
        mScaledNewtonStep(vector_->clone())
{
}

DOTk_DoubleDoglegTrustRegion::~DOTk_DoubleDoglegTrustRegion()
{
}

Real DOTk_DoubleDoglegTrustRegion::getParamPromotesMonotonicallyDecreasingQuadraticModel() const
{
    return (mParamPromoteMonotonicallyDecreasingQuadraticModel);
}

void DOTk_DoubleDoglegTrustRegion::setParamPromotesMonotonicallyDecreasingQuadraticModel(Real value_)
{
    mParamPromoteMonotonicallyDecreasingQuadraticModel = value_;
}

Real DOTk_DoubleDoglegTrustRegion::computeDoubleDoglegRoot(const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & newton_direction_,
                                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_grad_)
{
    /// Computes point in the Newton direction at which the reduction of the \n
    /// quadratic approximation to the objectivte function resulting from \n
    /// taking Newton step is the same as taking the Cauchy step. \n
    /// The root is given by: 1 - alpha(1-constant), see J.E. Dennis and H.W. Mei. \n
    Real grad_dot_grad = grad_->dot(*grad_);
    Real grad_dot_negative_newton_direction = static_cast<Real>( - 1.0) * grad_->dot(*newton_direction_);
    Real curvature = grad_->dot(*matrix_times_grad_);
    Real constant = grad_dot_grad * grad_dot_grad / (curvature * grad_dot_negative_newton_direction);
    Real factor = static_cast<Real>(1.0) - this->getParamPromotesMonotonicallyDecreasingQuadraticModel();
    Real root = factor + (this->getParamPromotesMonotonicallyDecreasingQuadraticModel() * constant);
    if(DOTk_TrustRegion::isTrustRegionStepInvalid(root) == true)
    {
        root = static_cast<Real>(1.0);
    }
    return (root);
}

void DOTk_DoubleDoglegTrustRegion::doubleDogleg(const Real & trust_region_radius_,
                                                const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                                                const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_grad_,
                                                const std::tr1::shared_ptr<dotk::Vector<Real> > & newton_step_)
{
    Real norm_newton_step = newton_step_->norm();
    if(dotk::DOTk_TrustRegion::isCurvatureInvalid() == true)
    {
        this->computeScaledNewtonStep(grad_, matrix_times_grad_, newton_step_);
        dotk::DOTk_TrustRegion::computeCauchyPoint(grad_, matrix_times_grad_, mCauchyPoint);
        this->computeConvexCombinationBetweenCauchyAndDoglegStep(trust_region_radius_, newton_step_);
        return;
    }
    else if(norm_newton_step <= trust_region_radius_)
    {
        return;
    }
    else
    {
        this->computeScaledNewtonStep(grad_, matrix_times_grad_, newton_step_);
        Real norm_scaled_newton_step = mScaledNewtonStep->norm();
        if(norm_scaled_newton_step <= trust_region_radius_)
        {
            // Dogleg Point is inside trust region
            Real scale_factor = trust_region_radius_ / norm_newton_step;
            newton_step_->scale(scale_factor);
        }
        else
        {
            // Dogleg Point is outside trust region
            dotk::DOTk_TrustRegion::computeCauchyPoint(grad_, matrix_times_grad_, mCauchyPoint);
            Real norm_grad = dotk::DOTk_TrustRegion::getNormGradient();
            Real curvature = dotk::DOTk_TrustRegion::getGradTimesMatrixTimesGrad();
            Real norm_cauchy_point = (norm_grad * norm_grad * norm_grad) / curvature;
            if(norm_cauchy_point >= trust_region_radius_)
            {
                // Cauchy Point is outside trust region
                Real scale_factor = -trust_region_radius_ / norm_grad;
                newton_step_->update(scale_factor, *grad_, 0.);
            }
            else
            {
                this->computeConvexCombinationBetweenCauchyAndDoglegStep(trust_region_radius_, newton_step_);
            }
        }
    }
}

void DOTk_DoubleDoglegTrustRegion::step(const dotk::DOTk_OptimizationDataMng * const mng_,
                                        const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_grad_,
                                        const std::tr1::shared_ptr<dotk::Vector<Real> > & scaled_direction_)
{
    Real trust_region = dotk::DOTk_TrustRegion::getTrustRegionRadius();
    this->doubleDogleg(trust_region, mng_->getNewGradient(), matrix_times_grad_, scaled_direction_);
}

void DOTk_DoubleDoglegTrustRegion::computeScaledNewtonStep(const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_grad_,
                                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & newton_step_)
{
    Real double_dogleg_root = this->computeDoubleDoglegRoot(grad_, newton_step_, matrix_times_grad_);
    mScaledNewtonStep->update(double_dogleg_root, *newton_step_, 0.);
}

void DOTk_DoubleDoglegTrustRegion::computeConvexCombinationBetweenCauchyAndDoglegStep
(const Real & trust_region_radius_, const std::tr1::shared_ptr<dotk::Vector<Real> > & newton_step_)
{
    mScaledNewtonStep->update(-1., *mCauchyPoint, 1.);
    Real dogleg_root = dotk::DOTk_TrustRegion::computeDoglegRoot(trust_region_radius_,
                                                                 mScaledNewtonStep,
                                                                 mCauchyPoint);
    newton_step_->update(1., *mCauchyPoint, 0.);
    newton_step_->update(dogleg_root, *mScaledNewtonStep, 1.);
}

}
