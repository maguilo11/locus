/*
 * DOTk_DoglegTrustRegion.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "vector.hpp"
#include "DOTk_DoglegTrustRegion.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_DoglegTrustRegion::DOTk_DoglegTrustRegion() :
        dotk::DOTk_TrustRegion(dotk::types::trustregion_t::TRUST_REGION_DOGLEG)
{
}

DOTk_DoglegTrustRegion::~DOTk_DoglegTrustRegion()
{
}

void DOTk_DoglegTrustRegion::dogleg(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & cauchy_step_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & newton_step_)
{
    Real current_trust_region_radius = dotk::DOTk_TrustRegion::getTrustRegionRadius();
    Real grad_dot_newton_step = grad_->dot(*newton_step_);
    if(grad_dot_newton_step >= std::numeric_limits<Real>::min())
    {
        dotk::DOTk_TrustRegion::computeCauchyPoint(grad_, matrix_times_grad_, newton_step_);
        newton_step_->scale(current_trust_region_radius);
        return;
    }
    else if(dotk::DOTk_TrustRegion::isCurvatureInvalid() == true)
    {
        Real theta = dotk::DOTk_TrustRegion::computeDoglegRoot(current_trust_region_radius, cauchy_step_, newton_step_);
        newton_step_->axpy(theta, *cauchy_step_);
    }
    else
    {
        Real norm_grad = grad_->norm();
        Real norm_newton_step = newton_step_->norm();
        Real grad_dot_matrix_times_grad = grad_->dot(*matrix_times_grad_);
        if(norm_newton_step <= current_trust_region_radius)
        {
            return;
        }
        else if(grad_dot_matrix_times_grad <= std::numeric_limits<Real>::min())
        {
            newton_step_->copy(*grad_);
            Real scale_factor = -current_trust_region_radius / norm_grad;
            newton_step_->scale(scale_factor);
        }
        else
        {
            Real theta = dotk::DOTk_TrustRegion::computeDoglegRoot(current_trust_region_radius, cauchy_step_, newton_step_);
            newton_step_->axpy(theta, *cauchy_step_);
        }
    }
}

void DOTk_DoglegTrustRegion::step(const dotk::DOTk_OptimizationDataMng * const mng_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & cauchy_direction_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & scaled_direction_)
{
    this->dogleg(mng_->getNewGradient(), mng_->getMatrixTimesVector(), cauchy_direction_, scaled_direction_);
}

}
