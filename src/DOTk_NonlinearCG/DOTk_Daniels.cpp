/*
 * DOTk_Daniels.cpp
 *
 *  Created on: Jul 6, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Daniels.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_Daniels::DOTk_Daniels(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_) :
        dotk::DOTk_DescentDirection(dotk::types::DANIELS_NLCG),
        m_Hessian(hessian_)
{
}

DOTk_Daniels::~DOTk_Daniels()
{
}

void DOTk_Daniels::direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    Real value = dotk::DOTk_DescentDirection::computeCosineAngle(mng_->getOldGradient(), mng_->getTrialStep());

    if(dotk::DOTk_DescentDirection::isTrialStepOrthogonalToSteepestDescent(value) == true)
    {
        dotk::DOTk_DescentDirection::steepestDescent(mng_->getNewGradient(), mng_->getTrialStep());
        return;
    }

    m_Hessian->apply(mng_, mng_->getTrialStep(), mng_->getMatrixTimesVector());

    Real gradient_dot_matrix_times_direction = mng_->getNewGradient()->dot(*mng_->getMatrixTimesVector());
    Real direction_dot_matrix_times_direction = mng_->getTrialStep()->dot(*mng_->getMatrixTimesVector());
    Real scale_factor = gradient_dot_matrix_times_direction / direction_dot_matrix_times_direction;

    dotk::DOTk_DescentDirection::setScaleFactor(scale_factor);
    mng_->getTrialStep()->scale(scale_factor);
    mng_->getTrialStep()->axpy(static_cast<Real>(-1.0), *mng_->getNewGradient());
}

}
