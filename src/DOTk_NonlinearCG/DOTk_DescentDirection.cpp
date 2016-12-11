/*
 * DOTk_DescentDirection.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include "vector.hpp"
#include "DOTk_DescentDirection.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_DescentDirection::DOTk_DescentDirection(dotk::types::nonlinearcg_t dir_) :
        mScaleFactor(0),
        mMinCosineAngleTol(1e-2),
        mNonlinearCGType(dir_)
{
}

DOTk_DescentDirection::~DOTk_DescentDirection()
{
}

void DOTk_DescentDirection::setScaleFactor(Real factor_)
{
    mScaleFactor = factor_;
}

Real DOTk_DescentDirection::getScaleFactor() const
{
    return (mScaleFactor);
}

void DOTk_DescentDirection::setMinCosineAngleTol(Real tol_)
{
    mMinCosineAngleTol = tol_;
}

Real DOTk_DescentDirection::getMinCosineAngleTol() const
{
    return (mMinCosineAngleTol);
}

void DOTk_DescentDirection::setNonlinearCGType(dotk::types::nonlinearcg_t type_)
{
    mNonlinearCGType = type_;
}

dotk::types::nonlinearcg_t DOTk_DescentDirection::getNonlinearCGType() const
{
    return (mNonlinearCGType);
}

Real DOTk_DescentDirection::computeCosineAngle(const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                                               const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    Real norm_dir = dir_->norm();
    Real grad_dot_dir = grad_->dot(*dir_);
    Real norm_grad = grad_->norm();

    Real value = grad_dot_dir / (norm_dir * norm_grad);
    value = std::abs(value);

    return (value);
}

bool DOTk_DescentDirection::isTrialStepOrthogonalToSteepestDescent(Real cosine_val_)
{
    bool is_orthogonal = false;
    if (cosine_val_ < this->getMinCosineAngleTol())
    {
        is_orthogonal = true;
    }
    else if (std::isnan(cosine_val_))
    {
        is_orthogonal = true;
    }
    else if (std::isinf(cosine_val_))
    {
        is_orthogonal = true;
    }
    return (is_orthogonal);
}

void DOTk_DescentDirection::steepestDescent(const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_,
                                            const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_)
{
    dir_->copy(*grad_);
    dir_->scale(static_cast<Real>(-1.));
}

}
