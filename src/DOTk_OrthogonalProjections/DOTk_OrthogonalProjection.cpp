/*
 * DOTk_OrthogonalProjection.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cstdio>

#include "DOTk_Types.hpp"
#include "DOTk_OrthogonalProjection.hpp"

namespace dotk
{

DOTk_OrthogonalProjection::DOTk_OrthogonalProjection(dotk::types::projection_t type_, size_t krylov_subspace_dim_) :
        mInitialResidual(0.),
        mKrylovSubspaceDim(krylov_subspace_dim_),
        mProjectionType(type_)
{
}

DOTk_OrthogonalProjection::~DOTk_OrthogonalProjection()
{
}

void DOTk_OrthogonalProjection::setInitialResidual(Real value_)
{
    mInitialResidual = value_;
}

Real DOTk_OrthogonalProjection::getInitialResidual() const
{
    return (mInitialResidual);
}

void DOTk_OrthogonalProjection::setKrylovSubspaceDim(size_t dim_)
{
    mKrylovSubspaceDim = dim_;
}

size_t DOTk_OrthogonalProjection::getKrylovSubspaceDim() const
{
    return (mKrylovSubspaceDim);
}

void DOTk_OrthogonalProjection::setProjectionType(dotk::types::projection_t type_)
{
    mProjectionType = type_;
}

dotk::types::projection_t DOTk_OrthogonalProjection::getProjectionType() const
{
    return (mProjectionType);
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_OrthogonalProjection::getLinearOperatorTimesOrthoVector(size_t index_) const
{
    std::perror("\n**** DOTk ERROR in DOTk_OrthogonalProjection::getLinearOperatorTimesOrthoVector. Function not implemented. ABORT. ****\n");
    std::abort();
}

void DOTk_OrthogonalProjection::setLinearOperatorTimesOrthoVector(size_t index_, const std::shared_ptr<dotk::Vector<Real> > & vec_)
{
    std::perror("\n**** DOTk ERROR in DOTk_OrthogonalProjection::setLinearOperatorTimesOrthoVector. Function not implemented. ABORT. ****\n");
    std::abort();
}

}
