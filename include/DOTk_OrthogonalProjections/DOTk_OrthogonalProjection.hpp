/*
 * DOTk_OrthogonalProjection.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_ORTHOGONALPROJECTION_HPP_
#define DOTK_ORTHOGONALPROJECTION_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename ScalarType>
class Vector;

class DOTk_OrthogonalProjection
{
public:
    DOTk_OrthogonalProjection(dotk::types::projection_t type_, size_t krylov_subspace_dim_);
    virtual ~DOTk_OrthogonalProjection();

    void setKrylovSubspaceDim(size_t dim_);
    size_t getKrylovSubspaceDim() const;

    void setProjectionType(dotk::types::projection_t type_);
    dotk::types::projection_t getProjectionType() const;

    virtual void setInitialResidual(Real value_);
    virtual Real getInitialResidual() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getLinearOperatorTimesOrthoVector(size_t index_) const;
    virtual void setLinearOperatorTimesOrthoVector(size_t index_,
                                                   const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_);

    virtual void clear() = 0;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getOrthogonalVector(size_t i_) const = 0;
    virtual void setOrthogonalVector(size_t index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_) = 0;
    virtual void apply(const dotk::DOTk_KrylovSolver * const solver_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_) = 0;

private:
    Real mInitialResidual;
    size_t mKrylovSubspaceDim;
    dotk::types::projection_t mProjectionType;

private:
    DOTk_OrthogonalProjection(const dotk::DOTk_OrthogonalProjection &);
    dotk::DOTk_OrthogonalProjection & operator=(const dotk::DOTk_OrthogonalProjection &);
};

}

#endif /* DOTK_ORTHOGONALPROJECTION_HPP_ */
