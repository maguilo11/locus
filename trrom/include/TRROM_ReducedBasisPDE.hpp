/*
 * TRROM_ReducedBasisPDE.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASISPDE_HPP_
#define TRROM_REDUCEDBASISPDE_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class ReducedBasisPDE
{
public:
    virtual ~ReducedBasisPDE()
    {
    }

    // Linear system solves
    virtual void solve(const trrom::Vector<double> & control_, trrom::Vector<double> & solution_) = 0;
    virtual void applyAdjointInverseJacobianState(const trrom::Vector<double> & state_,
                                                  const trrom::Vector<double> & control_,
                                                  const trrom::Vector<double> & rhs_,
                                                  trrom::Vector<double> & solution_) = 0;

    // First order operators
    virtual void partialDerivativeState(const trrom::Vector<double> & state_,
                                        const trrom::Vector<double> & control_,
                                        const trrom::Vector<double> & vector_,
                                        trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeControl(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          const trrom::Vector<double> & vector_,
                                          trrom::Vector<double> & output_) = 0;
    virtual void adjointPartialDerivativeState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & dual_,
                                               trrom::Vector<double> & output_) = 0;
    virtual void adjointPartialDerivativeControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & dual_,
                                                 trrom::Vector<double> & output_) = 0;

    // Second order operators
    virtual void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                             const trrom::Vector<double> & control_,
                                             const trrom::Vector<double> & dual_,
                                             const trrom::Vector<double> & vector_,
                                             trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & dual_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & dual_,
                                                 const trrom::Vector<double> & vector_,
                                                 trrom::Vector<double> & output_) = 0;
    virtual void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & dual_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;

    // Reduced Basis specific methods
    virtual void getLeftHandSideSnapshot(trrom::Vector<double> & lhs_) = 0;
    virtual void getRightHandSideSnapshot(trrom::Vector<double> & rhs_) = 0;
    virtual void getReducedLeftHandSideSnapshot(const trrom::Vector<double> & control_,
                                                trrom::Vector<double> & lhs_) = 0;
    virtual void updateLeftHandSideActiveIndices(const trrom::Matrix<double> & indices_) = 0;
};

}

#endif /* TRROM_REDUCEDBASISPDE_HPP_ */
