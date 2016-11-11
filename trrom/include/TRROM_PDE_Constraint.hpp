/*
 * TRROM_PDE_Constraint.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_PDE_CONSTRAINT_HPP_
#define TRROM_PDE_CONSTRAINT_HPP_

namespace trrom
{

template<typename ScalarType>
class Vector;

class PDE_Constraint
{
public:
    virtual ~PDE_Constraint()
    {
    }

    // Forward and adjoint evaluations
    virtual void solve(const trrom::Vector<double> & control_, trrom::Vector<double> & solution_) = 0;
    virtual void applyInverseJacobianState(const trrom::Vector<double> & state_,
                                           const trrom::Vector<double> & control_,
                                           const trrom::Vector<double> & rhs_,
                                           trrom::Vector<double> & solution_) = 0;
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
};

}

#endif /* TRROM_PDE_CONSTRAINT_HPP_ */
