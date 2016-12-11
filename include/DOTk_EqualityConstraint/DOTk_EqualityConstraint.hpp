/*
 * DOTk_EqualityConstraint.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EQUALITYCONSTRAINT_HPP_
#define DOTK_EQUALITYCONSTRAINT_HPP_

#include "vector.hpp"
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

template<typename ScalarType>
class DOTk_EqualityConstraint
{
public:
    DOTk_EqualityConstraint()
    {
    }
    virtual ~DOTk_EqualityConstraint()
    {
    }

    virtual void jacobian(const dotk::Vector<ScalarType> & primal_,
                          const dotk::Vector<ScalarType> & vector_,
                          dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::jacobian. ABORT. ****\n");
        std::abort();
    }
    virtual void residual(const dotk::Vector<ScalarType> & primal_, dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::residual(control,output). ABORT. ****\n");
        std::abort();
    }
    virtual void adjointJacobian(const dotk::Vector<ScalarType> & primal_,
                                 const dotk::Vector<ScalarType> & dual_,
                                 dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::adjointJacobian. ABORT. ****\n");
        std::abort();
    }
    virtual void hessian(const dotk::Vector<ScalarType> & primal_,
                         const dotk::Vector<ScalarType> & dual_,
                         const dotk::Vector<ScalarType> & vector_,
                         dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::hessian. ABORT. ****\n");
        std::abort();
    }

    virtual void solve(const dotk::Vector<ScalarType> & control_, dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::solve. ABORT. ****\n");
        std::abort();
    }
    virtual void applyInverseJacobianState(const dotk::Vector<ScalarType> & state_,
                                           const dotk::Vector<ScalarType> & control_,
                                           const dotk::Vector<ScalarType> & rhs_,
                                           dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::applyInverseJacobianState. ABORT. ****\n");
        std::abort();
    }
    virtual void applyAdjointInverseJacobianState(const dotk::Vector<ScalarType> & state_,
                                                  const dotk::Vector<ScalarType> & control_,
                                                  const dotk::Vector<ScalarType> & rhs_,
                                                  dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::applyInverseAdjointJacobianState. ABORT. ****\n");
        std::abort();
    }

    virtual void residual(const dotk::Vector<ScalarType> & state,
                          const dotk::Vector<ScalarType> & control_,
                          dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::residual(state,control,output). ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                        const dotk::Vector<ScalarType> & control_,
                                        const dotk::Vector<ScalarType> & vector_,
                                        dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                          const dotk::Vector<ScalarType> & control_,
                                          const dotk::Vector<ScalarType> & vector_,
                                          dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeControl. ABORT. ****\n");
        std::abort();
    }

    virtual void adjointPartialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::adjointFirstDerivativeState. ABORT. ****\n");
        std::abort();
    }
    virtual void adjointPartialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                                 const dotk::Vector<ScalarType> & control_,
                                                 const dotk::Vector<ScalarType> & dual_,
                                                 dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::adjointFirstDerivativeControl. ABORT. ****\n");
        std::abort();
    }

    virtual void partialDerivativeStateState(const dotk::Vector<ScalarType> & state_,
                                             const dotk::Vector<ScalarType> & control_,
                                             const dotk::Vector<ScalarType> & dual_,
                                             const dotk::Vector<ScalarType> & vector_,
                                             dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeStateState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeStateControl(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeStateControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlControl(const dotk::Vector<ScalarType> & state_,
                                                 const dotk::Vector<ScalarType> & control_,
                                                 const dotk::Vector<ScalarType> & dual_,
                                                 const dotk::Vector<ScalarType> & vector_,
                                                 dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeControlControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlState(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeControlState. ABORT. ****\n");
        std::abort();
    }

private:
    DOTk_EqualityConstraint(const dotk::DOTk_EqualityConstraint<ScalarType> &);
    dotk::DOTk_EqualityConstraint<ScalarType> & operator=(const dotk::DOTk_EqualityConstraint<ScalarType> &);
};

}

#endif /* DOTK_EQUALITYCONSTRAINT_HPP_ */
