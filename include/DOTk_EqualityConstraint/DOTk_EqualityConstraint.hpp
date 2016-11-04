/*
 * DOTk_EqualityConstraint.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EQUALITYCONSTRAINT_HPP_
#define DOTK_EQUALITYCONSTRAINT_HPP_

#include "vector.hpp"

namespace dotk
{

template<class Type>
class vector;

template<class Type>
class DOTk_EqualityConstraint
{
public:
    DOTk_EqualityConstraint()
    {
    }
    virtual ~DOTk_EqualityConstraint()
    {
    }

    virtual void jacobian(const dotk::vector<Type> & primal_,
                          const dotk::vector<Type> & vector_,
                          dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::jacobian. ABORT. ****\n");
        std::abort();
    }
    virtual void residual(const dotk::vector<Type> & primal_, dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::residual(control,output). ABORT. ****\n");
        std::abort();
    }
    virtual void adjointJacobian(const dotk::vector<Type> & primal_,
                                 const dotk::vector<Type> & dual_,
                                 dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::adjointJacobian. ABORT. ****\n");
        std::abort();
    }
    virtual void hessian(const dotk::vector<Type> & primal_,
                         const dotk::vector<Type> & dual_,
                         const dotk::vector<Type> & vector_,
                         dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::hessian. ABORT. ****\n");
        std::abort();
    }

    virtual void solve(const dotk::vector<Type> & control_, dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::solve. ABORT. ****\n");
        std::abort();
    }
    virtual void applyInverseJacobianState(const dotk::vector<Type> & state_,
                                           const dotk::vector<Type> & control_,
                                           const dotk::vector<Type> & rhs_,
                                           dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::applyInverseJacobianState. ABORT. ****\n");
        std::abort();
    }
    virtual void applyAdjointInverseJacobianState(const dotk::vector<Type> & state_,
                                                  const dotk::vector<Type> & control_,
                                                  const dotk::vector<Type> & rhs_,
                                                  dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::applyInverseAdjointJacobianState. ABORT. ****\n");
        std::abort();
    }

    virtual void residual(const dotk::vector<Type> & state,
                          const dotk::vector<Type> & control_,
                          dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::residual(state,control,output). ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeState(const dotk::vector<Type> & state_,
                                        const dotk::vector<Type> & control_,
                                        const dotk::vector<Type> & vector_,
                                        dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControl(const dotk::vector<Type> & state_,
                                          const dotk::vector<Type> & control_,
                                          const dotk::vector<Type> & vector_,
                                          dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeControl. ABORT. ****\n");
        std::abort();
    }

    virtual void adjointPartialDerivativeState(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & dual_,
                                               dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::adjointFirstDerivativeState. ABORT. ****\n");
        std::abort();
    }
    virtual void adjointPartialDerivativeControl(const dotk::vector<Type> & state_,
                                                 const dotk::vector<Type> & control_,
                                                 const dotk::vector<Type> & dual_,
                                                 dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::adjointFirstDerivativeControl. ABORT. ****\n");
        std::abort();
    }

    virtual void partialDerivativeStateState(const dotk::vector<Type> & state_,
                                             const dotk::vector<Type> & control_,
                                             const dotk::vector<Type> & dual_,
                                             const dotk::vector<Type> & vector_,
                                             dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeStateState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeStateControl(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & dual_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeStateControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlControl(const dotk::vector<Type> & state_,
                                                 const dotk::vector<Type> & control_,
                                                 const dotk::vector<Type> & dual_,
                                                 const dotk::vector<Type> & vector_,
                                                 dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeControlControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlState(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & dual_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_EqualityConstraint::partialDerivativeControlState. ABORT. ****\n");
        std::abort();
    }

private:
    DOTk_EqualityConstraint(const dotk::DOTk_EqualityConstraint<Type> &);
    dotk::DOTk_EqualityConstraint<Type> & operator=(const dotk::DOTk_EqualityConstraint<Type> &);
};

}

#endif /* DOTK_EQUALITYCONSTRAINT_HPP_ */
