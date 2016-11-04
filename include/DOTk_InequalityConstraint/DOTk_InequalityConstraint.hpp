/*
 * DOTk_InequalityConstraint.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEQUALITYCONSTRAINT_HPP_
#define DOTK_INEQUALITYCONSTRAINT_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<class Type>
class vector;

template<class Type>
class DOTk_InequalityConstraint
{
public:
    DOTk_InequalityConstraint()
    {
    }
    virtual ~DOTk_InequalityConstraint()
    {
    }

    virtual Type bound()
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::bound. ABORT.\n");
        std::abort();
        return (0.);
    }
    virtual Type value(const dotk::vector<Type> & primal_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::value. ABORT.\n");
        std::abort();
        return (0.);
    }
    virtual void gradient(const dotk::vector<Type> & primal_, dotk::vector<Type> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::gradient. ABORT.\n");
        std::abort();
    }
    virtual void hessian(const dotk::vector<Type> & primal_,
                         const dotk::vector<Type> & delta_primal_,
                         dotk::vector<Type> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::hessian. ABORT.\n");
        std::abort();
    }

    virtual Type value(const dotk::vector<Type> & state_, const dotk::vector<Type> & control_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::value. ABORT.\n");
        std::abort();
        return (0.);
    }
    virtual void partialDerivativeState(const dotk::vector<Type> & state_,
                                        const dotk::vector<Type> & control_,
                                        dotk::vector<Type> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::partialDerivativeState. ABORT.\n");
        std::abort();
    }
    virtual void partialDerivativeControl(const dotk::vector<Type> & state_,
                                          const dotk::vector<Type> & control_,
                                          dotk::vector<Type> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::partialDerivativeControl. ABORT.\n");
        std::abort();
    }

private:
    DOTk_InequalityConstraint(const dotk::DOTk_InequalityConstraint<Type> &);
    dotk::DOTk_InequalityConstraint<Type> & operator=(const dotk::DOTk_InequalityConstraint<Type> &);
};

}

#endif /* DOTK_INEQUALITYCONSTRAINT_HPP_ */
