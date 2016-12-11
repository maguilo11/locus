/*
 * DOTk_InequalityConstraint.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEQUALITYCONSTRAINT_HPP_
#define DOTK_INEQUALITYCONSTRAINT_HPP_

#include <iostream>

#include <vector.hpp>
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

template<typename ScalarType>
class DOTk_InequalityConstraint
{
public:
    DOTk_InequalityConstraint()
    {
    }
    virtual ~DOTk_InequalityConstraint()
    {
    }

    virtual ScalarType bound()
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::bound. ABORT.\n");
        std::abort();
        return (0.);
    }
    virtual ScalarType value(const dotk::Vector<ScalarType> & primal_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::value. ABORT.\n");
        std::abort();
        return (0.);
    }
    virtual void gradient(const dotk::Vector<ScalarType> & primal_, dotk::Vector<ScalarType> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::gradient. ABORT.\n");
        std::abort();
    }
    virtual void hessian(const dotk::Vector<ScalarType> & primal_,
                         const dotk::Vector<ScalarType> & delta_primal_,
                         dotk::Vector<ScalarType> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::hessian. ABORT.\n");
        std::abort();
    }

    virtual ScalarType value(const dotk::Vector<ScalarType> & state_, const dotk::Vector<ScalarType> & control_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::value. ABORT.\n");
        std::abort();
        return (0.);
    }
    virtual void partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                        const dotk::Vector<ScalarType> & control_,
                                        dotk::Vector<ScalarType> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::partialDerivativeState. ABORT.\n");
        std::abort();
    }
    virtual void partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                          const dotk::Vector<ScalarType> & control_,
                                          dotk::Vector<ScalarType> & derivative_)
    {
        std::perror("\nUnimplemented Function dotk::DOTk_InequalityConstraint::partialDerivativeControl. ABORT.\n");
        std::abort();
    }

private:
    DOTk_InequalityConstraint(const dotk::DOTk_InequalityConstraint<ScalarType> &);
    dotk::DOTk_InequalityConstraint<ScalarType> & operator=(const dotk::DOTk_InequalityConstraint<ScalarType> &);
};

}

#endif /* DOTK_INEQUALITYCONSTRAINT_HPP_ */
