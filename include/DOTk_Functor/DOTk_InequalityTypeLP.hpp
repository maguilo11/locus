/*
 * DOTk_InequalityTypeLP.hpp
 *
 *  Created on: Mar 28, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEQUALITYTYPELP_HPP_
#define DOTK_INEQUALITYTYPELP_HPP_

#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace lp
{

struct InequalityConstraintEvaluate
{
public:
    InequalityConstraintEvaluate()
    {
    }
    ~InequalityConstraintEvaluate()
    {
    }

    Real operator()(const std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_) const
    {
        Real value = inequality_->value(*primal_);

        return (value);
    }

private:
    InequalityConstraintEvaluate(const dotk::lp::InequalityConstraintEvaluate&);
    dotk::lp::InequalityConstraintEvaluate operator=(const dotk::lp::InequalityConstraintEvaluate&);
};

struct InequalityConstraintFirstDerivative
{
public:
    InequalityConstraintFirstDerivative()
    {
    }
    ~InequalityConstraintFirstDerivative()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_,
                    const std::shared_ptr<dotk::Vector<Real> > & derivative_) const
    {
        inequality_->gradient(*primal_, *derivative_);
    }

private:
    InequalityConstraintFirstDerivative(const dotk::lp::InequalityConstraintFirstDerivative&);
    dotk::lp::InequalityConstraintFirstDerivative operator=(const dotk::lp::InequalityConstraintFirstDerivative&);
};

struct InequalityConstraintSecondDerivative
{
public:
    InequalityConstraintSecondDerivative()
    {
    }
    ~InequalityConstraintSecondDerivative()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_,
                    const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                    const std::shared_ptr<dotk::Vector<Real> > & derivative_) const
    {
        inequality_->hessian(*primal_, *delta_primal_, *derivative_);
    }

private:
    InequalityConstraintSecondDerivative(const dotk::lp::InequalityConstraintSecondDerivative&);
    dotk::lp::InequalityConstraintSecondDerivative operator=(const dotk::lp::InequalityConstraintSecondDerivative&);
};

}

}

#endif /* DOTK_INEQUALITYTYPELP_HPP_ */
