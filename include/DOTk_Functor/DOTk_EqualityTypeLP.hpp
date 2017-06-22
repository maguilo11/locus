/*
 * DOTk_EqualityTypeLP.hpp
 *
 *  Created on: Mar 28, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EQUALITYTYPELP_HPP_
#define DOTK_EQUALITYTYPELP_HPP_

#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace lp
{

struct EqualityConstraintResidual
{
    EqualityConstraintResidual()
    {
    }
    ~EqualityConstraintResidual()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_,
                    std::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        equality_->residual(*primal_, *output_);
    }

private:
    EqualityConstraintResidual(const dotk::lp::EqualityConstraintResidual&);
    dotk::lp::EqualityConstraintResidual operator=(const dotk::lp::EqualityConstraintResidual&);
};

struct EqualityConstraintFirstDerivative
{
public:
    EqualityConstraintFirstDerivative()
    {
    }
    ~EqualityConstraintFirstDerivative()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                    const std::shared_ptr< dotk::Vector<Real> > & primal_,
                    const std::shared_ptr< dotk::Vector<Real> > & delta_primal_,
                    std::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        equality_->jacobian(*primal_, *delta_primal_, *output_);
    }

private:
    EqualityConstraintFirstDerivative(const dotk::lp::EqualityConstraintFirstDerivative&);
    dotk::lp::EqualityConstraintFirstDerivative operator=(const dotk::lp::EqualityConstraintFirstDerivative&);
};

struct EqualityConstraintAdjointFirstDerivative
{
public:
    EqualityConstraintAdjointFirstDerivative()
    {
    }
    ~EqualityConstraintAdjointFirstDerivative()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::shared_ptr< dotk::Vector<Real> > & primal_,
                    const std::shared_ptr< dotk::Vector<Real> > & dual_,
                    std::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->adjointJacobian(*primal_, *dual_, *output_);
    }

private:
    EqualityConstraintAdjointFirstDerivative(const dotk::lp::EqualityConstraintAdjointFirstDerivative&);
    dotk::lp::EqualityConstraintAdjointFirstDerivative operator=(const dotk::lp::EqualityConstraintAdjointFirstDerivative&);
};

struct EqualityConstraintSecondDerivative
{
public:
    EqualityConstraintSecondDerivative()
    {
    }
    ~EqualityConstraintSecondDerivative()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                    const std::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::shared_ptr< dotk::Vector<Real> > & dual_,
                    const std::shared_ptr< dotk::Vector<Real> > & delta_primal_,
                    std::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        equality_->hessian(*state_, *dual_, *delta_primal_, *output_);
    }

private:
    EqualityConstraintSecondDerivative(const dotk::lp::EqualityConstraintSecondDerivative&);
    dotk::lp::EqualityConstraintSecondDerivative operator=(const dotk::lp::EqualityConstraintSecondDerivative&);
};


}

}

#endif /* DOTK_EQUALITYTYPELP_HPP_ */
