/*
 * DOTk_EqualityTypeNP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_EQUALITYTYPENP_HPP_
#define DOTK_EQUALITYTYPENP_HPP_

#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace nlp
{

struct EqualityConstraintResidual
{
public:
    EqualityConstraintResidual()
    {
    }
    ~EqualityConstraintResidual()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->residual(*state_, *control_, *output_);
    }

private:
    EqualityConstraintResidual(const dotk::nlp::EqualityConstraintResidual&);
    dotk::nlp::EqualityConstraintResidual operator=(const dotk::nlp::EqualityConstraintResidual&);
};

struct EqualityConstraintFirstDerivativeState
{
public:
    EqualityConstraintFirstDerivativeState()
    {
    }
    ~EqualityConstraintFirstDerivativeState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & delta_state_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeState(*state_, *control_, *delta_state_, *output_);
    }

private:
    EqualityConstraintFirstDerivativeState(const dotk::nlp::EqualityConstraintFirstDerivativeState&);
    dotk::nlp::EqualityConstraintFirstDerivativeState operator=(const dotk::nlp::EqualityConstraintFirstDerivativeState&);
};

struct EqualityConstraintFirstDerivativeControl
{
public:
    EqualityConstraintFirstDerivativeControl()
    {
    }
    ~EqualityConstraintFirstDerivativeControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & delta_control_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeControl(*state_, *control_, *delta_control_, *output_);
    }

private:
    EqualityConstraintFirstDerivativeControl(const dotk::nlp::EqualityConstraintFirstDerivativeControl&);
    dotk::nlp::EqualityConstraintFirstDerivativeControl operator=(const dotk::nlp::EqualityConstraintFirstDerivativeControl&);
};

struct EqualityConstraintAdjointFirstDerivativeState
{
public:
    EqualityConstraintAdjointFirstDerivativeState()
    {
    }
    ~EqualityConstraintAdjointFirstDerivativeState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & dual_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->adjointPartialDerivativeState(*state_, *control_, *dual_, *output_);
    }

private:
    EqualityConstraintAdjointFirstDerivativeState(const dotk::nlp::EqualityConstraintAdjointFirstDerivativeState&);
    dotk::nlp::EqualityConstraintAdjointFirstDerivativeState operator=(const dotk::nlp::EqualityConstraintAdjointFirstDerivativeState&);
};

struct EqualityConstraintAdjointFirstDerivativeControl
{
public:
    EqualityConstraintAdjointFirstDerivativeControl()
    {
    }
    ~EqualityConstraintAdjointFirstDerivativeControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & dual_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->adjointPartialDerivativeControl(*state_, *control_, *dual_, *output_);
    }

private:
    EqualityConstraintAdjointFirstDerivativeControl(const dotk::nlp::EqualityConstraintAdjointFirstDerivativeControl&);
    dotk::nlp::EqualityConstraintAdjointFirstDerivativeControl operator=(const dotk::nlp::EqualityConstraintAdjointFirstDerivativeControl&);
};

struct EqualityConstraintSecondDerivativeControlControl
{
public:
    EqualityConstraintSecondDerivativeControlControl()
    {
    }
    ~EqualityConstraintSecondDerivativeControlControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & dual_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & delta_control_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeControlControl(*state_, *control_, *dual_, *delta_control_, *output_);
    }

private:
    EqualityConstraintSecondDerivativeControlControl(const dotk::nlp::EqualityConstraintSecondDerivativeControlControl&);
    dotk::nlp::EqualityConstraintSecondDerivativeControlControl operator=(const dotk::nlp::EqualityConstraintSecondDerivativeControlControl&);
};

struct EqualityConstraintSecondDerivativeControlState
{
public:
    EqualityConstraintSecondDerivativeControlState()
    {
    }
    ~EqualityConstraintSecondDerivativeControlState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & dual_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & delta_state_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeControlState(*state_, *control_, *dual_, *delta_state_, *output_);
    }

private:
    EqualityConstraintSecondDerivativeControlState(const dotk::nlp::EqualityConstraintSecondDerivativeControlState&);
    dotk::nlp::EqualityConstraintSecondDerivativeControlState operator=(const dotk::nlp::EqualityConstraintSecondDerivativeControlState&);
};

struct EqualityConstraintSecondDerivativeStateState
{
public:
    EqualityConstraintSecondDerivativeStateState()
    {
    }
    ~EqualityConstraintSecondDerivativeStateState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & dual_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & delta_state_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeStateState(*state_, *control_, *dual_, *delta_state_, *output_);
    }

private:
    EqualityConstraintSecondDerivativeStateState(const dotk::nlp::EqualityConstraintSecondDerivativeStateState&);
    dotk::nlp::EqualityConstraintSecondDerivativeStateState operator=(const dotk::nlp::EqualityConstraintSecondDerivativeStateState&);
};

struct EqualityConstraintSecondDerivativeStateControl
{
public:
    EqualityConstraintSecondDerivativeStateControl()
    {
    }
    ~EqualityConstraintSecondDerivativeStateControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & operators_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & dual_,
                    const std::tr1::shared_ptr< dotk::Vector<Real> > & delta_state_,
                    std::tr1::shared_ptr< dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeStateControl(*state_, *control_, *dual_, *delta_state_, *output_);
    }

private:
    EqualityConstraintSecondDerivativeStateControl(const dotk::nlp::EqualityConstraintSecondDerivativeStateControl&);
    dotk::nlp::EqualityConstraintSecondDerivativeStateControl operator=(const dotk::nlp::EqualityConstraintSecondDerivativeStateControl&);
};

}

}

#endif /* DOTK_EQUALITYTYPENP_HPP_ */
