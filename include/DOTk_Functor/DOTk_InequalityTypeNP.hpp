/*
 * DOTk_InequalityTypeNP.hpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEQUALITYTYPENP_HPP_
#define DOTK_INEQUALITYTYPENP_HPP_

#include "DOTk_InequalityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace nlp
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
                    const std::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::shared_ptr<dotk::Vector<Real> > & control_) const
    {
        Real value = inequality_->value(*state_, *control_);

        return (value);
    }

private:
    InequalityConstraintEvaluate(const dotk::nlp::InequalityConstraintEvaluate&);
    dotk::nlp::InequalityConstraintEvaluate operator=(const dotk::nlp::InequalityConstraintEvaluate&);
};

struct InequalityConstraintFirstDerivativeWrtState
{
public:
    InequalityConstraintFirstDerivativeWrtState()
    {
    }
    ~InequalityConstraintFirstDerivativeWrtState()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_,
                    const std::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::shared_ptr<dotk::Vector<Real> > & control_,
                    const std::shared_ptr<dotk::Vector<Real> > & derivative_) const
    {
        inequality_->partialDerivativeState(*state_, *control_, *derivative_);
    }

private:
    InequalityConstraintFirstDerivativeWrtState(const dotk::nlp::InequalityConstraintFirstDerivativeWrtState&);
    dotk::nlp::InequalityConstraintFirstDerivativeWrtState operator=(const dotk::nlp::InequalityConstraintFirstDerivativeWrtState&);
};

struct InequalityConstraintFirstDerivativeWrtControl
{
public:
    InequalityConstraintFirstDerivativeWrtControl()
    {
    }
    ~InequalityConstraintFirstDerivativeWrtControl()
    {
    }

    void operator()(const std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_,
                    const std::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::shared_ptr<dotk::Vector<Real> > & control_,
                    const std::shared_ptr<dotk::Vector<Real> > & derivative_) const
    {
        inequality_->partialDerivativeControl(*state_, *control_, *derivative_);
    }

private:
    InequalityConstraintFirstDerivativeWrtControl(const dotk::nlp::InequalityConstraintFirstDerivativeWrtControl&);
    dotk::nlp::InequalityConstraintFirstDerivativeWrtControl operator=(const dotk::nlp::InequalityConstraintFirstDerivativeWrtControl&);
};

}

}

#endif /* DOTK_INEQUALITYTYPENP_HPP_ */
