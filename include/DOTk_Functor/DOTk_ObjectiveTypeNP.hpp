/*
 * DOTk_ObjectiveTypeNP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OBJECTIVETYPENP_HPP_
#define DOTK_OBJECTIVETYPENP_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace nlp
{

struct ObjectiveValue
{
public:
    ObjectiveValue()
    {
    }
    ~ObjectiveValue()
    {
    }

    Real operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_) const
    {
        Real objective_function_value = operators_->value(*state_, *control_);
        return (objective_function_value);
    }

private:
    ObjectiveValue(const dotk::nlp::ObjectiveValue&);
    dotk::nlp::ObjectiveValue operator=(const dotk::nlp::ObjectiveValue&);
};

struct PartialDerivativeObjectiveControl
{
public:
    PartialDerivativeObjectiveControl()
    {
    }
    ~PartialDerivativeObjectiveControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeControl(*state_, *control_, *output_);
    }

private:
    PartialDerivativeObjectiveControl(const dotk::nlp::PartialDerivativeObjectiveControl&);
    dotk::nlp::PartialDerivativeObjectiveControl operator=(const dotk::nlp::PartialDerivativeObjectiveControl&);
};

struct PartialDerivativeObjectiveState
{
public:
    PartialDerivativeObjectiveState()
    {
    }
    ~PartialDerivativeObjectiveState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeState(*state_, *control_, *output_);
    }

private:
    PartialDerivativeObjectiveState(const dotk::nlp::PartialDerivativeObjectiveState&);
    dotk::nlp::PartialDerivativeObjectiveState operator=(const dotk::nlp::PartialDerivativeObjectiveState&);
};

struct PartialDerivativeObjectiveStateState
{
public:
    PartialDerivativeObjectiveStateState()
    {
    }
    ~PartialDerivativeObjectiveStateState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & delta_state_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeStateState(*state_, *control_, *delta_state_, *output_);
    }

private:
    PartialDerivativeObjectiveStateState(const dotk::nlp::PartialDerivativeObjectiveStateState&);
    dotk::nlp::PartialDerivativeObjectiveStateState operator=(const dotk::nlp::PartialDerivativeObjectiveStateState&);
};

struct PartialDerivativeObjectiveStateControl
{
public:
    PartialDerivativeObjectiveStateControl()
    {
    }
    ~PartialDerivativeObjectiveStateControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & delta_control_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeStateControl(*state_, *control_, *delta_control_, *output_);
    }

private:
    PartialDerivativeObjectiveStateControl(const dotk::nlp::PartialDerivativeObjectiveStateControl&);
    dotk::nlp::PartialDerivativeObjectiveStateControl operator=(const dotk::nlp::PartialDerivativeObjectiveStateControl&);
};

struct PartialDerivativeObjectiveControlControl
{
public:
    PartialDerivativeObjectiveControlControl()
    {
    }
    ~PartialDerivativeObjectiveControlControl()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & delta_control_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeControlControl(*state_, *control_, *delta_control_, *output_);
    }

private:
    PartialDerivativeObjectiveControlControl(const dotk::nlp::PartialDerivativeObjectiveControlControl&);
    dotk::nlp::PartialDerivativeObjectiveControlControl operator=(const dotk::nlp::PartialDerivativeObjectiveControlControl&);
};

struct PartialDerivativeObjectiveControlState
{
public:
    PartialDerivativeObjectiveControlState()
    {
    }
    ~PartialDerivativeObjectiveControlState()
    {
    }

    void operator()(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & state_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & delta_state_,
                    std::tr1::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->partialDerivativeControlState(*state_, *control_, *delta_state_, *output_);
    }

private:
    PartialDerivativeObjectiveControlState(const dotk::nlp::PartialDerivativeObjectiveControlState&);
    dotk::nlp::PartialDerivativeObjectiveControlState operator=(const dotk::nlp::PartialDerivativeObjectiveControlState&);
};

}

}

#endif /* DOTK_OBJECTIVETYPENP_HPP_ */
