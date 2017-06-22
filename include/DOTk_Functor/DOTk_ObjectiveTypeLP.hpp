/*
 * DOTk_ObjectiveTypeLP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OBJECTIVETYPELP_HPP_
#define DOTK_OBJECTIVETYPELP_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace lp
{

struct ObjectiveFunctionEvaluate
{
public:
    ObjectiveFunctionEvaluate()
    {
    }
    ~ObjectiveFunctionEvaluate()
    {
    }

    Real operator()(const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_) const
    {
        Real objective_function_value = operators_->value(*primal_);
        return (objective_function_value);
    }

private:
    ObjectiveFunctionEvaluate(const dotk::lp::ObjectiveFunctionEvaluate&);
    dotk::lp::ObjectiveFunctionEvaluate operator=(const dotk::lp::ObjectiveFunctionEvaluate&);
};

struct ObjectiveFunctionFirstDerivative
{
public:
    explicit ObjectiveFunctionFirstDerivative(dotk::types::variable_t type_) :
            m_Codomain(type_)
    {
    }
    ~ObjectiveFunctionFirstDerivative()
    {
    }

    dotk::types::variable_t getCodomain() const
    {
        return (m_Codomain);
    }
    void operator()(const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_,
                    std::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->gradient(*primal_, *output_);
    }

private:
    dotk::types::variable_t m_Codomain;

private:
    ObjectiveFunctionFirstDerivative(const dotk::lp::ObjectiveFunctionFirstDerivative&);
    dotk::lp::ObjectiveFunctionFirstDerivative operator=(const dotk::lp::ObjectiveFunctionFirstDerivative&);
};

struct ObjectiveFunctionSecondDerivative
{
public:
    explicit ObjectiveFunctionSecondDerivative(dotk::types::variable_t type_) :
            m_Codomain(type_)
    {
    }
    ~ObjectiveFunctionSecondDerivative()
    {
    }

    dotk::types::variable_t getCodomain() const
    {
        return (m_Codomain);
    }
    void operator()(const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_,
                    const std::shared_ptr<dotk::Vector<Real> > & primal_,
                    const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                    std::shared_ptr<dotk::Vector<Real> > & output_) const
    {
        operators_->hessian(*primal_, *delta_primal_, *output_);
    }

private:
    dotk::types::variable_t m_Codomain;

private:
    ObjectiveFunctionSecondDerivative(const dotk::lp::ObjectiveFunctionSecondDerivative&);
    dotk::lp::ObjectiveFunctionSecondDerivative operator=(const dotk::lp::ObjectiveFunctionSecondDerivative&);
};

}

}

#endif /* DOTK_OBJECTIVETYPELP_HPP_ */
