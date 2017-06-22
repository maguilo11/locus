/*
 * DOTk_RoutinesTypeENP.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RoutinesTypeENP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

DOTk_RoutinesTypeENP::DOTk_RoutinesTypeENP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> >& objective_,
                                           const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> >& equality_) :
        dotk::DOTk_AssemblyManager(),
        m_StateWorkVector(),
        m_ControlWorkVector(),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(equality_)
{
    this->initialize(primal_);
}

DOTk_RoutinesTypeENP::~DOTk_RoutinesTypeENP()
{
}

Real DOTk_RoutinesTypeENP::objective(const std::shared_ptr<dotk::Vector<Real> > & primal_)
{
    /// DOTk interface: Objective function for nonlinear programming problems. \n
    ///
    Real value = m_ObjectiveFunction->value(*primal_->state(), *primal_->control());

    dotk::DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();

    return (value);
}
void DOTk_RoutinesTypeENP::equalityConstraint(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                              const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    /// DOTk interface: Evaluate equality constraint for nonlinear programming problems. \n
    ///
    m_EqualityConstraint->residual(*primal_->state(), *primal_->control(), *output_);

    dotk::DOTk_AssemblyManager::updateEqualityConstraintEvaluationCounter();
}

void DOTk_RoutinesTypeENP::gradient(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & dual_,
                                    const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    /// DOTk interface: Assemble gradient operator for nonlinear programming problems. \n
    ///
    output_->state()->fill(0.);
    m_ObjectiveFunction->partialDerivativeState(*primal_->state(), *primal_->control(), *output_->state());
    m_StateWorkVector->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeState(*primal_->state(),
                                                        *primal_->control(),
                                                        *dual_,
                                                        *m_StateWorkVector);
    output_->state()->update(static_cast<Real>(1.0), *m_StateWorkVector, 1.);

    output_->control()->fill(0.);
    m_ObjectiveFunction->partialDerivativeControl(*primal_->state(), *primal_->control(), *output_->control());

    m_ControlWorkVector->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeControl(*primal_->state(),
                                                          *primal_->control(),
                                                          *dual_,
                                                          *m_ControlWorkVector);
    output_->control()->update(static_cast<Real>(1.0), *m_ControlWorkVector, 1.);

    dotk::DOTk_AssemblyManager::updateGradientEvaluationCounter();
}

void DOTk_RoutinesTypeENP::jacobian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    /// DOTk interface: Compute the application of the perturbation vector to the Jacobian operator \n
    ///                 for nonlinear programming problems. \n
    ///
    output_->fill(0.);
    m_EqualityConstraint->partialDerivativeState(*primal_->state(),
                                                 *primal_->control(),
                                                 *delta_primal_->state(),
                                                 *output_);

    m_StateWorkVector->fill(0.);
    m_EqualityConstraint->partialDerivativeControl(*primal_->state(),
                                                   *primal_->control(),
                                                   *delta_primal_->control(),
                                                   *m_StateWorkVector);
    output_->update(static_cast<Real>(1.0), *m_StateWorkVector, 1.);

    dotk::DOTk_AssemblyManager::updateJacobianEvaluationCounter();
}

void DOTk_RoutinesTypeENP::adjointJacobian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                           const std::shared_ptr<dotk::Vector<Real> > & dual_,
                                           const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    /// DOTk interface: Compute the application of the perturbation vector to the Jacobian operator \n
    ///                  for nonlinear programming problems. \n
    ///
    output_->state()->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeState(*primal_->state(),
                                                        *primal_->control(),
                                                        *dual_,
                                                        *(output_->state()));

    output_->control()->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeControl(*primal_->state(),
                                                          *primal_->control(),
                                                          *dual_,
                                                          *output_->control());

    dotk::DOTk_AssemblyManager::updateAdjointJacobianEvaluationCounter();
}

void DOTk_RoutinesTypeENP::hessian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                   const std::shared_ptr<dotk::Vector<Real> > & dual_,
                                   const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                                   const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    /// DOTk interface: Compute the application of the perturbation vector to the Hessian operator \n
    ///                 for nonlinear programming problems. \n
    ///

    // Hessian Block UU, where Hessian = [H_uu H_uz; H_zu H_zz]
    output_->state()->fill(0.);
    m_ObjectiveFunction->partialDerivativeStateState(*primal_->state(),
                                                     *primal_->control(),
                                                     *delta_primal_->state(),
                                                     *(output_->state()));
    m_StateWorkVector->fill(0.);
    m_EqualityConstraint->partialDerivativeStateState(*primal_->state(),
                                                      *primal_->control(),
                                                      *dual_,
                                                      *delta_primal_->state(),
                                                      *(m_StateWorkVector));
    output_->state()->update(static_cast<Real>(1.0), *m_StateWorkVector, 1.);

    // Hessian Block UZ, i.e. Hessian Block 12, where Hessian = [H_uu H_uz; H_zu H_zz]
    output_->control()->fill(0.);
    m_StateWorkVector->fill(0.);
    m_ObjectiveFunction->partialDerivativeStateControl(*primal_->state(),
                                                       *primal_->control(),
                                                       *delta_primal_->control(),
                                                       *(m_StateWorkVector));
    output_->state()->update(static_cast<Real>(1.0), *m_StateWorkVector, 1.);
    m_StateWorkVector->fill(0.);
    m_EqualityConstraint->partialDerivativeStateControl(*primal_->state(),
                                                        *primal_->control(),
                                                        *dual_,
                                                        *delta_primal_->control(),
                                                        *(m_StateWorkVector));
    output_->state()->update(static_cast<Real>(1.0), *m_StateWorkVector, 1.);

    // Hessian Block ZU, where Hessian = [H_uu H_uz; H_zu H_zz]
    m_ObjectiveFunction->partialDerivativeControlState(*primal_->state(),
                                                       *primal_->control(),
                                                       *delta_primal_->state(),
                                                       *output_->control());
    m_ControlWorkVector->fill(0.);
    m_EqualityConstraint->partialDerivativeControlState(*primal_->state(),
                                                        *primal_->control(),
                                                        *dual_,
                                                        *delta_primal_->state(),
                                                        *(m_ControlWorkVector));

    // Hessian Block ZZ, where Hessian = [H_uu H_uz; H_zu H_zz]
    output_->control()->update(static_cast<Real>(1.0), *m_ControlWorkVector, 1.);
    m_ControlWorkVector->fill(0.);
    m_ObjectiveFunction->partialDerivativeControlControl(*primal_->state(),
                                                         *primal_->control(),
                                                         *delta_primal_->control(),
                                                         *(m_ControlWorkVector));
    output_->control()->update(static_cast<Real>(1.0), *m_ControlWorkVector, 1.);
    m_ControlWorkVector->fill(0.);
    m_EqualityConstraint->partialDerivativeControlControl(*primal_->state(),
                                                          *primal_->control(),
                                                          *dual_,
                                                          *delta_primal_->control(),
                                                          *(m_ControlWorkVector));
    output_->control()->update(static_cast<Real>(1.0), *m_ControlWorkVector, 1.);

    dotk::DOTk_AssemblyManager::updateHessianEvaluationCounter();
}

void DOTk_RoutinesTypeENP::initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->state().use_count() > 0)
    {
        m_StateWorkVector = primal_->state()->clone();
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> STATE vector is NULL. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }

    if(primal_->control().use_count() > 0)
    {
        m_ControlWorkVector = primal_->control()->clone();
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> CONTROL vector is NULL. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }
}

}
