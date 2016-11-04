/*
 * DOTk_RoutinesTypeELP.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_RoutinesTypeELP.hpp"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"


namespace dotk
{

DOTk_RoutinesTypeELP::DOTk_RoutinesTypeELP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> >& objective_,
                                           const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> >& constraint_) :
        dotk::DOTk_AssemblyManager(),
        m_WorkVector(primal_->control()->clone()),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(constraint_)
{
}

DOTk_RoutinesTypeELP::~DOTk_RoutinesTypeELP()
{
}

Real DOTk_RoutinesTypeELP::objective(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_)
{
    /// DOTk interface: Objective function for linear programming problems. \n
    ///
    Real value = m_ObjectiveFunction->value(*primal_->control());
    dotk::DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();

    return (value);
}

void DOTk_RoutinesTypeELP::equalityConstraint(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                              const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    /// DOTk interface: Evaluate equality constraint for linear programming problems. \n
    ///
    output_->fill(0);
    m_EqualityConstraint->residual(*primal_->control(), *output_);

    dotk::DOTk_AssemblyManager::updateEqualityConstraintEvaluationCounter();
}

void DOTk_RoutinesTypeELP::gradient(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & dual_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    /// DOTk interface: Assemble gradient operator for linear programming problems. \n
    ///
    output_->fill(0.);
    m_ObjectiveFunction->gradient(*primal_->control(), *output_->control());

    m_WorkVector->fill(0.);
    m_EqualityConstraint->adjointJacobian(*primal_->control(), *dual_, *m_WorkVector);

    output_->control()->axpy(static_cast<Real>(1.0), *m_WorkVector);

    dotk::DOTk_AssemblyManager::updateGradientEvaluationCounter();
}

void DOTk_RoutinesTypeELP::jacobian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    /// DOTk interface: Compute the application of the perturbation vector to the Jacobian operator \n
    ///                 for linear programming problems. \n
    ///
    output_->fill(0.);
    m_EqualityConstraint->jacobian(*primal_->control(), *delta_primal_->control(), *output_);

    dotk::DOTk_AssemblyManager::updateJacobianEvaluationCounter();
}

void DOTk_RoutinesTypeELP::adjointJacobian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & dual_,
                                           const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    /// DOTk interface: Compute the application of the perturbation vector to the Jacobian operator \n
    ///                  for linear programming problems. \n
    ///
    output_->fill(0.);
    m_EqualityConstraint->adjointJacobian(*primal_->control(), *dual_, *output_);

    dotk::DOTk_AssemblyManager::updateAdjointJacobianEvaluationCounter();
}

void DOTk_RoutinesTypeELP::hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & dual_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    /// DOTk interface: Compute the application of the perturbation vector to the Hessian operator \n
    ///                 for linear programming problems. \n
    ///
    output_->fill(0.);
    m_ObjectiveFunction->hessian(*primal_->control(), *delta_primal_->control(), *output_->control());

    m_WorkVector->fill(0.);
    m_EqualityConstraint->hessian(*primal_->control(), *dual_, *delta_primal_->control(), *m_WorkVector);

    output_->control()->axpy(static_cast<Real>(1.0), *m_WorkVector);
    dotk::DOTk_AssemblyManager::updateHessianEvaluationCounter();
}

}
