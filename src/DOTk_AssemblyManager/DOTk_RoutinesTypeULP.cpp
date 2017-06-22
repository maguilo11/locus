/*
 * DOTk_RoutinesTypeULP.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_RoutinesTypeULP.hpp"

#include "vector.hpp"
#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

DOTk_RoutinesTypeULP::DOTk_RoutinesTypeULP(const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> >& objective_) :
        dotk::DOTk_AssemblyManager(),
        m_ObjectiveFunction(objective_)
{
}

DOTk_RoutinesTypeULP::~DOTk_RoutinesTypeULP()
{
}

Real DOTk_RoutinesTypeULP::objective(const std::shared_ptr<dotk::Vector<Real> > & primal_)
{
    /// DOTk interface: Objective function interface. \n
    /// In: \n
    ///     primal_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///       (std::vector < Real >) \n
    /// Out: \n
    ///     objective_func_val = value of the objective function. \n
    ///       (Real) \n
    ///
    Real objective_func_val = m_ObjectiveFunction->value(*primal_);
    dotk::DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();

    return (objective_func_val);
}

void DOTk_RoutinesTypeULP::objective(const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_,
                                     const std::shared_ptr<dotk::Vector<Real> > & values_)
{
    /// DOTk interface: Objective function parallel interface. \n
    /// In: \n
    ///     primal_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///       (std::vector< std::vector<Real> >) \n
    /// In/Out: \n
    ///     objective_func_val_ = value of the objective function. \n
    ///       (std::vector < Real >) \n
    ///
    m_ObjectiveFunction->value(primal_, values_);
    dotk::DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();
}

void DOTk_RoutinesTypeULP::objective(const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_plus_,
                                     const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_minus_,
                                     const std::shared_ptr<dotk::Vector<Real> > & values_plus_,
                                     const std::shared_ptr<dotk::Vector<Real> > & values_minus_)
{
    /// DOTk interface: Interface to parallel central difference interface. This routine is used when a parallel central
    ///                 difference scheme is applied to compute the gradient operator. \n
    /// In: \n
    ///     vars_plus_ = forward perturbed state vector, unchanged on exist. \n
    ///       (std::vector< std::map<dotk::types::variable_t, std::shared_ptr< dotk::Vector<Real> > > >) \n
    ///     vars_minus_ = backward perturbed state vector, unchanged on exist. \n
    ///       (std::vector< std::map<dotk::types::variable_t, std::shared_ptr< dotk::Vector<Real> > > >) \n
    /// In/Out: \n
    ///     fval_plus_ = place holder for the evaluation of the objective function for each forward perturbed state vector. \n
    ///       (std::map<dotk::types::variable_t, std::shared_ptr< dotk::Vector<Real> > >) \n
    ///     fval_minus_ = place holder for the evaluation of the objective function for each backward perturbed state vector. \n
    ///       (std::map<dotk::types::variable_t, std::shared_ptr< dotk::Vector<Real> > >) \n
    ///
    m_ObjectiveFunction->value(primal_plus_, primal_minus_, values_plus_, values_minus_);
    dotk::DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();
}

void DOTk_RoutinesTypeULP::gradient(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    /// Reduced space interface: Assemble the reduced space gradient operator. \n
    /// In: \n
    ///     primal_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///        (std::vector < Real >) \n
    /// Out: \n
    ///     gradient_ = reduced space gradient operator. \n
    ///        (std::vector < Real >) \n
    ///
    // objective function contribution
    m_ObjectiveFunction->gradient(*primal_, *gradient_);
    dotk::DOTk_AssemblyManager::updateGradientEvaluationCounter();
}

void DOTk_RoutinesTypeULP::hessian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                   const std::shared_ptr<dotk::Vector<Real> > & trial_step_,
                                   const std::shared_ptr<dotk::Vector<Real> > & Hess_times_vector_)
{
    /// Reduced space interface: Assemble the reduced space gradient operator. \n
    /// In: \n
    ///     primal_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///        (std::vector < Real >) \n
    ///     trial_step_ = perturbation vector, unchanged on exist. \n
    ///        (std::vector < Real >) \n
    /// Out: \n
    ///     Hessian_times_vector_ = application of the trial step to the Hessian operator. \n
    ///        (std::vector < Real >) \n
    ///
    m_ObjectiveFunction->hessian(*primal_, *trial_step_, *Hess_times_vector_);
    dotk::DOTk_AssemblyManager::updateHessianEvaluationCounter();
}

}
