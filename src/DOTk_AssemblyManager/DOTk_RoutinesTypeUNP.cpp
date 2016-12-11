/*
 * DOTk_RoutinesTypeUNP.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_RoutinesTypeUNP.hpp"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

DOTk_RoutinesTypeUNP::DOTk_RoutinesTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                           const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & constraint_) :
        dotk::DOTk_AssemblyManager(),
        m_State(),
        m_Dual(),
        m_DeltaState(),
        m_DeltaDual(),
        m_StateWorkVec(),
        m_ControlWorkVec(),
        m_HessCalcWorkVec(),
        m_ObjectiveFunction(objective_),
        m_EqualityConstraint(constraint_)
{
    this->initialize(primal_);
}

DOTk_RoutinesTypeUNP::~DOTk_RoutinesTypeUNP()
{
}

Real DOTk_RoutinesTypeUNP::objective(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_)
{
    /// DOTk interface: Objective function interface. \n
    /// In: \n
    ///     variables_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///       (std::vector < Real >) \n
    /// Out: \n
    ///     objective_func_val = value of the objective function. \n
    ///       (Real) \n
    ///
    m_State->fill(0.);
    m_StateWorkVec->fill(0.);
    m_EqualityConstraint->solve(*primal_, *m_State);
    dotk::DOTk_AssemblyManager::updateEqualityConstraintEvaluationCounter();

    Real objective_func_val = m_ObjectiveFunction->value(*m_State, *primal_);
    dotk::DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter();

    return (objective_func_val);
}

void DOTk_RoutinesTypeUNP::gradient
(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
 const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    /// Reduced space interface: Assemble the reduced space gradient operator. \n
    /// In: \n
    ///     variables_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///        (std::vector < Real >) \n
    /// Out: \n
    ///     gradient_ = reduced space gradient operator. \n
    ///        (std::vector < Real >) \n
    // set right-hand-side vector for adjoint problem
    m_StateWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeState(*m_State, *primal_, *m_StateWorkVec);
    m_StateWorkVec->scale(static_cast<Real>(-1.0));

    m_Dual->fill(0.);
    m_EqualityConstraint->applyAdjointInverseJacobianState(*m_State, *primal_, *m_StateWorkVec, *m_Dual);
    dotk::DOTk_AssemblyManager::updateAdjointInverseJacobianStateCounter();

    // get equality constraint contribution to the gradient operator
    m_ControlWorkVec->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeControl(*m_State, *primal_, *m_Dual, *m_ControlWorkVec);

    // assemble gradient operator
    gradient_->update(1., *m_ControlWorkVec, 0.);
    m_ControlWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeControl(*m_State, *primal_, *m_ControlWorkVec);
    gradient_->update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);
    dotk::DOTk_AssemblyManager::updateGradientEvaluationCounter();
}

void DOTk_RoutinesTypeUNP::hessian
(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
 const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
 const std::tr1::shared_ptr<dotk::Vector<Real> > & hessian_times_vector_)
{
    /// Reduced space interface: Assemble the reduced space gradient operator. \n
    /// In: \n
    ///     variables_ = state vector, i.e. optimization parameters, unchanged on exist. \n
    ///        (std::vector < Real >) \n
    ///     trial_step_ = trial step, i.e. perturbation vector, unchanged on exist. \n
    ///        (std::vector < Real >) \n
    /// Out: \n
    ///     Hessian_times_vector_ = application of the trial step to the Hessian operator. \n
    ///        (std::vector < Real >) \n
    ///
    /* FIRST SOLVE: set right-hand-side vector (using mStateWorkVec as rhs vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_EqualityConstraint->partialDerivativeControl(*m_State, *primal_, *vector_, *m_StateWorkVec);
    m_StateWorkVec->scale(static_cast<Real>(-1.0));

    // FIRST SOLVE: Solve c_u(u(variables_); variables_) du = c_z(u(variables_); variables_) trial_step_ for du
    m_DeltaState->fill(0.);
    m_EqualityConstraint->applyInverseJacobianState(*m_State, *primal_, *m_StateWorkVec, *m_DeltaState);
    dotk::DOTk_AssemblyManager::updateInverseJacobianStateCounter();

    /* SECOND SOLVE: set right-hand-side vector (using mControlWorkVec as rhs vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_HessCalcWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeStateState(*m_State, *primal_, *m_DeltaState, *m_StateWorkVec);
    m_EqualityConstraint->partialDerivativeStateState(*m_State, *primal_, *m_Dual, *m_DeltaState, *m_HessCalcWorkVec);
    m_StateWorkVec->update(static_cast<Real>(1.0), *m_HessCalcWorkVec, 1.);

    m_HessCalcWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeStateControl(*m_State, *primal_, *vector_, *m_HessCalcWorkVec);
    m_StateWorkVec->update(static_cast<Real>(1.0), *m_HessCalcWorkVec, 1.);
    m_HessCalcWorkVec->fill(0.);
    m_EqualityConstraint->partialDerivativeStateControl(*m_State, *primal_, *m_Dual, *vector_, *m_HessCalcWorkVec);
    m_StateWorkVec->update(static_cast<Real>(1.0), *m_HessCalcWorkVec, 1.);
    m_StateWorkVec->scale(static_cast<Real>(-1.0));

    /* Solve c_u(u(variables_); variables_) dlambda = -(L_uu(u(variables_), variables_, lambda(variables_)) du
     + L_uz(u(variables_), variables_, lambda(variables_)) trial_step_) for dlambda */
    m_DeltaDual->fill(0.);
    m_EqualityConstraint->applyInverseJacobianState(*m_State, *primal_, *m_StateWorkVec, *m_DeltaDual);
    dotk::DOTk_AssemblyManager::updateInverseJacobianStateCounter();

    /* FINAL STEP: Assemble application of the trial step to the Hessian operator:
     * H*trial_step_ = L_zu(u(variables_); variables_; lambda(variables_))*du +
     * L_zz(u(variables_); variables_; lambda(variables_)) trial_step_ + c_z(u(variables_); variables_)*dlambda */
    this->computeHessianTimesVector(*primal_, *vector_, *hessian_times_vector_);
    dotk::DOTk_AssemblyManager::updateHessianEvaluationCounter();
}

void DOTk_RoutinesTypeUNP::computeHessianTimesVector(const dotk::Vector<Real> & control_,
                                                     const dotk::Vector<Real> & trial_step_,
                                                     dotk::Vector<Real> & hessian_times_vector_)
{
    hessian_times_vector_.fill(0.);
    m_ObjectiveFunction->partialDerivativeControlControl(*m_State, control_, trial_step_, hessian_times_vector_);
    m_ControlWorkVec->fill(0.);
    m_EqualityConstraint->partialDerivativeControlControl(*m_State,
                                                            control_,
                                                            *m_Dual,
                                                            trial_step_,
                                                            *m_ControlWorkVec);
    hessian_times_vector_.update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);

    // add L_zl(u(variables_); variables_; lambda(variables_))*dlambda contribution, where L denotes the Lagrangian functional
    m_ControlWorkVec->fill(0.);
    m_EqualityConstraint->adjointPartialDerivativeControl(*m_State, control_, *m_DeltaDual, *m_ControlWorkVec);
    hessian_times_vector_.update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);

    // add L_zu(u(variables_); variables_; lambda(variables_))*du contribution, where L denotes the Lagrangian functional
    m_ControlWorkVec->fill(0.);
    m_ObjectiveFunction->partialDerivativeControlState(*m_State, control_, *m_DeltaState, *m_ControlWorkVec);
    hessian_times_vector_.update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);
    m_ControlWorkVec->fill(0.);
    m_EqualityConstraint->partialDerivativeControlState(*m_State,
                                                          control_,
                                                          *m_Dual,
                                                          *m_DeltaState,
                                                          *m_ControlWorkVec);
    hessian_times_vector_.update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);
}

void DOTk_RoutinesTypeUNP::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->dual().use_count() > 0)
    {
        this->allocate(dotk::types::DUAL, primal_->dual());
    }
    else
    {
        std::perror("\n**** Error in DOTk_RoutinesTypeUNP::initialize. User did not define dual data. ABORT. ****\n");
        std::abort();
    }
    if(primal_->control().use_count() > 0)
    {
        this->allocate(dotk::types::CONTROL, primal_->control());
    }
    else
    {
        std::perror("\n**** Error in DOTk_RoutinesTypeUNP::initialize. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

void DOTk_RoutinesTypeUNP::allocate(dotk::types::variable_t type_, const std::tr1::shared_ptr<dotk::Vector<Real> > & data_)
{
    switch(type_)
    {
        case dotk::types::CONTROL:
        {
            m_ControlWorkVec = data_->clone();
            break;
        }
        case dotk::types::DUAL:
        {
            m_Dual = data_->clone();
            m_State = data_->clone();
            m_DeltaDual = data_->clone();
            m_DeltaState = data_->clone();
            m_StateWorkVec = data_->clone();
            m_HessCalcWorkVec = data_->clone();
            break;
        }
        case dotk::types::STATE:
        case dotk::types::PRIMAL:
        case dotk::types::UNDEFINED_VARIABLE:
        {
            std::perror("\n**** Error in DOTk_RoutinesTypeUNP::allocate. User did not define control data. ABORT. ****\n");
            std::abort();
            break;
        }
    }
}

}
