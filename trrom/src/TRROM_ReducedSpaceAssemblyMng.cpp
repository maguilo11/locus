/*
 * TRROM_ReducedSpaceAssemblyMng.cpp
 *
 *  Created on: Aug 18, 2016
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_PDE_Constraint.hpp"
#include "TRROM_ReducedSpaceAssemblyMng.hpp"
#include "TRROM_ReducedObjectiveOperators.hpp"

namespace trrom
{

ReducedSpaceAssemblyMng::ReducedSpaceAssemblyMng(const std::tr1::shared_ptr<trrom::Data> & input_,
                                                 const std::tr1::shared_ptr<trrom::PDE_Constraint> & pde_,
                                                 const std::tr1::shared_ptr<trrom::ReducedObjectiveOperators> & objective_) :
        m_FullNewton(true),
        m_HessianCounter(0),
        m_GradientCounter(0),
        m_ObjectiveCounter(0),
        m_EqualityEvaluationCounter(0),
        m_InverseJacobianStateCounter(0),
        m_AdjointInverseJacobianStateCounter(0),
        m_Dual(input_->dual()->create()),
        m_State(input_->state()->create()),
        m_DeltaDual(input_->dual()->create()),
        m_DeltaState(input_->state()->create()),
        m_HessWorkVec(input_->state()->create()),
        m_StateWorkVec(input_->state()->create()),
        m_ControlWorkVec(input_->control()->create()),
        m_PDE(pde_),
        m_Objective(objective_)
{
}

ReducedSpaceAssemblyMng::~ReducedSpaceAssemblyMng()
{
}

int ReducedSpaceAssemblyMng::getHessianCounter() const
{
    return (m_HessianCounter);
}

void ReducedSpaceAssemblyMng::updateHessianCounter()
{
    m_HessianCounter++;
}

int ReducedSpaceAssemblyMng::getGradientCounter() const
{
    return (m_GradientCounter);
}

void ReducedSpaceAssemblyMng::updateGradientCounter()
{
    m_GradientCounter++;
}

int ReducedSpaceAssemblyMng::getObjectiveCounter() const
{
    return (m_ObjectiveCounter);
}

void ReducedSpaceAssemblyMng::updateObjectiveCounter()
{
    m_ObjectiveCounter++;
}

double ReducedSpaceAssemblyMng::objective(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                          const double & tolerance_,
                                          bool & inexactness_violated_)
{
    // Solve for state\in\mathbb{R}^{n_{u}}, \mathbf{u}(\mathbf{z})
    m_State->fill(0.);
    m_StateWorkVec->fill(0.);
    m_PDE->solve(*control_, *m_State);
    this->updateEqualityEvaluationCounter();

    // Evaluate objective function, J(\mathbf{u}(\mathbf{z}),\mathbf{z})
    double value = m_Objective->value(*m_State, *control_);
    this->updateObjectiveCounter();

    // check objective inexactness tolerance
    inexactness_violated_ = false;
    double objective_error = m_Objective->evaluateObjectiveInexactness(*m_State, *control_);
    if(objective_error > tolerance_)
    {
        inexactness_violated_ = true;
    }
    return (value);
}

void ReducedSpaceAssemblyMng::gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                       const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_,
                                       const double & tolerance_,
                                       bool & inexactness_violated_)
{
    m_StateWorkVec->fill(0.);
    m_Objective->partialDerivativeState(*m_State, *control_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1.);

    m_Dual->fill(0.);
    m_PDE->applyInverseAdjointJacobianState(*m_State, *control_, *m_StateWorkVec, *m_Dual);
    this->updateInverseAdjointJacobianStateCounter();

    // get equality constraint contribution to the gradient operator
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControl(*m_State, *control_, *m_Dual, *m_ControlWorkVec);

    // assemble gradient operator
    gradient_->update(1., *m_ControlWorkVec, 0.);
    m_ControlWorkVec->fill(0.);
    m_Objective->partialDerivativeControl(*m_State, *control_, *m_ControlWorkVec);
    gradient_->update(1., *m_ControlWorkVec, 1.);
    this->updateGradientCounter();

    // check gradient inexactness tolerance
    inexactness_violated_ = false;
    double gradient_error = m_Objective->evaluateGradientInexactness(*m_State, *control_);
    if(gradient_error > tolerance_)
    {
        inexactness_violated_ = true;
    }
}

void ReducedSpaceAssemblyMng::hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                      const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                      const std::tr1::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                                      const double & tolerance_,
                                      bool & inexactness_violated_)
{
    /// Reduced space interface: Assemble the reduced space gradient operator. \n
    /// In: \n
    ///     variables_ = state Vector, i.e. optimization parameters, unchanged on exist. \n
    ///        (std::Vector < double >) \n
    ///     trial_step_ = trial step, i.e. perturbation Vector, unchanged on exist. \n
    ///        (std::Vector < double >) \n
    /// Out: \n
    ///     Hessian_times_vector_ = application of the trial step to the Hessian operator. \n
    ///        (std::Vector < double >) \n
    ///
    if(m_FullNewton == true)
    {
        this->computeFullNewtonHessian(*control_, *vector_, *hess_times_vec_);
    }
    else
    {
        this->computeGaussNewtonHessian(*control_, *vector_, *hess_times_vec_);
    }
    this->updateHessianCounter();
}

int ReducedSpaceAssemblyMng::getEqualityEvaluationCounter() const
{
    return (m_EqualityEvaluationCounter);
}

void ReducedSpaceAssemblyMng::updateEqualityEvaluationCounter()
{
    m_EqualityEvaluationCounter++;
}

int ReducedSpaceAssemblyMng::getInverseJacobianStateCounter() const
{
    return (m_InverseJacobianStateCounter);
}

void ReducedSpaceAssemblyMng::updateInverseJacobianStateCounter()
{
    m_InverseJacobianStateCounter++;
}

int ReducedSpaceAssemblyMng::getAdjointInverseJacobianStateCounter() const
{
    return (m_AdjointInverseJacobianStateCounter);
}

void ReducedSpaceAssemblyMng::updateInverseAdjointJacobianStateCounter()
{
    m_AdjointInverseJacobianStateCounter++;
}

void ReducedSpaceAssemblyMng::computeHessianTimesVector(const trrom::Vector<double> & input_,
                                                        const trrom::Vector<double> & trial_step_,
                                                        trrom::Vector<double> & hess_times_vec_)
{
    hess_times_vec_.fill(0.);
    m_Objective->partialDerivativeControlControl(*m_State, input_, trial_step_, hess_times_vec_);
    m_ControlWorkVec->fill(0.);
    m_PDE->partialDerivativeControlControl(*m_State, input_, *m_Dual, trial_step_, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);

    // add L_zl(u(variables_); variables_; lambda(variables_))*dlambda contribution, where L denotes the Lagrangian functional
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControl(*m_State, input_, *m_DeltaDual, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);

    // add L_zu(u(variables_); variables_; lambda(variables_))*du contribution, where L denotes the Lagrangian functional
    m_ControlWorkVec->fill(0.);
    m_Objective->partialDerivativeControlState(*m_State, input_, *m_DeltaState, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);
    m_ControlWorkVec->fill(0.);
    m_PDE->partialDerivativeControlState(*m_State, input_, *m_Dual, *m_DeltaState, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);
}

void ReducedSpaceAssemblyMng::computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                                       const trrom::Vector<double> & vector_,
                                                       trrom::Vector<double> & hess_times_vec_)
{
    /* FIRST SOLVE: set right-hand-side Vector (using mStateWorkVec as rhs Vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_PDE->partialDerivativeControl(*m_State, control_, vector_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1.);

    // FIRST SOLVE: Solve c_u(u(variables_); variables_) du = c_z(u(variables_); variables_) trial_step_ for du
    m_DeltaState->fill(0.);
    m_PDE->applyInverseJacobianState(*m_State, control_, *m_StateWorkVec, *m_DeltaState);
    this->updateInverseJacobianStateCounter();

    /* SECOND SOLVE: set right-hand-side Vector (using mControlWorkVec as rhs Vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeStateState(*m_State, control_, *m_DeltaState, *m_StateWorkVec);
    m_PDE->partialDerivativeStateState(*m_State, control_, *m_Dual, *m_DeltaState, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);

    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeStateControl(*m_State, control_, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_HessWorkVec->fill(0.);
    m_PDE->partialDerivativeStateControl(*m_State, control_, *m_Dual, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_StateWorkVec->scale(-1.);

    /* Solve c_u(u(variables_); variables_) dlambda = -(L_uu(u(variables_), variables_, lambda(variables_)) du
     + L_uz(u(variables_), variables_, lambda(variables_)) trial_step_) for dlambda */
    m_DeltaDual->fill(0.);
    m_PDE->applyInverseAdjointJacobianState(*m_State, control_, *m_StateWorkVec, *m_DeltaDual);
    this->updateInverseAdjointJacobianStateCounter();

    /* FINAL STEP: Assemble application of the trial step to the Hessian operator:
     * H*trial_step_ = L_zu(u(variables_); variables_; lambda(variables_))*du +
     * L_zz(u(variables_); variables_; lambda(variables_)) trial_step_ + c_z(u(variables_); variables_)*dlambda */
    this->computeHessianTimesVector(control_, vector_, hess_times_vec_);
}

void ReducedSpaceAssemblyMng::computeGaussNewtonHessian(const trrom::Vector<double> & control_,
                                                        const trrom::Vector<double> & vector_,
                                                        trrom::Vector<double> & hess_times_vec_)
{
    ///
    /// Compute Gauss Newton Hessian approximation: \mathcal{L}_{zz}(\mathbf{u}(\mathbf{z}), \mathbf{z};
    /// \lambda(\mathbf{z}))\delta{z} contribution, where \mathbf{u} denotes state vector, \mathbf{z}
    /// denotes control vector, and \mathcal{L} denotes the Lagrangian functional.
    ///
    hess_times_vec_.fill(0.);
    m_Objective->partialDerivativeControlControl(*m_State, control_, vector_, hess_times_vec_);
    m_ControlWorkVec->fill(0.);
    m_PDE->partialDerivativeControlControl(*m_State, control_, *m_Dual, vector_, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);
}

}
