/*
 * TRROM_ReducedBasisAssemblyMng.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include "TRROM_Matrix.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_ReducedBasis.hpp"
#include "TRROM_ReducedBasisPDE.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_ReducedBasisInterface.hpp"
#include "TRROM_ReducedBasisAssemblyMng.hpp"
#include "TRROM_ReducedBasisObjectiveOperators.hpp"

namespace trrom
{

ReducedBasisAssemblyMng::ReducedBasisAssemblyMng(const std::shared_ptr<trrom::ReducedBasisData> & data_,
                                                 const std::shared_ptr<trrom::ReducedBasisInterface> & interface_,
                                                 const std::shared_ptr<trrom::ReducedBasisObjectiveOperators> & objective_,
                                                 const std::shared_ptr<trrom::ReducedBasisPDE> & partial_differential_equation_) :
        m_UseFullNewtonHessian(true),
        m_HessianCounter(0),
        m_GradientCounter(0),
        m_ObjectiveCounter(0),
        m_LowFidelitySolveCounter(0),
        m_HighFidelitySolveCounter(0),
        m_LowFidelityAdjointSolveCounter(0),
        m_HighFidelityAdjointSolveCounter(0),
        m_LowFidelityJacobianSolveCounter(0),
        m_HighFidelityJacobianSolveCounter(0),
        m_LowFidelityAdjointJacobianSolveCounter(0),
        m_HighFidelityAdjointJacobianSolveCounter(0),
        m_Dual(data_->dual()->create()),
        m_State(data_->state()->create()),
        m_DeltaDual(data_->dual()->create()),
        m_DeltaState(data_->state()->create()),
        m_HessWorkVec(data_->state()->create()),
        m_StateWorkVec(data_->state()->create()),
        m_ControlWorkVec(data_->control()->create()),
        m_PDE(partial_differential_equation_),
        m_Objective(objective_),
        m_ReducedBasisInterface(interface_)
{
}

ReducedBasisAssemblyMng::~ReducedBasisAssemblyMng()
{
}

int ReducedBasisAssemblyMng::getHessianCounter() const
{
    return (m_HessianCounter);
}

void ReducedBasisAssemblyMng::updateHessianCounter()
{
    m_HessianCounter++;
}

int ReducedBasisAssemblyMng::getGradientCounter() const
{
    return (m_GradientCounter);
}

void ReducedBasisAssemblyMng::updateGradientCounter()
{
    m_GradientCounter++;
}

int ReducedBasisAssemblyMng::getObjectiveCounter() const
{
    return (m_ObjectiveCounter);
}

void ReducedBasisAssemblyMng::updateObjectiveCounter()
{
    m_ObjectiveCounter++;
}

int ReducedBasisAssemblyMng::getLowFidelitySolveCounter() const
{
    return (m_LowFidelitySolveCounter);
}

void ReducedBasisAssemblyMng::updateLowFidelitySolveCounter()
{
    m_LowFidelitySolveCounter++;
}

int ReducedBasisAssemblyMng::getHighFidelitySolveCounter() const
{
    return (m_HighFidelitySolveCounter);
}

void ReducedBasisAssemblyMng::updateHighFidelitySolveCounter()
{
    m_HighFidelitySolveCounter++;
}

int ReducedBasisAssemblyMng::getLowFidelityAdjointSolveCounter() const
{
    return (m_LowFidelityAdjointSolveCounter);
}

void ReducedBasisAssemblyMng::updateLowFidelityAdjointSolveCounter()
{
    m_LowFidelityAdjointSolveCounter++;
}

int ReducedBasisAssemblyMng::getHighFidelityAdjointSolveCounter() const
{
    return (m_HighFidelityAdjointSolveCounter);
}

void ReducedBasisAssemblyMng::updateHighFidelityAdjointSolveCounter()
{
    m_HighFidelityAdjointSolveCounter++;
}

int ReducedBasisAssemblyMng::getLowFidelityJacobianSolveCounter() const
{
    return (m_LowFidelityJacobianSolveCounter);
}

void ReducedBasisAssemblyMng::updateLowFidelityJacobianSolveCounter()
{
    m_LowFidelityJacobianSolveCounter++;
}

int ReducedBasisAssemblyMng::getHighFidelityJacobianSolveCounter() const
{
    return (m_HighFidelityJacobianSolveCounter);
}

void ReducedBasisAssemblyMng::updateHighFidelityJacobianSolveCounter()
{
    m_HighFidelityJacobianSolveCounter++;
}

int ReducedBasisAssemblyMng::getLowFidelityAdjointJacobianSolveCounter() const
{
    return (m_LowFidelityAdjointJacobianSolveCounter);
}

void ReducedBasisAssemblyMng::updateLowFidelityAdjointJacobianSolveCounter()
{
    m_LowFidelityAdjointJacobianSolveCounter++;
}

int ReducedBasisAssemblyMng::getHighFidelityAdjointJacobianSolveCounter() const
{
    return (m_HighFidelityAdjointJacobianSolveCounter);
}

void ReducedBasisAssemblyMng::updateHighFidelityAdjointJacobianSolveCounter()
{
    m_HighFidelityAdjointJacobianSolveCounter++;
}

double ReducedBasisAssemblyMng::objective(const std::shared_ptr<trrom::Vector<double> > & control_,
                                          const double & tolerance_,
                                          bool & inexactness_violated_)
{
    m_State->fill(0.);

    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    if(fidelity == trrom::types::LOW_FIDELITY)
    {
        this->solveLowFidelityProblem(*control_);
    }
    else
    {
        this->solveHighFidelityProblem(*control_);
    }

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

void ReducedBasisAssemblyMng::gradient(const std::shared_ptr<trrom::Vector<double> > & control_,
                                       const std::shared_ptr<trrom::Vector<double> > & gradient_,
                                       const double & tolerance_,
                                       bool & inexactness_violated_)
{
    m_Dual->fill(0.);
    m_StateWorkVec->fill(0.);

    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    if(fidelity == trrom::types::LOW_FIDELITY)
    {
        this->solveLowFidelityAdjointProblem(*control_);
    }
    else
    {
        this->solveHighFidelityAdjointProblem(*control_);
    }

    // Compute equality constraint contribution to the gradient operator
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControl(*m_State, *control_, *m_Dual, *m_ControlWorkVec);

    // Assemble gradient operator
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

void ReducedBasisAssemblyMng::hessian(const std::shared_ptr<trrom::Vector<double> > & control_,
                                      const std::shared_ptr<trrom::Vector<double> > & vector_,
                                      const std::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                                      const double & tolerance_,
                                      bool & inexactness_violated_)
{
    if(m_UseFullNewtonHessian == true)
    {
        this->computeFullNewtonHessian(*control_, *vector_, *hess_times_vec_);
    }
    else
    {
        this->computeGaussNewtonHessian(*control_, *vector_, *hess_times_vec_);
    }
    this->updateHessianCounter();
}

void ReducedBasisAssemblyMng::computeHessianTimesVector(const trrom::Vector<double> & input_,
                                                        const trrom::Vector<double> & trial_step_,
                                                        trrom::Vector<double> & hess_times_vec_)
{
    hess_times_vec_.fill(0.);
    m_Objective->partialDerivativeControlControl(*m_State, input_, trial_step_, hess_times_vec_);
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControlControl(*m_State, input_, *m_Dual, trial_step_, *m_ControlWorkVec);
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
    m_PDE->adjointPartialDerivativeControlState(*m_State, input_, *m_Dual, *m_DeltaState, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);
}

void ReducedBasisAssemblyMng::applyInverseJacobianState(const trrom::Vector<double> & control_,
                                                        const trrom::Vector<double> & rhs_)
{
    m_DeltaState->fill(0.);
    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    if(fidelity == trrom::types::LOW_FIDELITY)
    {
        m_ReducedBasisInterface->applyLowFidelityInverseJacobian(rhs_, *m_DeltaState);
        this->updateLowFidelityJacobianSolveCounter();
    }
    else
    {
        m_PDE->applyInverseJacobianState(*m_State, control_, rhs_, *m_DeltaState);
        this->updateHighFidelityJacobianSolveCounter();
    }

}

void ReducedBasisAssemblyMng::applyInverseAdjointJacobianState(const trrom::Vector<double> & control_,
                                                               const trrom::Vector<double> & rhs_)
{
    m_DeltaDual->fill(0.);
    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    if(fidelity == trrom::types::LOW_FIDELITY)
    {
        m_ReducedBasisInterface->applyLowFidelityInverseAdjointJacobian(rhs_, *m_DeltaDual);
        this->updateLowFidelityAdjointJacobianSolveCounter();
    }
    else
    {
        m_PDE->applyInverseAdjointJacobianState(*m_State, control_, rhs_, *m_DeltaDual);
        this->updateHighFidelityAdjointJacobianSolveCounter();
    }

}

void ReducedBasisAssemblyMng::computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                                       const trrom::Vector<double> & vector_,
                                                       trrom::Vector<double> & hess_times_vec_)
{
    /* FIRST SOLVE: set right-hand-side Vector (using mStateWorkVec as rhs Vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_PDE->partialDerivativeControl(*m_State, control_, vector_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1.);

    // FIRST SOLVE: Solve c_u(u(variables_); variables_) du = c_z(u(variables_); variables_) trial_step_ for du
    this->applyInverseJacobianState(control_, *m_StateWorkVec);

    /* SECOND SOLVE: set right-hand-side Vector (using mControlWorkVec as rhs Vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeStateState(*m_State, control_, *m_DeltaState, *m_StateWorkVec);
    m_PDE->adjointPartialDerivativeStateState(*m_State, control_, *m_Dual, *m_DeltaState, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);

    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeStateControl(*m_State, control_, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_HessWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeStateControl(*m_State, control_, *m_Dual, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_StateWorkVec->scale(-1.);

    /* Solve c_u(u(variables_); variables_) dlambda = -(L_uu(u(variables_), variables_, lambda(variables_)) du
     + L_uz(u(variables_), variables_, lambda(variables_)) trial_step_) for dlambda */
    this->applyInverseAdjointJacobianState(control_, *m_StateWorkVec);

    /* FINAL STEP: Assemble application of the trial step to the Hessian operator:
     * H*trial_step_ = L_zu(u(variables_); variables_; lambda(variables_))*du +
     * L_zz(u(variables_); variables_; lambda(variables_)) trial_step_ + c_z(u(variables_); variables_)*dlambda */
    this->computeHessianTimesVector(control_, vector_, hess_times_vec_);
}

void ReducedBasisAssemblyMng::computeGaussNewtonHessian(const trrom::Vector<double> & control_,
                                                        const trrom::Vector<double> & vector_,
                                                        trrom::Vector<double> & hess_times_vec_)
{
    /*!
     Compute Gauss Newton Hessian approximation: \mathcal{L}_{zz}(\mathbf{u}(\mathbf{z}), \mathbf{z};
     \lambda(\mathbf{z}))\delta{z} contribution, where \mathbf{u} denotes state vector, \mathbf{z}
     denotes control vector, and \mathcal{L} denotes the Lagrangian functional.
    */
    hess_times_vec_.fill(0.);
    m_Objective->partialDerivativeControlControl(*m_State, control_, vector_, hess_times_vec_);
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControlControl(*m_State, control_, *m_Dual, vector_, *m_ControlWorkVec);
    hess_times_vec_.update(1., *m_ControlWorkVec, 1.);
}

trrom::types::fidelity_t ReducedBasisAssemblyMng::fidelity() const
{
    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    return (fidelity);
}

void ReducedBasisAssemblyMng::fidelity(trrom::types::fidelity_t input_)
{
    m_Objective->fidelity(input_);
    m_ReducedBasisInterface->fidelity(input_);
}

void ReducedBasisAssemblyMng::useGaussNewtonHessian()
{
    m_UseFullNewtonHessian = false;
}

void ReducedBasisAssemblyMng::updateLowFidelityModel()
{
    // Update orthonormal bases (state, dual, and left hand side bases)
    m_ReducedBasisInterface->updateOrthonormalBases();
    // Update left hand side Discrete Empirical Interpolation Method (DEIM) active indices
    m_ReducedBasisInterface->updateLeftHandSideDeimDataStructures();
    // Update reduced right hand side vector
    m_ReducedBasisInterface->updateReducedStateRightHandSide();
}

void ReducedBasisAssemblyMng::solveHighFidelityProblem(const trrom::Vector<double> & control_)
{
    /*! Solve for \mathbf{u}(\mathbf{z})\in\mathbb{R}^{n_u}, \mathbf{K}(\mathbf{z})\mathbf{u} = \mathbf{f}. If the parametric
     * reduced-order model (low-fidelity model) is disabled, the user is expected to provide the current nonaffine, parameter
     * dependent matrix snapshot by calling the respective store snapshot functionality in the ReducedBasisInterface class. If
     * the low-fidelity model is enabled, the user is expected to provide the respective reduced snapshot for the nonaffine,
     * parameter dependent matrices and right-hand side vectors  */
    m_PDE->solve(control_, *m_State, *m_ReducedBasisInterface->data());
    m_ReducedBasisInterface->storeStateSnapshot(*m_State);
    m_ReducedBasisInterface->storeLeftHandSideSnapshot(*m_ReducedBasisInterface->data()->getLeftHandSideSnapshot());
    this->updateHighFidelitySolveCounter();
}

void ReducedBasisAssemblyMng::solveLowFidelityProblem(const trrom::Vector<double> & control_)
{
    /*! Compute current left hand side matrix approximation using the Active Indices computed using the Discrete Empirical
     * Interpolation Method (DEIM). Since the low-fidelity model is enabled, the third-party application code is required
     * to only provide the reduced left hand matrix approximation in vectorized format. Thus, the user is not required to
     * solve the low-fidelity system of equations. The low-fidelity system of equations will be solved in-situ using the
     * (default or custom) solver interface. */
    m_PDE->solve(control_, *m_State, *m_ReducedBasisInterface->data());
    m_ReducedBasisInterface->solveLowFidelityProblem(*m_State);
    this->updateLowFidelitySolveCounter();
}

void ReducedBasisAssemblyMng::solveHighFidelityAdjointProblem(const trrom::Vector<double> & control_)
{
    // Compute right hand side (RHS) vector of adjoint system of equations.
    m_Objective->partialDerivativeState(*m_State, control_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1);
    // Solve adjoint system of equations, \mathbf{K}(z)\lambda = \mathbf{f}_{\lambda},
    // where \mathbf{f}_{\lambda} = -\frac{\partial{J}(\mathbf{u},\mathbf{z})}
    // {\partial\mathbf{u}}
    m_PDE->applyInverseAdjointJacobianState(*m_State, control_, *m_StateWorkVec, *m_Dual);
    m_ReducedBasisInterface->storeDualSnapshot(*m_Dual);
    this->updateHighFidelityAdjointSolveCounter();
}

void ReducedBasisAssemblyMng::solveLowFidelityAdjointProblem(const trrom::Vector<double> & control_)
{
    // Compute reduced right hand side (RHS) vector of adjoint system of equations.
    m_Objective->partialDerivativeState(*m_State, control_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1);
    m_ReducedBasisInterface->solveLowFidelityAdjointProblem(*m_StateWorkVec, *m_Dual);
    this->updateLowFidelityAdjointSolveCounter();
}

}
