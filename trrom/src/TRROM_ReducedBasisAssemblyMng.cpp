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
#include "TRROM_ReducedBasisObjective.hpp"
#include "TRROM_ReducedBasisInterface.hpp"
#include "TRROM_ReducedBasisAssemblyMng.hpp"

namespace trrom
{

ReducedBasisAssemblyMng::ReducedBasisAssemblyMng(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                                                 const std::tr1::shared_ptr<trrom::ReducedBasisInterface> & interface_,
                                                 const std::tr1::shared_ptr<trrom::ReducedBasisPDE> & pde_,
                                                 const std::tr1::shared_ptr<trrom::ReducedBasisObjective> & objective_) :
        m_UseFullNewtonHessian(true),
        m_HessianCounter(0),
        m_GradientCounter(0),
        m_ObjectiveCounter(0),
        m_LowFidelitySolveCounter(0),
        m_HighFidelitySolveCounter(0),
        m_LowFidelityAdjointSolveCounter(0),
        m_HighFidelityAdjointSolveCounter(0),
        m_Dual(data_->dual()->create()),
        m_State(data_->state()->create()),
        m_StateWorkVec(data_->state()->create()),
        m_ControlWorkVec(data_->control()->create()),
        m_LeftHandSideSnapshot(data_->createLeftHandSideSnapshotCopy()),
        m_PDE(pde_),
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

double ReducedBasisAssemblyMng::objective(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                          const double & tolerance_,
                                          bool & inexactness_violated_)
{
    m_State->fill(0.);

    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    if(fidelity == trrom::types::LOW_FIDELITY)
    {
        this->solveLowFidelityPDE(control_);
    }
    else
    {
        this->solveHighFidelityPDE(control_);
    }

    double value = m_Objective->value(*m_State, *control_, tolerance_, inexactness_violated_);
    this->updateObjectiveCounter();

    return (value);
}

void ReducedBasisAssemblyMng::gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                       const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_,
                                       const double & tolerance_,
                                       bool & inexactness_violated_)
{
    m_Dual->fill(0.);
    m_StateWorkVec->fill(0.);

    trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
    if(fidelity == trrom::types::LOW_FIDELITY)
    {
        this->solveLowFidelityAdjointPDE(control_);
    }
    else
    {
        this->solveHighFidelityAdjointPDE(control_);
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

    // Check gradient inexactness tolerance
    inexactness_violated_ = m_Objective->checkGradientInexactness(*m_State, *control_, *gradient_, tolerance_);
}

void ReducedBasisAssemblyMng::hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                      const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                      const std::tr1::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
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

void ReducedBasisAssemblyMng::computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                                       const trrom::Vector<double> & vector_,
                                                       trrom::Vector<double> & hess_times_vec_)
{
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

void ReducedBasisAssemblyMng::solveHighFidelityPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_)
{
    /*! Solve for \mathbf{u}(\mathbf{z})\in\mathbb{R}^{n_u}, \mathbf{K}(\mathbf{z})\mathbf{u} = \mathbf{f}. If the parametric
     * reduced-order model (low-fidelity model) is disabled, the user is expected to provide the current nonaffine, parameter
     * dependent matrix snapshot by calling the respective store snapshot functionality in the ReducedBasisInterface class. If
     * the low-fidelity model is enabled, the user is expected to provide the respective reduced snapshot for the nonaffine,
     * parameter dependent matrices and right-hand side vectors  */
    m_LeftHandSideSnapshot->fill(0.);
    m_PDE->solve(*control_, *m_State, *m_ReducedBasisInterface->data());
    m_ReducedBasisInterface->storeStateSnapshot(*m_State);
    this->updateHighFidelitySolveCounter();
}

void ReducedBasisAssemblyMng::solveLowFidelityPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_)
{
    /*! Compute current left hand side matrix approximation using the Active Indices computed using the Discrete Empirical
     * Interpolation Method (DEIM). Since the low-fidelity model is enabled, the third-party application code is required
     * to only provide the reduced left hand matrix approximation in vectorized format. Thus, the user is not required to
     * solve the low-fidelity system of equations. The low-fidelity system of equations will be solved in-situ using the
     * (default or custom) solver interface. */
    m_LeftHandSideSnapshot->fill(0);
    m_PDE->solve(*control_, *m_State, *m_ReducedBasisInterface->data());
    m_ReducedBasisInterface->solveLowFidelityPDE(m_State);
    this->updateLowFidelitySolveCounter();
}

void ReducedBasisAssemblyMng::solveHighFidelityAdjointPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_)
{
    // Compute right hand side (RHS) vector of adjoint system of equations.
    m_Objective->partialDerivativeState(*m_State, *control_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1);
    // Solve adjoint system of equations, \mathbf{K}(z)\lambda = \mathbf{f}_{\lambda},
    // where \mathbf{f}_{\lambda} = -\frac{\partial{J}(\mathbf{u},\mathbf{z})}
    // {\partial\mathbf{u}}
    m_PDE->applyAdjointInverseJacobianState(*m_State, *control_, *m_StateWorkVec, *m_Dual);
    m_ReducedBasisInterface->storeDualSnapshot(*m_Dual);
    this->updateHighFidelityAdjointSolveCounter();
}

void ReducedBasisAssemblyMng::solveLowFidelityAdjointPDE(const std::tr1::shared_ptr<trrom::Vector<double> > & control_)
{
    // Compute reduced right hand side (RHS) vector of adjoint system of equations.
    m_Objective->partialDerivativeState(*m_State, *control_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1);
    m_ReducedBasisInterface->solveLowFidelityAdjointPDE(m_StateWorkVec, m_Dual);
    this->updateLowFidelityAdjointSolveCounter();
}

}
