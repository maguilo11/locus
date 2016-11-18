/*
 * TRROM_AugmentedLagrangianTypeNP.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_PDE_Constraint.hpp"
#include "TRROM_InequalityOperators.hpp"
#include "TRROM_ReducedObjectiveOperators.hpp"
#include "TRROM_AugmentedLagrangianTypeNP.hpp"

namespace trrom
{

AugmentedLagrangianTypeNP::AugmentedLagrangianTypeNP(const std::tr1::shared_ptr<trrom::Data> & input_,
                                                     const std::tr1::shared_ptr<trrom::PDE_Constraint> & pde_,
                                                     const std::tr1::shared_ptr<trrom::ReducedObjectiveOperators> & objective_,
                                                     const std::vector<std::tr1::shared_ptr<trrom::InequalityOperators> > & inequality_) :
        m_FullNewton(true),
        m_HessianCounter(0),
        m_GradientCounter(0),
        m_ObjectiveCounter(0),
        m_InequalityCounter(0),
        m_InequalityGradientCounter(0),
        m_EqualityEvaluationCounter(0),
        m_InverseJacobianStateCounter(0),
        m_AdjointInverseJacobianStateCounter(0),
        m_Penalty(1),
        m_MinPenalty(1e-10),
        m_PenaltyScaling(0.1),
        m_NormLagrangianGradient(0.),
        m_NormInequalityConstraints(0.),
        m_State(input_->state()->create()),
        m_Slacks(input_->slacks()->create()),
        m_DeltaState(input_->state()->create()),
        m_HessWorkVec(input_->state()->create()),
        m_StateWorkVec(input_->state()->create()),
        m_ObjectiveDual(input_->dual()->create()),
        m_ControlWorkVec(input_->control()->create()),
        m_GradientWorkVec(input_->control()->create()),
        m_LagrangianGradient(input_->control()->create()),
        m_ObjectiveDeltaDual(input_->dual()->create()),
        m_LagrangeMultipliers(input_->slacks()->create()),
        m_InequalityConstraintValues(input_->slacks()->create()),
        m_CurrentInequalityConstraintValues(input_->slacks()->create()),
        m_InequalityDual(),
        m_InequalityDeltaDual(),
        m_PDE(pde_),
        m_Objective(objective_),
        m_Inequality(inequality_.begin(), inequality_.end())
{
}

AugmentedLagrangianTypeNP::~AugmentedLagrangianTypeNP()
{
}

int AugmentedLagrangianTypeNP::getHessianCounter() const
{
    return (m_HessianCounter);
}

void AugmentedLagrangianTypeNP::updateHessianCounter()
{
    m_HessianCounter++;
}

int AugmentedLagrangianTypeNP::getGradientCounter() const
{
    return (m_GradientCounter);
}

void AugmentedLagrangianTypeNP::updateGradientCounter()
{
    m_GradientCounter++;
}

int AugmentedLagrangianTypeNP::getObjectiveCounter() const
{
    return (m_ObjectiveCounter);
}

void AugmentedLagrangianTypeNP::updateObjectiveCounter()
{
    m_ObjectiveCounter++;
}

int AugmentedLagrangianTypeNP::getInequalityCounter() const
{
    return (m_InequalityCounter);
}

void AugmentedLagrangianTypeNP::updateInequalityCounter()
{
    m_InequalityCounter++;
}

int AugmentedLagrangianTypeNP::getInequalityGradientCounter() const
{
    return (m_InequalityGradientCounter);
}

void AugmentedLagrangianTypeNP::updateInequalityGradientCounter()
{
    m_InequalityGradientCounter++;
}

double AugmentedLagrangianTypeNP::getPenalty() const
{
    return (m_Penalty);
}

double AugmentedLagrangianTypeNP::getNormLagrangianGradient() const
{
    return (m_NormLagrangianGradient);
}

double AugmentedLagrangianTypeNP::getNormInequalityConstraints() const
{
    return (m_NormInequalityConstraints);
}

void AugmentedLagrangianTypeNP::updateInequalityConstraintValues()
{
    ///
    /// Update current inequality constraint values.
    ///
    int num_inequality_constraints = m_InequalityConstraintValues->size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        (*m_CurrentInequalityConstraintValues)[index] = (*m_InequalityConstraintValues)[index];
    }
    m_NormInequalityConstraints = m_CurrentInequalityConstraintValues->norm();
}

double AugmentedLagrangianTypeNP::objective(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                            const double & tolerance_,
                                            bool & inexactness_violated_)
{
    // Solve for state, \mathbf{u}(\mathbf{z}), \mathbf{A}(\mathbf{z})\mathbf{u} = \mathbf{f}
    m_State->fill(0.);
    m_StateWorkVec->fill(0.);
    m_PDE->solve(*control_, *m_State);
    this->updateEqualityEvaluationCounter();

    // Evaluate objective function, f(\mathbf{u}(\mathbf{z}),\mathbf{z})
    double objective_value = m_Objective->value(*m_State, *control_);
    this->updateObjectiveCounter();

    // Evaluate inequality constraint residual, h(\mathbf{u}(\mathbf{z}),\mathbf{z})
    int num_inequality_constraints = m_InequalityConstraintValues->size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        (*m_InequalityConstraintValues)[index] = m_Inequality[index]->value(*m_State, *control_)
                - m_Inequality[index]->bound();
        this->updateInequalityCounter();
    }

    // Evaluate Lagrangian functional, \ell(\mathbf{u}(\mathbf{z}),\mathbf{z},\mu) =
    //   f(\mathbf{u}(\mathbf{z}),\mathbf{z}) + \mu^{T}h(\mathbf{u}(\mathbf{z}),\mathbf{z})
    double lagrange_multipliers_dot_inequality_residuals = m_LagrangeMultipliers->dot(*m_InequalityConstraintValues);
    double lagrangian_value = objective_value + lagrange_multipliers_dot_inequality_residuals;

    // Evaluate augmented Lagrangian functional, \mathcal{L}(\mathbf{z}),\mathbf{z},\mu) =
    //   \ell(\mathbf{u}(\mathbf{z}),\mathbf{z},\mu) +
    //   \frac{1}{2\beta}(h(\mathbf{u}(\mathbf{z}),\mathbf{z})^{T}h(\mathbf{u}(\mathbf{z}),\mathbf{z})),
    //   where \beta\in\mathbb{R} denotes a penalty parameter
    double inequality_residual_dot_inequality_residual =
            m_InequalityConstraintValues->dot(*m_InequalityConstraintValues);
    double augmented_lagrangian_value = lagrangian_value
            + ((static_cast<double>(0.5) / m_Penalty) * inequality_residual_dot_inequality_residual);

    // Check objective inexactness tolerance. NOTE: THIS IS ONLY CHECKING OPTIMALITY, FEASIBILITY IS NOT CHECKED.
    inexactness_violated_ = m_Objective->checkObjectiveInexactness(tolerance_, objective_value, *m_State, *control_);

    return (augmented_lagrangian_value);
}

void AugmentedLagrangianTypeNP::gradient(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                         const std::tr1::shared_ptr<trrom::Vector<double> > & gradient_,
                                         const double & tolerance_,
                                         bool & inexactness_violated_)
{
    // Compute objective function gradient, \nabla{f}(\mathbf{u}(\mathbf{z}),\mathbf{z})
    this->computeObjectiveGradient(*control_);

    // Compute inequality constraint gradient, \nabla{h}(\mathbf{u}(\mathbf{z}),\mathbf{z})
    this->computeInequalityConstraintGradient(*control_, *gradient_);

    // Check objective inexactness tolerance. NOTE: THIS IS ONLY CHECKING OPTIMALITY, FEASIBILITY IS NOT CHECKED.
    inexactness_violated_ = m_Objective->checkGradientInexactness(tolerance_,
                                                                  *m_State,
                                                                  *control_,
                                                                  *m_LagrangianGradient);

    this->updateGradientCounter();
}

void AugmentedLagrangianTypeNP::hessian(const std::tr1::shared_ptr<trrom::Vector<double> > & control_,
                                        const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                                        const std::tr1::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                                        const double & tolerance_,
                                        bool & inexactness_violated_)
{
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

bool AugmentedLagrangianTypeNP::updateLagrangeMultipliers()
{
    bool penalty_below_tolerance = false;
    double current_penalty = m_Penalty;
    m_Penalty = m_PenaltyScaling * m_Penalty;
    int num_inequality_constraints = m_CurrentInequalityConstraintValues->size();
    if(m_Penalty >= m_MinPenalty)
    {
        for(int index = 0; index < num_inequality_constraints; ++ index)
        {
            double alpha = static_cast<double>(1.) / current_penalty;
            double beta = alpha * (*m_CurrentInequalityConstraintValues)[index];
            (*m_LagrangeMultipliers)[index] = (*m_LagrangeMultipliers)[index] + beta;
        }
    }
    else
    {
        penalty_below_tolerance = true;
    }
    return (penalty_below_tolerance);
}

int AugmentedLagrangianTypeNP::getEqualityEvaluationCounter() const
{
    return (m_EqualityEvaluationCounter);
}

void AugmentedLagrangianTypeNP::updateEqualityEvaluationCounter()
{
    m_EqualityEvaluationCounter++;
}

int AugmentedLagrangianTypeNP::getInverseJacobianStateCounter() const
{
    return (m_InverseJacobianStateCounter);
}

void AugmentedLagrangianTypeNP::updateInverseJacobianStateCounter()
{
    m_InverseJacobianStateCounter++;
}

int AugmentedLagrangianTypeNP::getAdjointInverseJacobianStateCounter() const
{
    return (m_AdjointInverseJacobianStateCounter);
}

void AugmentedLagrangianTypeNP::updateInverseAdjointJacobianStateCounter()
{
    m_AdjointInverseJacobianStateCounter++;
}

void AugmentedLagrangianTypeNP::setMinPenalty(double input_)
{
    m_MinPenalty = input_;
}

void AugmentedLagrangianTypeNP::setPenaltyScaling(double input_)
{
    m_PenaltyScaling = input_;
}

void AugmentedLagrangianTypeNP::useGaussNewtonHessian()
{
    m_FullNewton = false;
}

void AugmentedLagrangianTypeNP::computeObjectiveGradient(const trrom::Vector<double> & control_)
{
    m_StateWorkVec->fill(0.);
    m_Objective->partialDerivativeState(*m_State, control_, *m_StateWorkVec);
    m_StateWorkVec->scale(-1.);

    m_ObjectiveDual->fill(0.);
    m_PDE->applyAdjointInverseJacobianState(*m_State, control_, *m_StateWorkVec, *m_ObjectiveDual);
    this->updateInverseAdjointJacobianStateCounter();

    // get equality constraint contribution to the gradient operator
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControl(*m_State, control_, *m_ObjectiveDual, *m_ControlWorkVec);

    // assemble gradient operator
    m_LagrangianGradient->update(1., *m_ControlWorkVec, 0.);
    m_ControlWorkVec->fill(0.);
    m_Objective->partialDerivativeControl(*m_State, control_, *m_ControlWorkVec);
    m_LagrangianGradient->update(1., *m_ControlWorkVec, 1.);
}

void AugmentedLagrangianTypeNP::computeInequalityConstraintGradient(const trrom::Vector<double> & control_,
                                                                    trrom::Vector<double> & gradient_)
{
    gradient_.fill(0.);
    double one_over_penalty = static_cast<double>(1.) / m_Penalty;
    int num_inequality_constraints = m_CurrentInequalityConstraintValues->size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        // Compute the partial derivative of the inequality constraint with respect to the state vector (\mathbf{u})
        m_StateWorkVec->fill(0.);
        m_Inequality[index]->partialDerivativeState(*m_State, control_, *m_StateWorkVec);
        m_StateWorkVec->scale(-1.);

        // Solve adjoint equations if the inequality constraint is nonlinear
        m_InequalityDual[index]->fill(0.);
        m_PDE->applyAdjointInverseJacobianState(*m_State, control_, *m_StateWorkVec, *(m_InequalityDual[index]));
        this->updateInverseAdjointJacobianStateCounter();

        // Get equality constraint contribution to the gradient operator if the inequality constraint is nonlinear
        m_ControlWorkVec->fill(0.);
        m_PDE->adjointPartialDerivativeControl(*m_State, control_, *(m_InequalityDual[index]), *m_ControlWorkVec);
        m_GradientWorkVec->update(1., *m_ControlWorkVec, 0.);

        // Compute the partial derivative of the inequality constraint with respect to the controls,
        //      i.e. \frac{\partial h_i}{\partial\mathbf{z}}
        m_ControlWorkVec->fill(0.);
        m_Inequality[index]->partialDerivativeControl(*m_State, control_, *m_ControlWorkVec);
        m_GradientWorkVec->update(1., *m_ControlWorkVec, 1.);

        // Add contribution from inequality constraint to Lagrangian gradient
        m_LagrangianGradient->update((*m_LagrangeMultipliers)[index], *m_GradientWorkVec, 1.);

        // Compute i-th inequality constraint contribution from: \mu*h_i(\mathbf{u}(\mathbf{z}),\mathbf{z})
        //      \frac{\partial h_i(\mathbf{u}(\mathbf{z}),\mathbf{z})}{\partial\mathbf{z}}
        double alpha = one_over_penalty * (*m_CurrentInequalityConstraintValues)[index];
        gradient_.update(alpha, *m_GradientWorkVec, 1.);

        this->updateInequalityGradientCounter();
    }
    // Compute augmented Lagrangian gradient
    gradient_.update(1., *m_LagrangianGradient, 1.);
    m_NormLagrangianGradient = m_LagrangianGradient->norm();
}

void AugmentedLagrangianTypeNP::computeObjectiveHessianDual(const trrom::Vector<double> & control_,
                                                            const trrom::Vector<double> & vector_)
{
    /* Solve \frac{\partial{g}^{\ast}}{\partial\mathbf{u}}\Delta\lambda^{f} = -left[
     * \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{u}\partial\mathbf{z}}\lambda^f\right)
     * \Delta\mathbf{z} + \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{u}^2}\right)\Delta
     * \mathbf{u} + \frac{\partial^2 f}{\partial\mathbf{u}\partial\mathbf{z}}\Delta\mathbf{z}
     * + \frac{\partial^2 f}{\partial\mathbf{u}^2}\Delta\mathbf{u}\right] for \Delta\lambda^{f}
     * \in\mathbb{R}^{n_{\lambda^f}}, where g \equiv g(\mathbf{u}(\mathbf{z}),\mathbf{z}) denotes
     * the equality constraint (pde) and f \equiv f(\mathbf{u}(\mathbf{z}),\mathbf{z}) denotes
     * the objective function. */

    m_StateWorkVec->fill(0.);
    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeStateState(*m_State, control_, *m_DeltaState, *m_StateWorkVec);
    m_PDE->partialDerivativeStateState(*m_State, control_, *m_ObjectiveDual, *m_DeltaState, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);

    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeStateControl(*m_State, control_, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_HessWorkVec->fill(0.);
    m_PDE->partialDerivativeStateControl(*m_State, control_, *m_ObjectiveDual, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_StateWorkVec->scale(-1.);

    m_ObjectiveDeltaDual->fill(0.);
    m_PDE->applyInverseJacobianState(*m_State, control_, *m_StateWorkVec, *m_ObjectiveDeltaDual);
    this->updateInverseAdjointJacobianStateCounter();
}

void AugmentedLagrangianTypeNP::computeObjectiveHessianTimesVector(const trrom::Vector<double> & control_,
                                                                   const trrom::Vector<double> & vector_,
                                                                   trrom::Vector<double> & hess_times_vector_)
{
    /*
     * Apply vector to objective function Hessian operator, \nabla^2{f}\Delta\mathbf{z}, as follows:
     *      \frac{\partial^2 f}{\partial\mathbf{z}^2}\Delta\mathbf{z} + \frac{\partial^2 f}{\partial\mathbf{z}
     *      \partial\mathbf{u}}\Delta\mathbf{u} + \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{z}^2}\lambda^f\right)
     *      \Delta\mathbf{z} + \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{z}\partial\mathbf{u}}\lambda^f\right)
     *      \Delta\mathbf{u} + \frac{\partial g^{\ast}}{\partial\mathbf{z}}\Delta\lambda^f
     */
    m_HessWorkVec->fill(0.);
    m_Objective->partialDerivativeControlControl(*m_State, control_, vector_, *m_HessWorkVec);
    m_ControlWorkVec->fill(0.);
    m_PDE->partialDerivativeControlControl(*m_State, control_, *m_ObjectiveDual, vector_, *m_ControlWorkVec);
    m_HessWorkVec->update(1., *m_ControlWorkVec, 1.);

    // add L_zl(u(variables_); variables_; lambda(variables_))*dlambda contribution, where L denotes the Lagrangian functional
    m_ControlWorkVec->fill(0.);
    m_PDE->adjointPartialDerivativeControl(*m_State, control_, *m_ObjectiveDeltaDual, *m_ControlWorkVec);
    m_HessWorkVec->update(1., *m_ControlWorkVec, 1.);

    // add L_zu(u(variables_); variables_; lambda(variables_))*du contribution, where L denotes the Lagrangian functional
    m_ControlWorkVec->fill(0.);
    m_Objective->partialDerivativeControlState(*m_State, control_, *m_DeltaState, *m_ControlWorkVec);
    m_HessWorkVec->update(1., *m_ControlWorkVec, 1.);
    m_ControlWorkVec->fill(0.);
    m_PDE->partialDerivativeControlState(*m_State, control_, *m_ObjectiveDual, *m_DeltaState, *m_ControlWorkVec);
    m_HessWorkVec->update(1., *m_ControlWorkVec, 1.);

    hess_times_vector_.update(1., *m_HessWorkVec, 0.);
}

void AugmentedLagrangianTypeNP::computeInequalityConstraintHessianDual(const int & index_,
                                                                       const trrom::Vector<double> & control_,
                                                                       const trrom::Vector<double> & vector_)
{
    /** Solve \frac{\partial{g}^{\ast}}{\partial\mathbf{u}}\Delta\lambda^{h} = -left[
     * \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{u}\partial\mathbf{z}}\lambda^h\right)
     * \Delta\mathbf{z} + \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{u}^2}\right)\Delta
     * \mathbf{u} + \frac{\partial^2 h}{\partial\mathbf{u}\partial\mathbf{z}}\Delta\mathbf{z}
     * + \frac{\partial^2 h}{\partial\mathbf{u}^2}\Delta\mathbf{u}\right] for \Delta\lambda^{h}
     * \in\mathbb{R}^{n_{\lambda^h}}, where g \equiv g(\mathbf{u}(\mathbf{z}),\mathbf{z})
     * denotes the pde constraint and h \equiv h(\mathbf{u}(\mathbf{z}),\mathbf{z}) denotes
     * the inequality constraint. */

    m_StateWorkVec->fill(0.);
    m_HessWorkVec->fill(0.);
    m_Inequality[index_]->partialDerivativeStateState(*m_State, control_, *m_DeltaState, *m_StateWorkVec);
    m_PDE->partialDerivativeStateState(*m_State, control_, *m_InequalityDual[index_], *m_DeltaState, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);

    m_HessWorkVec->fill(0.);
    m_Inequality[index_]->partialDerivativeStateControl(*m_State, control_, vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_HessWorkVec->fill(0.);
    m_PDE->partialDerivativeStateControl(*m_State, control_, *m_InequalityDual[index_], vector_, *m_HessWorkVec);
    m_StateWorkVec->update(1., *m_HessWorkVec, 1.);
    m_StateWorkVec->scale(-1.);

    m_InequalityDeltaDual[index_]->fill(0.);
    m_PDE->applyInverseJacobianState(*m_State, control_, *m_StateWorkVec, *m_InequalityDeltaDual[index_]);
    this->updateInverseAdjointJacobianStateCounter();
}

void AugmentedLagrangianTypeNP::computeInequalityHessianTimesVector(const trrom::Vector<double> & control_,
                                                                    const trrom::Vector<double> & vector_,
                                                                    trrom::Vector<double> & hess_times_vector_)
{
    /**
     * Let \alpha_i = \lambda_i + \mu*h_i(\mathbf{u}(\mathbf{z}),\mathbf{z}), where \lambda_i\in\mathbb{R} is
     * the i-th lagrange multiplier associated with inequality constraint h_i(\mathbf{u}(\mathbf{z}),\mathbf{z})
     * and \beta_i\in\mathbb{R}\colon\beta_i = \mu\left(\frac{\partial h_i}{\partial\mathbf{z}}\Delta\mathbf{z}
     * + \frac{\partial h_i}{\partial\mathbf{u}}\Delta\mathbf{u}\right). Recall that \mu\in\mathbb{R} denotes
     * a penalty parameter. Then, the application of a vector to the i-th inequality constraint Hessian operator,
     * \nabla^2{h}\Delta\mathbf{z}, is defined as follows:
     *      \alpha_i\left(\frac{\partial^2 h_i}{\partial\mathbf{z}^2} + \frac{\partial^2 h_i}
     *      {\partial\mathbf{z}\partial\mathbf{u}}\Delta\mathbf{u}\right) + \alpha_i\left(\left(
     *      \frac{\partial^2 g^{\ast}}{\partial\mathbf{z}^2}\lambda_i^h\right)\Delta\mathbf{z} +
     *      \left(\frac{\partial^2 g^{\ast}}{\partial\mathbf{z}\partial\mathbf{u}}\lambda_i^h\right)
     *      \Delta\mathbf{u}\right) + \beta_i\left(\frac{\partial h_i}{\partial\mathbf{z}} + \left(
     *      \frac{\partial g^{\ast}}{\partial\mathbf{z}}\lambda_i^h\right)\right) + \alpha_i\left(
     *      \frac{\partial g^{\ast}}{\partial\mathbf{z}}\lambda_i^h\right)
     */
    int num_constraints = m_InequalityDual.size();
    double one_over_penalty = static_cast<double>(1.) / m_Penalty;
    for(int index = 0; index < num_constraints; ++ index)
    {
        double alpha = (*m_LagrangeMultipliers)[index]
                + (one_over_penalty * (*m_CurrentInequalityConstraintValues)[index]);
        /*
         * Compute \alpha_i\left(\frac{\partial^2 h_i}{\partial\mathbf{z}^2} + \left(\frac{\partial^2 g^{\ast}}
         * {\partial\amhbf{z}^2}\lambda_i^h\right)\right)\Delta{z}. Here, \Delta{z} denotes the trial direction.
         */
        m_HessWorkVec->fill(0.);
        m_Inequality[index]->partialDerivativeControlControl(*m_State, control_, vector_, *m_HessWorkVec);
        m_HessWorkVec->scale(alpha);
        m_ControlWorkVec->fill(0.);
        m_PDE->partialDerivativeControlControl(*m_State,
                                               control_,
                                               *m_InequalityDual[index],
                                               vector_,
                                               *m_ControlWorkVec);
        m_HessWorkVec->update(alpha, *m_ControlWorkVec, 1.);

        /*
         * Add contribution from \mathcal{L}_{\mathbf{z}\lambda}\left(\mathbf{u}(\mathbf{z}), \mathbf{z}; \lambda^h_i
         * (\mathbf{z})\right)\Delta\lambda_i^h = \alpha_i\left(\frac{\partial{g}^{\ast}}{\partial\mathbf{z}}\Delta
         * \lambda_i^h\right), where \mathcal{L} denotes the augmented Lagrangian functional.
         */
        m_GradientWorkVec->fill(0.);
        m_PDE->adjointPartialDerivativeControl(*m_State, control_, *m_InequalityDeltaDual[index], *m_GradientWorkVec);
        m_HessWorkVec->update(alpha, *m_GradientWorkVec, 1.);

        /* Add contribution from \mathcal{L}_{zu}\left(\mathbf{u}(\mathbf{z}),\mathbf{z};\lambda(\mathbf{z})\right)
         * \Delta\mathbf{u} = \alpha_i\left(\frac{\partial^2 h_i}{\partial\mathbf{z}\partial\mathbf{u}} + \left(
         * \frac{\partial^2 g^{\ast}}{\partial\mathbf{z}\partial\mathbf{u}}\lambda_i^h\right)\right)\Delta\mathbf{u},
         * where \mathcal{L} denotes the augmented Lagrangian functional.
         */
        m_ControlWorkVec->fill(0.);
        m_Inequality[index]->partialDerivativeControlState(*m_State, control_, *m_DeltaState, *m_ControlWorkVec);
        m_HessWorkVec->update(alpha, *m_ControlWorkVec, 1.);
        m_ControlWorkVec->fill(0.);
        m_PDE->partialDerivativeControlState(*m_State,
                                             control_,
                                             *m_InequalityDual[index],
                                             *m_DeltaState,
                                             *m_ControlWorkVec);
        m_HessWorkVec->update(alpha, *m_ControlWorkVec, 1.);

        /* Add contribution from \beta_i\left(\frac{\partial h_i}{\partial\mathbf{z}} + \left(\frac{\partial g^{\ast}}
         * {\partial\mathbf{z}}\lambda_i^h\right)\right), where \beta_i = \mu\left(\frac{\partial h_i}{\partial\mathbf{z}}
         * \Delta\mathbf{z} + \frac{\partial h_i}{\partial\mathbf{u}}\Delta\mathbf{u}\right)
         */
        m_GradientWorkVec->fill(0.);
        m_PDE->adjointPartialDerivativeControl(*m_State, control_, *m_InequalityDual[index], *m_GradientWorkVec);
        m_ControlWorkVec->fill(0);
        m_Inequality[index]->partialDerivativeControl(*m_State, control_, *m_ControlWorkVec);
        m_GradientWorkVec->update(1., *m_ControlWorkVec, 1.);
        double partial_derivative_control_dot_vector = m_ControlWorkVec->dot(vector_);
        m_StateWorkVec->fill(0);
        m_Inequality[index]->partialDerivativeState(*m_State, control_, *m_StateWorkVec);
        double partial_derivative_state_dot_delta_state = m_StateWorkVec->dot(*m_DeltaState);
        double beta = one_over_penalty
                * (partial_derivative_control_dot_vector + partial_derivative_state_dot_delta_state);
        m_HessWorkVec->update(beta, *m_GradientWorkVec, 1.);
        hess_times_vector_.update(1., *m_HessWorkVec, 1.);
    }
}

void AugmentedLagrangianTypeNP::computeFullNewtonHessian(const trrom::Vector<double> & control_,
                                                         const trrom::Vector<double> & vector_,
                                                         trrom::Vector<double> & hess_times_vec_)
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
    /* FIRST SOLVE: set right-hand-side Vector (using mStateWorkVec as rhs Vector to recycle
     * member data and optimize implementation) for forward solve needed for Hessian calculation */
    m_StateWorkVec->fill(0.);
    m_PDE->partialDerivativeControl(*m_State, control_, vector_, *m_StateWorkVec);
    m_StateWorkVec->scale(static_cast<double>(-1.0));

    // FIRST SOLVE for \Delta\mathbf{u}
    m_DeltaState->fill(0.);
    m_PDE->applyInverseJacobianState(*m_State, control_, *m_StateWorkVec, *m_DeltaState);
    this->updateInverseJacobianStateCounter();

    // Solve for \Delta\lambda^f, i.e. \Delta\lambda associated with the objective function
    this->computeObjectiveHessianDual(control_, vector_);

    // Apply vector to objective function Hessian operator
    this->computeObjectiveHessianTimesVector(control_, vector_, hess_times_vec_);

    // Solve for \Delta\lambda_i^h, i.e. \Delta\lambda associated with the i-th inequality constraint
    int num_inequality_constraints = m_CurrentInequalityConstraintValues->size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        this->computeInequalityConstraintHessianDual(index, control_, vector_);
    }

    // Apply vector to inequality constraint Hessian operator and add contribution to total Hessian
    this->computeInequalityHessianTimesVector(control_, vector_, hess_times_vec_);
}

void AugmentedLagrangianTypeNP::computeGaussNewtonHessian(const trrom::Vector<double> & control_,
                                                          const trrom::Vector<double> & vector_,
                                                          trrom::Vector<double> & hess_times_vec_)
{
    /*
     * Compute objective function contribution: \frac{\partial^2 f}{\partial\mathbf{z}^2}
     */
    hess_times_vec_.fill(0.);
    m_Objective->partialDerivativeControlControl(*m_State, control_, vector_, hess_times_vec_);
    m_ControlWorkVec->fill(0.);
    m_PDE->partialDerivativeControlControl(*m_State, control_, *m_ObjectiveDual, vector_, *m_ControlWorkVec);
    hess_times_vec_.update(static_cast<double>(1.0), *m_ControlWorkVec, 1.);

    int num_constraints = m_InequalityDual.size();
    double one_over_penalty = static_cast<double>(1.) / m_Penalty;
    for(int index = 0; index < num_constraints; ++ index)
    {
        double alpha = (*m_LagrangeMultipliers)[index]
                + (one_over_penalty * (*m_CurrentInequalityConstraintValues)[index]);
        /*
         * Compute \alpha_i\left(\frac{\partial^2 h_i}{\partial\mathbf{z}^2} + \left(\frac{\partial^2 g^{\ast}}
         * {\partial\amhbf{z}^2}\lambda_i^h\right)\right)\Delta{z}. Here, \Delta{z} denotes the trial direction.
         */
        m_HessWorkVec->fill(0.);
        m_Inequality[index]->partialDerivativeControlControl(*m_State, control_, vector_, *m_HessWorkVec);
        m_HessWorkVec->scale(alpha);
        m_ControlWorkVec->fill(0.);
        m_PDE->partialDerivativeControlControl(*m_State,
                                               control_,
                                               *m_InequalityDual[index],
                                               vector_,
                                               *m_ControlWorkVec);
        m_HessWorkVec->update(alpha, *m_ControlWorkVec, 1.);

        /* Add contribution from \beta_i\left(\frac{\partial h_i}{\partial\mathbf{z}} + \left(\frac{\partial g^{\ast}}
         * {\partial\mathbf{z}}\lambda_i^h\right)\right), where \beta_i = \mu\left(\frac{\partial h_i}{\partial\mathbf{z}}
         * \Delta\mathbf{z} + \frac{\partial h_i}{\partial\mathbf{u}}\Delta\mathbf{u}\right)
         */
        m_GradientWorkVec->fill(0.);
        m_PDE->adjointPartialDerivativeControl(*m_State, control_, *m_InequalityDual[index], *m_GradientWorkVec);
        m_ControlWorkVec->fill(0);
        m_Inequality[index]->partialDerivativeControl(*m_State, control_, *m_ControlWorkVec);
        m_GradientWorkVec->update(static_cast<double>(1.), *m_ControlWorkVec, 1.);
        double partial_derivative_control_dot_vector = m_ControlWorkVec->dot(vector_);
        double beta = one_over_penalty * partial_derivative_control_dot_vector;
        m_HessWorkVec->update(beta, *m_GradientWorkVec, 1.);
        hess_times_vec_.update(static_cast<double>(1.), *m_HessWorkVec, 1.);
    }
}

}
