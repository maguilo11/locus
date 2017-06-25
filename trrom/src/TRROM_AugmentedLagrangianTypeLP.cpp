/*
 * TRROM_AugmentedLagrangianTypeLP.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_ObjectiveTypeLP.hpp"
#include "TRROM_InequalityTypeLP.hpp"
#include "TRROM_AugmentedLagrangianTypeLP.hpp"

namespace trrom
{

AugmentedLagrangianTypeLP::AugmentedLagrangianTypeLP(const std::shared_ptr<trrom::Data> & input_,
                                                     const std::shared_ptr<trrom::ObjectiveTypeLP> & objective_,
                                                     const std::vector<std::shared_ptr<trrom::InequalityTypeLP> > & inequality_) :
        m_HessianCounter(0),
        m_GradientCounter(0),
        m_ObjectiveCounter(0),
        m_InequalityCounter(0),
        m_InequalityGradientCounter(0),
        m_Penalty(1),
        m_MinPenalty(1e-10),
        m_PenaltyScaling(0.2),
        m_NormLagrangianGradient(0.),
        m_NormInequalityConstraints(0.),
        m_ControlWorkVec(input_->control()->create()),
        m_LagrangianGradient(input_->control()->create()),
        m_LagrangeMultipliers(input_->slacks()->create()),
        m_InequalityConstraintValues(input_->slacks()->create()),
        m_CurrentInequalityConstraintValues(input_->slacks()->create()),
        m_Objective(objective_),
        m_Inequality(inequality_.begin(), inequality_.end())
{
}

AugmentedLagrangianTypeLP::~AugmentedLagrangianTypeLP()
{
}

int AugmentedLagrangianTypeLP::getHessianCounter() const
{
    return (m_HessianCounter);
}

void AugmentedLagrangianTypeLP::updateHessianCounter()
{
    m_HessianCounter++;
}

int AugmentedLagrangianTypeLP::getGradientCounter() const
{
    return (m_GradientCounter);
}

void AugmentedLagrangianTypeLP::updateGradientCounter()
{
    m_GradientCounter++;
}

int AugmentedLagrangianTypeLP::getObjectiveCounter() const
{
    return (m_ObjectiveCounter);
}

void AugmentedLagrangianTypeLP::updateObjectiveCounter()
{
    m_ObjectiveCounter++;
}

int AugmentedLagrangianTypeLP::getInequalityCounter() const
{
    return (m_InequalityCounter);
}

void AugmentedLagrangianTypeLP::updateInequalityCounter()
{
    m_InequalityCounter++;
}

int AugmentedLagrangianTypeLP::getInequalityGradientCounter() const
{
    return (m_InequalityGradientCounter);
}

void AugmentedLagrangianTypeLP::updateInequalityGradientCounter()
{
    m_InequalityGradientCounter++;
}

double AugmentedLagrangianTypeLP::getPenalty() const
{
    return (m_Penalty);
}

double AugmentedLagrangianTypeLP::getNormLagrangianGradient() const
{
    return (m_NormLagrangianGradient);
}

double AugmentedLagrangianTypeLP::getNormInequalityConstraints() const
{
    return (m_NormInequalityConstraints);
}

void AugmentedLagrangianTypeLP::updateInequalityConstraintValues()
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

double AugmentedLagrangianTypeLP::objective(const std::shared_ptr<trrom::Vector<double> > & control_,
                                            const double & tolerance_,
                                            bool & inexactness_violated_)
{
    // Evaluate objective function, f(\mathbf{z})
    double objective_value = m_Objective->value(tolerance_, *control_, inexactness_violated_);
    this->updateObjectiveCounter();

    // Evaluate inequality constraints, h(\mathbf{u}(\mathbf{z}),\mathbf{z})
    int num_inequality_constraints = m_InequalityConstraintValues->size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        (*m_InequalityConstraintValues)[index] = m_Inequality[index]->value(*control_) - m_Inequality[index]->bound();
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

    return (augmented_lagrangian_value);
}

void AugmentedLagrangianTypeLP::gradient(const std::shared_ptr<trrom::Vector<double> > & control_,
                                         const std::shared_ptr<trrom::Vector<double> > & gradient_,
                                         const double & tolerance_,
                                         bool & inexactness_violated_)
{
    // Compute objective function gradient: \frac{\partial f}{\partial\mathbf{z}}
    gradient_->fill(0.);
    m_LagrangianGradient->fill(0.);
    m_Objective->gradient(tolerance_, *control_, *m_LagrangianGradient, inexactness_violated_);

    // Compute inequality constraint gradient: \frac{\partial h_i}{\partial\mathbf{z}}
    double one_over_penalty = static_cast<double>(1.) / m_Penalty;
    int num_inequality_constraints = m_CurrentInequalityConstraintValues->size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        // Add contribution from: \lambda_i\frac{\partial h_i}{\partial\mathbf{z}} to Lagrangian gradient
        m_ControlWorkVec->fill(0.);
        m_Inequality[index]->gradient(*control_, *m_ControlWorkVec);
        this->updateInequalityGradientCounter();
        m_LagrangianGradient->update((*m_LagrangeMultipliers)[index], *m_ControlWorkVec, 1.);

        // Add contribution from \mu*h_i(\mathbf{u}(\mathbf{z}),\mathbf{z})\frac{\partial h_i}{\partial\mathbf{z}}.
        double alpha = one_over_penalty * (*m_CurrentInequalityConstraintValues)[index];
        gradient_->update(alpha, *m_ControlWorkVec, 1.);
    }
    m_NormLagrangianGradient = m_LagrangianGradient->norm();
    // Compute augmented Lagrangian gradient
    gradient_->update(1., *m_LagrangianGradient, 1.);
    this->updateGradientCounter();
}

void AugmentedLagrangianTypeLP::hessian(const std::shared_ptr<trrom::Vector<double> > & control_,
                                        const std::shared_ptr<trrom::Vector<double> > & vector_,
                                        const std::shared_ptr<trrom::Vector<double> > & hessian_times_vec_,
                                        const double & tolerance_,
                                        bool & inexactness_violated_)
{
    /// Reduced space interface: Assemble the reduced space gradient operator. \n
    /// In: \n
    ///     control_ = control_ Vector, i.e. optimization parameters, unchanged on exist. \n
    ///        (std::Vector < double >) \n
    ///     vector_ = trial direction, i.e. perturbation vector, unchanged on exist. \n
    ///        (std::Vector < double >) \n
    /// Out: \n
    ///     hessian_times_vec_ = application of the trial step to the Hessian operator. \n
    ///        (std::Vector < double >) \n
    ///
    // Apply vector to objective function Hessian operator
    hessian_times_vec_->fill(0.);
    m_Objective->hessian(tolerance_, *control_, *vector_, *hessian_times_vec_, inexactness_violated_);

    // Apply vector to inequality constraint Hessian operator and add contribution to total Hessian
    double one_over_penalty = static_cast<double>(1.) / m_Penalty;
    int num_inequality_constraints = m_Inequality.size();
    for(int index = 0; index < num_inequality_constraints; ++ index)
    {
        // Add contribution from: \lambda_i\frac{\partial^2 h_i}{\partial\mathbf{z}^2}
        m_ControlWorkVec->fill(0.);
        m_Inequality[index]->hessian(*control_, *vector_, *m_ControlWorkVec);
        hessian_times_vec_->update((*m_LagrangeMultipliers)[index], *m_ControlWorkVec, 1.);

        // Add contribution from: \mu\frac{\partial^2 h_i}{\partial\mathbf{z}^2}\h_i(\mathbf{z})
        double alpha = one_over_penalty * (*m_CurrentInequalityConstraintValues)[index];
        hessian_times_vec_->update(alpha, *m_ControlWorkVec, 1.);

        // Compute Jacobian, i.e. \frac{\partial h_i}{\partial\mathbf{z}}
        m_ControlWorkVec->fill(0.);
        m_Inequality[index]->gradient(*control_, *m_ControlWorkVec);
        double jacobian_dot_trial_direction = m_ControlWorkVec->dot(*vector_);
        double beta = one_over_penalty * jacobian_dot_trial_direction;
        // Add contribution from: \mu\left(\frac{\partial h_i}{\partial\mathbf{z}}^{T}
        //                        \frac{\partial h_i}{\partial\mathbf{z}}\right)
        hessian_times_vec_->update(beta, *m_ControlWorkVec, 1.);
    }

    this->updateHessianCounter();
}

bool AugmentedLagrangianTypeLP::updateLagrangeMultipliers()
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

void AugmentedLagrangianTypeLP::setMinPenalty(double input_)
{
    m_MinPenalty = input_;
}

void AugmentedLagrangianTypeLP::setPenaltyScaling(double input_)
{
    m_PenaltyScaling = input_;
}

}
