/*
 * DOTk_DualObjectiveFunctionMMA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <cassert>
#include <algorithm>

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_DualObjectiveFunctionMMA.hpp"

namespace dotk
{

DOTk_DualObjectiveFunctionMMA::DOTk_DualObjectiveFunctionMMA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_) :
        dotk::DOTk_ObjectiveFunction<Real>(),
        m_Epsilon(1e-6),
        m_ObjectiveCoefficientA(data_mng_->m_ObjectiveCoefficientsA),
        m_ObjectiveCoefficientR(0),
        m_TrialAuxiliaryVariableZ(data_mng_->m_InitialAuxiliaryVariableZ),
        m_CurrentObjectiveFunctionValue(0),
        m_TermA(data_mng_->m_CurrentControl->clone()),
        m_TermB(data_mng_->m_CurrentControl->clone()),
        m_TrialControl(data_mng_->m_CurrentControl->clone()),
        m_LowerAsymptote(data_mng_->m_CurrentControl->clone()),
        m_UpperAsymptote(data_mng_->m_CurrentControl->clone()),
        m_DualDotPcoeffMatrix(data_mng_->m_CurrentControl->clone()),
        m_DualDotQcoeffMatrix(data_mng_->m_CurrentControl->clone()),
        m_TrialControlLowerBound(data_mng_->m_CurrentControl->clone()),
        m_TrialControlUpperBound(data_mng_->m_CurrentControl->clone()),
        m_ObjectiveCoefficientsP(data_mng_->m_CurrentControl->clone()),
        m_ObjectiveCoefficientsQ(data_mng_->m_CurrentControl->clone()),
        m_InequalityCoefficientsA(data_mng_->m_Dual->clone()),
        m_InequalityCoefficientsC(data_mng_->m_Dual->clone()),
        m_InequalityCoefficientsD(data_mng_->m_Dual->clone()),
        m_InequalityCoefficientsP(data_mng_->m_CurrentInequalityGradients->clone()),
        m_InequalityCoefficientsQ(data_mng_->m_CurrentInequalityGradients->clone()),
        m_InequalityCoefficientsR(data_mng_->m_Dual->clone()),
        m_TrialAuxiliaryVariableY(data_mng_->m_Dual->clone()),
        m_CurrentInequalityConstraintResiduals(data_mng_->m_Dual->clone())
{
    this->initialize(data_mng_);
}

DOTk_DualObjectiveFunctionMMA::~DOTk_DualObjectiveFunctionMMA()
{
}

Real DOTk_DualObjectiveFunctionMMA::value(const dotk::Vector<Real> & dual_)
{
    // Evaluate dual problem objective function
    this->updateTrialControl(dual_);
    this->updateTrialAuxiliaryVariables(dual_);

    Real objective_term = m_ObjectiveCoefficientR + (m_TrialAuxiliaryVariableZ * m_ObjectiveCoefficientA)
            + (m_Epsilon * m_TrialAuxiliaryVariableZ * m_TrialAuxiliaryVariableZ);

    Real inequality_summation_term = this->computeInequalityConstraintContribution(dual_);

    // Moving Asymptotes Term
    Real p_coefficients_term = 0;
    Real q_coefficients_term = 0;
    size_t number_controls = m_TrialControl->size();
    for(size_t index = 0; index < number_controls; ++index)
    {
        Real numerator = (*m_ObjectiveCoefficientsP)[index] + (*m_DualDotPcoeffMatrix)[index];
        Real denominator = (*m_UpperAsymptote)[index] - (*m_TrialControl)[index];
        p_coefficients_term += numerator / denominator;

        numerator = (*m_ObjectiveCoefficientsQ)[index] + (*m_DualDotQcoeffMatrix)[index];
        denominator = (*m_TrialControl)[index] - (*m_LowerAsymptote)[index];
        q_coefficients_term += numerator / denominator;
    }
    Real moving_asymptotes_term = p_coefficients_term + q_coefficients_term;

    // Add all contributions to dual objective function
    Real output = -1. * (objective_term + inequality_summation_term + moving_asymptotes_term);

    return (output);
}

void DOTk_DualObjectiveFunctionMMA::gradient(const dotk::Vector<Real> & dual_, dotk::Vector<Real> & gradient_)
{
    size_t number_inequalities = dual_.size();
    size_t number_controls = m_TrialControl->size();
    for(size_t index_i = 0; index_i < number_inequalities; ++index_i)
    {

        gradient_[index_i] = (*m_InequalityCoefficientsR)[index_i] - (*m_TrialAuxiliaryVariableY)[index_i]
                - ((*m_InequalityCoefficientsA)[index_i] * m_TrialAuxiliaryVariableZ);

        Real p_coefficients_sum = 0;
        Real q_coefficients_sum = 0;

        for(size_t index_j = 0; index_j < number_controls; ++index_j)
        {
            p_coefficients_sum += (*m_InequalityCoefficientsP->basis(index_i))[index_j]
                    / ((*m_UpperAsymptote)[index_j] - (*m_TrialControl)[index_j]);

            q_coefficients_sum += (*m_InequalityCoefficientsQ->basis(index_i))[index_j]
                    / ((*m_TrialControl)[index_j] - (*m_LowerAsymptote)[index_j]);
        }
        // Add contribution to dual gradient
        gradient_[index_i] = -(gradient_[index_i] + p_coefficients_sum + q_coefficients_sum);

    }
}

void DOTk_DualObjectiveFunctionMMA::setEpsilon(Real epsilon_)
{
    m_Epsilon = epsilon_;
}

void DOTk_DualObjectiveFunctionMMA::setCurrentObjectiveFunctionValue(Real value_)
{
    m_CurrentObjectiveFunctionValue = value_;
}

void DOTk_DualObjectiveFunctionMMA::setTrialControl(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_)
{
    assert(input_->size() == m_TrialControl->size());
    m_TrialControl->update(1., *input_, 0.);
}

void DOTk_DualObjectiveFunctionMMA::setCurrentInequalityConstraintResiduals
(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_)
{
    assert(input_->size() == m_CurrentInequalityConstraintResiduals->size());
    m_CurrentInequalityConstraintResiduals->update(1., *input_, 0.);
}

void DOTk_DualObjectiveFunctionMMA::gatherTrialControl(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_)
{
    assert(input_->size() == m_TrialControl->size());
    input_->update(1., *m_TrialControl, 0.);
}

void DOTk_DualObjectiveFunctionMMA::gatherMovingAsymptotes(const std::tr1::shared_ptr<dotk::Vector<Real> > & lower_asymptote_,
                                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & upper_asymptote_)
{
    assert(lower_asymptote_->size() == m_LowerAsymptote->size());
    assert(upper_asymptote_->size() == m_UpperAsymptote->size());
    lower_asymptote_->update(1., *m_LowerAsymptote, 0.);
    upper_asymptote_->update(1., *m_UpperAsymptote, 0.);
}

void DOTk_DualObjectiveFunctionMMA::gatherTrialControlBounds(const std::tr1::shared_ptr<dotk::Vector<Real> > & lower_bound_,
                                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & upper_bound_)
{
    assert(lower_bound_->size() == m_TrialControlLowerBound->size());
    assert(upper_bound_->size() == m_TrialControlUpperBound->size());
    lower_bound_->update(1., *m_TrialControlLowerBound, 0.);
    upper_bound_->update(1., *m_TrialControlUpperBound, 0.);
}

void DOTk_DualObjectiveFunctionMMA::gatherObjectiveCoefficients(const std::tr1::shared_ptr<dotk::Vector<Real> > & p_coefficients_,
                                                                const std::tr1::shared_ptr<dotk::Vector<Real> > & q_coefficients_,
                                                                Real & r_coefficients_)
{
    assert(p_coefficients_->size() == m_ObjectiveCoefficientsP->size());
    assert(q_coefficients_->size() == m_ObjectiveCoefficientsQ->size());
    p_coefficients_->update(1., *m_ObjectiveCoefficientsP, 0.);
    q_coefficients_->update(1., *m_ObjectiveCoefficientsQ, 0.);
    r_coefficients_ = m_ObjectiveCoefficientR;
}

void DOTk_DualObjectiveFunctionMMA::gatherInequalityCoefficients(const std::tr1::shared_ptr<dotk::matrix<Real> > & p_coefficients_,
                                                                 const std::tr1::shared_ptr<dotk::matrix<Real> > & q_coefficients_,
                                                                 const std::tr1::shared_ptr<dotk::Vector<Real> > & r_coefficients_)
{
    assert(p_coefficients_->size() == m_InequalityCoefficientsP->size());
    assert(q_coefficients_->size() == m_InequalityCoefficientsQ->size());
    assert(r_coefficients_->size() == m_InequalityCoefficientsR->size());
    p_coefficients_->copy(*m_InequalityCoefficientsP);
    q_coefficients_->copy(*m_InequalityCoefficientsQ);
    r_coefficients_->update(1., *m_InequalityCoefficientsR, 0.);
}

void DOTk_DualObjectiveFunctionMMA::updateMovingAsymptotes(const std::tr1::shared_ptr<dotk::Vector<Real> > & current_control_,
                                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_)
{
    m_LowerAsymptote->update(1., *current_control_, 0.);
    m_LowerAsymptote->update(-1., *current_sigma_, 1.);

    m_UpperAsymptote->update(1., *current_control_, 0.);
    m_UpperAsymptote->update(1., *current_sigma_, 1.);
}

void DOTk_DualObjectiveFunctionMMA::updateTrialControlBounds(const Real & scale_,
                                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & current_control_,
                                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_)
{
    m_TrialControlLowerBound->update(1., *current_control_, 0.);
    m_TrialControlLowerBound->update(-scale_, *current_sigma_, 1.);

    m_TrialControlUpperBound->update(1., *current_control_, 0.);
    m_TrialControlUpperBound->update(scale_, *current_sigma_, 1.);
}

void DOTk_DualObjectiveFunctionMMA::updateObjectiveCoefficientVectors(const Real & globalization_scaling_,
                                                                      const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_,
                                                                      const std::tr1::shared_ptr<dotk::Vector<Real> > & current_gradient_)
{
    size_t number_primals = current_sigma_->size();
    m_ObjectiveCoefficientR = m_CurrentObjectiveFunctionValue;
    for(size_t index = 0; index < number_primals; ++index)
    {
        Real value = (*current_sigma_)[index] * (*current_sigma_)[index];
        (*m_ObjectiveCoefficientsP)[index] = value * std::max(0., (*current_gradient_)[index])
                + ((globalization_scaling_ * (*current_sigma_)[index]) / static_cast<Real>(4));

        (*m_ObjectiveCoefficientsQ)[index] = value * std::max(0., -(*current_gradient_)[index])
                + ((globalization_scaling_ * (*current_sigma_)[index]) / static_cast<Real>(4));

        m_ObjectiveCoefficientR -= ((*m_ObjectiveCoefficientsP)[index] + (*m_ObjectiveCoefficientsQ)[index])
                / (*current_sigma_)[index];
    }
}

void DOTk_DualObjectiveFunctionMMA::updateInequalityCoefficientVectors(const std::tr1::shared_ptr<dotk::Vector<Real> > & globalization_scaling_,
                                                                       const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_,
                                                                       const std::tr1::shared_ptr<dotk::matrix<Real> > & current_gradients_)
{
    size_t number_primals = current_sigma_->size();
    size_t number_inequalitites = globalization_scaling_->size();
    for(size_t index_i = 0; index_i < number_inequalitites; ++index_i)
    {
        (*m_InequalityCoefficientsR)[index_i] = (*m_CurrentInequalityConstraintResiduals)[index_i];

        for(size_t index_j = 0; index_j < number_primals; ++index_j)
        {
            Real value = (*current_sigma_)[index_j] * (*current_sigma_)[index_j];
            (*m_InequalityCoefficientsP->basis(index_i))[index_j] = value
                    * std::max(0., (*current_gradients_->basis(index_i))[index_j])
                    + (((*globalization_scaling_)[index_i] * (*current_sigma_)[index_j]) / static_cast<Real>(4));

            (*m_InequalityCoefficientsQ->basis(index_i))[index_j] = value
                    * std::max(0., -(*current_gradients_->basis(index_i))[index_j])
                    + (((*globalization_scaling_)[index_i] * (*current_sigma_)[index_j]) / static_cast<Real>(4));

            (*m_InequalityCoefficientsR)[index_i] -= ((*m_InequalityCoefficientsP->basis(index_i))[index_j]
                    + (*m_InequalityCoefficientsQ->basis(index_i))[index_j]) / (*current_sigma_)[index_j];
        }
    }
}

void DOTk_DualObjectiveFunctionMMA::initialize(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    m_InequalityCoefficientsA->update(1., *data_mng_->m_InputInequalityCoefficientsA, 0.);
    m_InequalityCoefficientsC->update(1., *data_mng_->m_InputInequalityCoefficientsC, 0.);
    m_InequalityCoefficientsD->update(1., *data_mng_->m_InputInequalityCoefficientsD, 0.);
}

void DOTk_DualObjectiveFunctionMMA::updateTrialControl(const dotk::Vector<Real> & dual_)
{
    // Update primal variables based on the following expression:
    // \[ x(\lambda)=\frac{u_j^k\mathtt{b}^{1/2}+l_j^k\mathtt{a}^{1/2}}{(\mathtt{a}^{1/2}+\mathtt{b}^{1/2})} \],
    // where
    //      \[ \mathtt{a}=(p_{0j}+\lambda^{\intercal}p_j) \] and [ \mathtt{b}=(q_{0j}+\lambda^{\intercal}q_j) ]
    m_InequalityCoefficientsP->matVec(dual_, *m_DualDotPcoeffMatrix, true);
    m_InequalityCoefficientsQ->matVec(dual_, *m_DualDotQcoeffMatrix, true);
    m_TermA->update(1., *m_ObjectiveCoefficientsP, 0.);
    m_TermA->update(1., *m_DualDotPcoeffMatrix, 1.);
    m_TermB->update(1., *m_ObjectiveCoefficientsQ, 0.);
    m_TermB->update(1., *m_DualDotQcoeffMatrix, 1.);

    size_t number_primals = m_DualDotPcoeffMatrix->size();
    for(size_t index = 0; index < number_primals; ++index)
    {
        Real sqrt_term_a = std::sqrt((*m_TermA)[index]);
        Real sqrt_term_b = std::sqrt((*m_TermB)[index]);
        Real numerator = ((*m_LowerAsymptote)[index] * sqrt_term_a) + ((*m_UpperAsymptote)[index] * sqrt_term_b);
        Real denominator = (sqrt_term_a + sqrt_term_b);
        (*m_TrialControl)[index] = numerator / denominator;
        // Project trial control to feasible set
        (*m_TrialControl)[index] = std::max((*m_TrialControl)[index], (*m_TrialControlLowerBound)[index]);
        (*m_TrialControl)[index] = std::min((*m_TrialControl)[index], (*m_TrialControlUpperBound)[index]);
    }
}

void DOTk_DualObjectiveFunctionMMA::updateTrialAuxiliaryVariables(const dotk::Vector<Real> & dual_)
{
    // Update auxiliary variables based on the following expression:
    // \[ y_i(\lambda)=\frac{\lambda_i-c_i}{2d_i} \]
    // and
    // \[ z(\lambda)=\frac{\lambda^{\intercal}a-a_0}{2\varepsilon} \]
    size_t number_inequalities = m_TrialAuxiliaryVariableY->size();
    for(size_t index = 0; index < number_inequalities; ++index)
    {

        (*m_TrialAuxiliaryVariableY)[index] = (dual_[index] - (*m_InequalityCoefficientsC)[index])
                / (*m_InequalityCoefficientsD)[index];
        // Project auxiliary variables Y to feasible set (Y >= 0)
        (*m_TrialAuxiliaryVariableY)[index] = std::max((*m_TrialAuxiliaryVariableY)[index], 0.);
    }
    Real dual_dot_inequality_coeff_a = m_InequalityCoefficientsA->dot(dual_);
    m_TrialAuxiliaryVariableZ = (dual_dot_inequality_coeff_a - m_ObjectiveCoefficientA)
            / (static_cast<Real>(2.) * m_Epsilon);
    // Project auxiliary variables Z to feasible set (Z >= 0)
    m_TrialAuxiliaryVariableZ = std::max(m_TrialAuxiliaryVariableZ, 0.);
}

Real DOTk_DualObjectiveFunctionMMA::computeInequalityConstraintContribution(const dotk::Vector<Real> & dual_)
{
    // Compute the following calculation:
    // \sum_{i=1}^{m}\left( c_iy_i + \frac{1}{2}d_iy_i^2 \right) - \lambda^{T}y - (\lambda^{T}a)z + \lambda^{T}r,
    // where m denotes the number of inequality constraints
    Real inequality_summation_term = 0.;
    size_t number_inequalities = dual_.size();
    for(size_t index = 0; index < number_inequalities; ++index)
    {
        inequality_summation_term += ((*m_InequalityCoefficientsC)[index] * (*m_TrialAuxiliaryVariableY)[index])
                + ((*m_InequalityCoefficientsD)[index] * (*m_TrialAuxiliaryVariableY)[index]
                        * (*m_TrialAuxiliaryVariableY)[index]);
    }
    // Add additional contributions to inequality summation term
    Real inequality_coeff_r_dot_dual = m_InequalityCoefficientsR->dot(dual_);
    Real inequality_coeff_a_dot_dual = m_InequalityCoefficientsA->dot(dual_);
    Real trial_auxiliary_variable_y_dot_dual = m_TrialAuxiliaryVariableY->dot(dual_);

    inequality_summation_term = inequality_summation_term - trial_auxiliary_variable_y_dot_dual
            - (inequality_coeff_a_dot_dual * m_TrialAuxiliaryVariableZ) + inequality_coeff_r_dot_dual;

    return (inequality_summation_term);
}

}
