/*
 * DOTk_DualObjectiveFunctionMMA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DUALOBJECTIVEFUNCTIONMMA_HPP_
#define DOTK_DUALOBJECTIVEFUNCTIONMMA_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;

template<typename ScalarType>
class Vector;
template<typename Type>
class matrix;

class DOTk_DualObjectiveFunctionMMA : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    explicit DOTk_DualObjectiveFunctionMMA(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    virtual ~DOTk_DualObjectiveFunctionMMA();

    Real value(const dotk::Vector<Real> & dual_);
    void gradient(const dotk::Vector<Real> & dual_, dotk::Vector<Real> & gradient_);

    void setEpsilon(Real epsilon_);

    void setCurrentObjectiveFunctionValue(Real value_);
    void setTrialControl(const std::shared_ptr<dotk::Vector<Real> > & input_);
    void setCurrentInequalityConstraintResiduals(const std::shared_ptr<dotk::Vector<Real> > & input_);

    void gatherTrialControl(const std::shared_ptr<dotk::Vector<Real> > & input_);
    void gatherMovingAsymptotes(const std::shared_ptr<dotk::Vector<Real> > & lower_asymptote_,
                                const std::shared_ptr<dotk::Vector<Real> > & upper_asymptote_);
    void gatherTrialControlBounds(const std::shared_ptr<dotk::Vector<Real> > & lower_bound_,
                                  const std::shared_ptr<dotk::Vector<Real> > & upper_bound_);
    void gatherObjectiveCoefficients(const std::shared_ptr<dotk::Vector<Real> > & p_coefficients_,
                                     const std::shared_ptr<dotk::Vector<Real> > & q_coefficients_,
                                     Real & r_coefficients_);
    void gatherInequalityCoefficients(const std::shared_ptr<dotk::matrix<Real> > & p_coefficients_,
                                      const std::shared_ptr<dotk::matrix<Real> > & q_coefficients_,
                                      const std::shared_ptr<dotk::Vector<Real> > & r_coefficients_);

    void updateMovingAsymptotes(const std::shared_ptr<dotk::Vector<Real> > & current_control_,
                                const std::shared_ptr<dotk::Vector<Real> > & current_sigma_);
    void updateTrialControlBounds(const Real & scale_,
                                  const std::shared_ptr<dotk::Vector<Real> > & current_control_,
                                  const std::shared_ptr<dotk::Vector<Real> > & current_sigma_);
    void updateObjectiveCoefficientVectors(const Real & globalization_scaling_,
                                           const std::shared_ptr<dotk::Vector<Real> > & current_sigma_,
                                           const std::shared_ptr<dotk::Vector<Real> > & current_gradient_);
    void updateInequalityCoefficientVectors(const std::shared_ptr<dotk::Vector<Real> > & globalization_scaling_,
                                            const std::shared_ptr<dotk::Vector<Real> > & current_sigma_,
                                            const std::shared_ptr<dotk::matrix<Real> > & current_gradients_);

private:
    void initialize(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    void updateTrialControl(const dotk::Vector<Real> & dual_);
    void updateTrialAuxiliaryVariables(const dotk::Vector<Real> & dual_);
    Real computeInequalityConstraintContribution(const dotk::Vector<Real> & dual_);

private:
    Real m_Epsilon;
    Real m_ObjectiveCoefficientA;
    Real m_ObjectiveCoefficientR;
    Real m_TrialAuxiliaryVariableZ;
    Real m_CurrentObjectiveFunctionValue;

    std::shared_ptr<dotk::Vector<Real> > m_TermA;
    std::shared_ptr<dotk::Vector<Real> > m_TermB;
    std::shared_ptr<dotk::Vector<Real> > m_TrialControl;
    std::shared_ptr<dotk::Vector<Real> > m_LowerAsymptote;
    std::shared_ptr<dotk::Vector<Real> > m_UpperAsymptote;
    std::shared_ptr<dotk::Vector<Real> > m_DualDotPcoeffMatrix;
    std::shared_ptr<dotk::Vector<Real> > m_DualDotQcoeffMatrix;
    std::shared_ptr<dotk::Vector<Real> > m_TrialControlLowerBound;
    std::shared_ptr<dotk::Vector<Real> > m_TrialControlUpperBound;
    std::shared_ptr<dotk::Vector<Real> > m_ObjectiveCoefficientsP;
    std::shared_ptr<dotk::Vector<Real> > m_ObjectiveCoefficientsQ;
    std::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsA;
    std::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsC;
    std::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsD;
    std::shared_ptr<dotk::matrix<Real> > m_InequalityCoefficientsP;
    std::shared_ptr<dotk::matrix<Real> > m_InequalityCoefficientsQ;
    std::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsR;
    std::shared_ptr<dotk::Vector<Real> > m_TrialAuxiliaryVariableY;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentInequalityConstraintResiduals;

private:
    DOTk_DualObjectiveFunctionMMA(const dotk::DOTk_DualObjectiveFunctionMMA &);
    dotk::DOTk_DualObjectiveFunctionMMA & operator=(const dotk::DOTk_DualObjectiveFunctionMMA &);
};

}

#endif /* DOTK_DUALOBJECTIVEFUNCTIONMMA_HPP_ */
