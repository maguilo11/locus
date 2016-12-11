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
    explicit DOTk_DualObjectiveFunctionMMA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    virtual ~DOTk_DualObjectiveFunctionMMA();

    Real value(const dotk::Vector<Real> & dual_);
    void gradient(const dotk::Vector<Real> & dual_, dotk::Vector<Real> & gradient_);

    void setEpsilon(Real epsilon_);

    void setCurrentObjectiveFunctionValue(Real value_);
    void setTrialControl(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_);
    void setCurrentInequalityConstraintResiduals(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_);

    void gatherTrialControl(const std::tr1::shared_ptr<dotk::Vector<Real> > & input_);
    void gatherMovingAsymptotes(const std::tr1::shared_ptr<dotk::Vector<Real> > & lower_asymptote_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & upper_asymptote_);
    void gatherTrialControlBounds(const std::tr1::shared_ptr<dotk::Vector<Real> > & lower_bound_,
                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & upper_bound_);
    void gatherObjectiveCoefficients(const std::tr1::shared_ptr<dotk::Vector<Real> > & p_coefficients_,
                                     const std::tr1::shared_ptr<dotk::Vector<Real> > & q_coefficients_,
                                     Real & r_coefficients_);
    void gatherInequalityCoefficients(const std::tr1::shared_ptr<dotk::matrix<Real> > & p_coefficients_,
                                      const std::tr1::shared_ptr<dotk::matrix<Real> > & q_coefficients_,
                                      const std::tr1::shared_ptr<dotk::Vector<Real> > & r_coefficients_);

    void updateMovingAsymptotes(const std::tr1::shared_ptr<dotk::Vector<Real> > & current_control_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_);
    void updateTrialControlBounds(const Real & scale_,
                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & current_control_,
                                  const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_);
    void updateObjectiveCoefficientVectors(const Real & globalization_scaling_,
                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_,
                                           const std::tr1::shared_ptr<dotk::Vector<Real> > & current_gradient_);
    void updateInequalityCoefficientVectors(const std::tr1::shared_ptr<dotk::Vector<Real> > & globalization_scaling_,
                                            const std::tr1::shared_ptr<dotk::Vector<Real> > & current_sigma_,
                                            const std::tr1::shared_ptr<dotk::matrix<Real> > & current_gradients_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    void updateTrialControl(const dotk::Vector<Real> & dual_);
    void updateTrialAuxiliaryVariables(const dotk::Vector<Real> & dual_);
    Real computeInequalityConstraintContribution(const dotk::Vector<Real> & dual_);

private:
    Real m_Epsilon;
    Real m_ObjectiveCoefficientA;
    Real m_ObjectiveCoefficientR;
    Real m_TrialAuxiliaryVariableZ;
    Real m_CurrentObjectiveFunctionValue;

    std::tr1::shared_ptr<dotk::Vector<Real> > m_TermA;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TermB;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialControl;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_LowerAsymptote;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_UpperAsymptote;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DualDotPcoeffMatrix;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DualDotQcoeffMatrix;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialControlLowerBound;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialControlUpperBound;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ObjectiveCoefficientsP;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ObjectiveCoefficientsQ;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsA;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsC;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsD;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_InequalityCoefficientsP;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_InequalityCoefficientsQ;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_InequalityCoefficientsR;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialAuxiliaryVariableY;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_CurrentInequalityConstraintResiduals;

private:
    DOTk_DualObjectiveFunctionMMA(const dotk::DOTk_DualObjectiveFunctionMMA &);
    dotk::DOTk_DualObjectiveFunctionMMA & operator=(const dotk::DOTk_DualObjectiveFunctionMMA &);
};

}

#endif /* DOTK_DUALOBJECTIVEFUNCTIONMMA_HPP_ */
