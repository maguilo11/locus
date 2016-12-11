/*
 * DOTk_TrustRegionMngTypeELP.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONMNGTYPEELP_HPP_
#define DOTK_TRUSTREGIONMNGTYPEELP_HPP_

#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_TrustRegion;
class DOTk_AssemblyManager;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;

class DOTk_TrustRegionMngTypeELP : public dotk::DOTk_OptimizationDataMng
{
public:
    DOTk_TrustRegionMngTypeELP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                               const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_TrustRegionMngTypeELP();

    virtual void setNewObjectiveFunctionValue(Real value_);
    virtual void setOldObjectiveFunctionValue(Real value_);
    virtual Real getNewObjectiveFunctionValue() const;
    virtual Real getOldObjectiveFunctionValue() const;

    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getOldDual() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getNewDual() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getMatrixTimesVector() const;

    virtual void setTrialStep(const dotk::Vector<Real> & input_);
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getTrialStep() const;

    virtual void setNewPrimal(const dotk::Vector<Real> & input_);
    virtual void setOldPrimal(const dotk::Vector<Real> & input_);
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getNewPrimal() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getOldPrimal() const;

    virtual void setNewGradient(const dotk::Vector<Real> & input_);
    virtual void setOldGradient(const dotk::Vector<Real> & input_);
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getNewGradient() const;
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getOldGradient() const;

    virtual const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & getRoutinesMng() const;

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaDual() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getLinearizedEqConstraint() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getJacobianTimesTangentialStep() const;

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getOldEqualityConstraintResidual() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getNewEqualityConstraintResidual() const;

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getNormalStep() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDeltaPrimal() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getTangentialStep() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getNormalCauchyStep() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getProjectedGradient() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getHessTimesNormalStep() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getTangentialStepResidual() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getProjTangentialCauchyStep() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getAugmentedSystemLeftHandSide() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getAugmentedSystemRightHandSide() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getProjectedTangentialStep() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getTangentialSubProblemRhs() const;

    size_t getObjectiveFunctionEvaluationCounter() const;

    void setTrustRegionRadius(Real radius_);
    Real getTrustRegionRadius() const;
    void setMinTrustRegionRadius(Real radius_);
    Real getMinTrustRegionRadius() const;
    void setMaxTrustRegionRadius(Real radius_);
    Real getMaxTrustRegionRadius() const;
    void setTrustRegionExpansionParameter(Real parameter_);
    Real getTrustRegionExpansionParameter() const;
    void setTrustRegionContractionParameter(Real parameter_);
    Real getTrustRegionContractionParameter() const;
    void setMinActualOverPredictedReductionAllowed(Real parameter_);
    Real getMinActualOverPredictedReductionAllowed() const;

    Real computeDoglegRoot(const Real & trust_region_radius_,
                           const std::tr1::shared_ptr<dotk::Vector<Real> > & x_,
                           const std::tr1::shared_ptr<dotk::Vector<Real> > & y_);
    void invalidCurvatureDetected(bool invalid_curvature_detected_);

    const std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & getTrustRegion() const;

public:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaDual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NormalStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TangentialStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NormalCauchyStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ProjectedGradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_HessTimesNormalStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewEqualityConstraint;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldEqualityConstraint;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_LinearizedEqConstraint;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TangentialStepResidual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ProjectedTangentialStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TangentialSubProblemRhs;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_JacobianTimesTangentialStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_AugmentedSystemLeftHandSide;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_AugmentedSystemRightHandSide;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ProjectedTangentialCauchyStep;

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                    const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                    const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);

private:
    Real m_OldObjectiveFunction;
    Real m_NewObjectiveFunction;

    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldDual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewDual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldPrimal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewPrimal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldGradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewGradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_MatrixTimesVector;

    std::tr1::shared_ptr<dotk::DOTk_TrustRegion> m_TrustRegion;
    std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> m_RoutinesMng;

private:
    // unimplemented
    DOTk_TrustRegionMngTypeELP(const dotk::DOTk_TrustRegionMngTypeELP &);
    dotk::DOTk_TrustRegionMngTypeELP operator=(const dotk::DOTk_TrustRegionMngTypeELP &);
};

}

#endif /* DOTK_TRUSTREGIONMNGTYPEELP_HPP_ */
