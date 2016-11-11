/*
 * DOTk_TrustRegionMngTypeELP.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_PrimalVector.hpp"
#include "DOTk_PrimalVector.cpp"
#include "DOTk_RoutinesTypeELP.hpp"
#include "DOTk_DoglegTrustRegion.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"

namespace dotk
{

DOTk_TrustRegionMngTypeELP::DOTk_TrustRegionMngTypeELP
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
 const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_OptimizationDataMng(),
        m_OldObjectiveFunction(0),
        m_NewObjectiveFunction(0),
        m_OldDual(primal_->dual()->clone()),
        m_NewDual(primal_->dual()->clone()),
        m_TrialStep(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_OldPrimal(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_NewPrimal(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_OldGradient(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_NewGradient(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_MatrixTimesVector(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_TrustRegion(new dotk::DOTk_DoglegTrustRegion),
        m_RoutinesMng(),
        m_DeltaDual(primal_->dual()->clone()),
        m_NormalStep(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_DeltaPrimal(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_TangentialStep(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_NormalCauchyStep(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_ProjectedGradient(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_HessTimesNormalStep(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_NewEqualityConstraint(primal_->dual()->clone()),
        m_OldEqualityConstraint(primal_->dual()->clone()),
        m_LinearizedEqConstraint(primal_->dual()->clone()),
        m_TangentialStepResidual(new dotk::DOTk_MultiVector<Real>(*primal_)),
        m_ProjectedTangentialStep(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_TangentialSubProblemRhs(new dotk::DOTk_PrimalVector<Real>(*primal_)),
        m_JacobianTimesTangentialStep(primal_->dual()->clone()),
        m_AugmentedSystemLeftHandSide(new dotk::DOTk_MultiVector<Real>(*primal_)),
        m_AugmentedSystemRightHandSide(new dotk::DOTk_MultiVector<Real>(*primal_)),
        m_ProjectedTangentialCauchyStep(new dotk::DOTk_PrimalVector<Real>(*primal_))
{
    this->initialize(primal_, objective_, equality_);
}

DOTk_TrustRegionMngTypeELP::~DOTk_TrustRegionMngTypeELP()
{
}

void DOTk_TrustRegionMngTypeELP::setNewObjectiveFunctionValue(Real value_)
{
    m_NewObjectiveFunction = value_;
}

Real DOTk_TrustRegionMngTypeELP::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunction);
}

void DOTk_TrustRegionMngTypeELP::setOldObjectiveFunctionValue(Real value_)
{
    m_OldObjectiveFunction = value_;
}

Real DOTk_TrustRegionMngTypeELP::getOldObjectiveFunctionValue() const
{
    return (m_OldObjectiveFunction);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getOldDual() const
{
    return (m_OldDual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getNewDual() const
{
    return (m_NewDual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getMatrixTimesVector() const
{
    return (m_MatrixTimesVector);
}

void DOTk_TrustRegionMngTypeELP::setTrialStep(const dotk::vector<Real> & input_)
{
    m_TrialStep->copy(input_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getTrialStep() const
{
    return (m_TrialStep);
}

void DOTk_TrustRegionMngTypeELP::setNewPrimal(const dotk::vector<Real> & input_)
{
    m_NewPrimal->copy(input_);
}

void DOTk_TrustRegionMngTypeELP::setOldPrimal(const dotk::vector<Real> & input_)
{
    m_OldPrimal->copy(input_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getNewPrimal() const
{
    return (m_NewPrimal);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getOldPrimal() const
{
    return (m_OldPrimal);
}

void DOTk_TrustRegionMngTypeELP::setNewGradient(const dotk::vector<Real> & input_)
{
    m_NewGradient->copy(input_);
}

void DOTk_TrustRegionMngTypeELP::setOldGradient(const dotk::vector<Real> & input_)
{
    m_OldGradient->copy(input_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getNewGradient() const
{
    return (m_NewGradient);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_TrustRegionMngTypeELP::getOldGradient() const
{
    return (m_OldGradient);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getDeltaDual() const
{
    return (m_DeltaDual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getLinearizedEqConstraint() const
{
    return (m_LinearizedEqConstraint);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getJacobianTimesTangentialStep() const
{
    return (m_JacobianTimesTangentialStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getOldEqualityConstraintResidual() const
{
    return (m_OldEqualityConstraint);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getNewEqualityConstraintResidual() const
{
    return (m_NewEqualityConstraint);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getNormalStep() const
{
    return (m_NormalStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getDeltaPrimal() const
{
    return (m_DeltaPrimal);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getTangentialStep() const
{
    return (m_TangentialStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getNormalCauchyStep() const
{
    return (m_NormalCauchyStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getProjectedGradient() const
{
    return (m_ProjectedGradient);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getHessTimesNormalStep() const
{
    return (m_HessTimesNormalStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getTangentialStepResidual() const
{
    return (m_TangentialStepResidual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getProjTangentialCauchyStep() const
{
    return (m_ProjectedTangentialCauchyStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getAugmentedSystemLeftHandSide() const
{
    return (m_AugmentedSystemLeftHandSide);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getAugmentedSystemRightHandSide() const
{
    return (m_AugmentedSystemRightHandSide);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getProjectedTangentialStep() const
{
    return (m_ProjectedTangentialStep);
}

const std::tr1::shared_ptr<dotk::vector<Real> > &
DOTk_TrustRegionMngTypeELP::getTangentialSubProblemRhs() const
{
    return (m_TangentialSubProblemRhs);
}

size_t DOTk_TrustRegionMngTypeELP::getObjectiveFunctionEvaluationCounter() const
{
    return (m_RoutinesMng->getObjectiveFunctionEvaluationCounter());
}

void DOTk_TrustRegionMngTypeELP::setTrustRegionRadius(Real radius_)
{
    m_TrustRegion->setTrustRegionRadius(radius_);
}

Real DOTk_TrustRegionMngTypeELP::getTrustRegionRadius() const
{
    Real trust_region_radius = m_TrustRegion->getTrustRegionRadius();
    return (trust_region_radius);
}

void DOTk_TrustRegionMngTypeELP::setMinTrustRegionRadius(Real radius_)
{
    m_TrustRegion->setMinTrustRegionRadius(radius_);
}

Real DOTk_TrustRegionMngTypeELP::getMinTrustRegionRadius() const
{
    Real min_trust_region_radius = m_TrustRegion->getMinTrustRegionRadius();
    return (min_trust_region_radius);
}

void DOTk_TrustRegionMngTypeELP::setMaxTrustRegionRadius(Real radius_)
{
    m_TrustRegion->setMaxTrustRegionRadius(radius_);
}

Real DOTk_TrustRegionMngTypeELP::getMaxTrustRegionRadius() const
{
    Real max_trust_region_radius = m_TrustRegion->getMaxTrustRegionRadius();
    return (max_trust_region_radius);
}

void DOTk_TrustRegionMngTypeELP::setTrustRegionExpansionParameter(Real parameter_)
{
    m_TrustRegion->setExpansionParameter(parameter_);
}

Real DOTk_TrustRegionMngTypeELP::getTrustRegionExpansionParameter() const
{
    Real trust_region_expansion_parameter = m_TrustRegion->getExpansionParameter();
    return (trust_region_expansion_parameter);
}

void DOTk_TrustRegionMngTypeELP::setTrustRegionContractionParameter(Real parameter_)
{
    m_TrustRegion->setContractionParameter(parameter_);
}

Real DOTk_TrustRegionMngTypeELP::getTrustRegionContractionParameter() const
{
    Real trust_region_contraction_parameter = m_TrustRegion->getContractionParameter();
    return (trust_region_contraction_parameter);
}

void DOTk_TrustRegionMngTypeELP::setMinActualOverPredictedReductionAllowed(Real parameter_)
{
    m_TrustRegion->setMinActualOverPredictedReductionAllowed(parameter_);
}

Real DOTk_TrustRegionMngTypeELP::getMinActualOverPredictedReductionAllowed() const
{
    Real minimun_actual_over_predicted_reduction_allowed =
            m_TrustRegion->getMinActualOverPredictedReductionAllowed();
    return (minimun_actual_over_predicted_reduction_allowed);
}

void DOTk_TrustRegionMngTypeELP::invalidCurvatureDetected(bool invalid_curvature_detected_)
{
    bool flag = invalid_curvature_detected_;
    m_TrustRegion->invalidCurvatureDetected(flag);
}

const std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & DOTk_TrustRegionMngTypeELP::getTrustRegion() const
{
    return (m_TrustRegion);
}

const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & DOTk_TrustRegionMngTypeELP::getRoutinesMng() const
{
    return (m_RoutinesMng);
}

Real DOTk_TrustRegionMngTypeELP::computeDoglegRoot(const Real & trust_region_radius_,
                                                   const std::tr1::shared_ptr<dotk::vector<Real> > & x_,
                                                   const std::tr1::shared_ptr<dotk::vector<Real> > & y_)
{
    Real root = m_TrustRegion->computeDoglegRoot(trust_region_radius_, x_, y_);
    return (root);
}

void DOTk_TrustRegionMngTypeELP::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                            const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                            const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_)
{
    m_TrialStep->fill(0);
    m_OldPrimal->fill(0);
    m_OldGradient->fill(0);
    m_NewGradient->fill(0);
    m_MatrixTimesVector->fill(0);

    m_NormalStep->fill(0);
    m_DeltaPrimal->fill(0);
    m_TangentialStep->fill(0);
    m_NormalCauchyStep->fill(0);
    m_ProjectedGradient->fill(0);
    m_HessTimesNormalStep->fill(0);
    m_ProjectedTangentialStep->fill(0);
    m_TangentialSubProblemRhs->fill(0);
    m_ProjectedTangentialCauchyStep->fill(0);

    m_TangentialStepResidual->fill(0);
    m_AugmentedSystemLeftHandSide->fill(0);
    m_AugmentedSystemRightHandSide->fill(0);

    m_NewDual->copy(*primal_->dual());
    m_OldDual->copy(*primal_->dual());

    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeELP(primal_, objective_, equality_));
}

}