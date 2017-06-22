/*
 * DOTk_TrustRegionAlgorithmsDataMng.cpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_TrustRegionFactory.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_TrustRegionAlgorithmsDataMng::DOTk_TrustRegionAlgorithmsDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_OptimizationDataMng(primal_),
        m_RoutinesMng(),
        m_FirstOrderOperator(),
        m_TrustRegion()
{
    this->setUserDefinedGradient();
    dotk::DOTk_TrustRegionFactory trust_region_factory(dotk::types::TRUST_REGION_DOGLEG);
    trust_region_factory.build(dotk::DOTk_OptimizationDataMng::getTrialStep(), m_TrustRegion);
}

DOTk_TrustRegionAlgorithmsDataMng::~DOTk_TrustRegionAlgorithmsDataMng()
{
}

size_t DOTk_TrustRegionAlgorithmsDataMng::getObjectiveFuncEvalCounter() const
{
    return (m_RoutinesMng->getObjectiveFunctionEvaluationCounter());
}

size_t DOTk_TrustRegionAlgorithmsDataMng::getAdjointInverseJacobianWrtStateCounter() const
{
    return (m_RoutinesMng->getAdjointInverseJacobianStateCounter());
}

size_t DOTk_TrustRegionAlgorithmsDataMng::getHessianEvaluationCounter() const
{
    return (m_RoutinesMng->getHessianEvaluationCounter());
}

size_t DOTk_TrustRegionAlgorithmsDataMng::getGradientEvaluationCounter() const
{
    return (m_RoutinesMng->getGradientEvaluationCounter());
}

size_t DOTk_TrustRegionAlgorithmsDataMng::getEqualityConstraintEvaluationCounter() const
{
    return (m_RoutinesMng->getEqualityConstraintEvaluationCounter());
}

void DOTk_TrustRegionAlgorithmsDataMng::setTrustRegionRadius(Real radius_)
{
    m_TrustRegion->setTrustRegionRadius(radius_);
}

Real DOTk_TrustRegionAlgorithmsDataMng::getTrustRegionRadius() const
{
    Real trust_region_radius = m_TrustRegion->getTrustRegionRadius();
    return (trust_region_radius);
}

void DOTk_TrustRegionAlgorithmsDataMng::setMinTrustRegionRadius(Real radius_)
{
    m_TrustRegion->setMinTrustRegionRadius(radius_);
}

Real DOTk_TrustRegionAlgorithmsDataMng::getMinTrustRegionRadius() const
{
    Real min_trust_region_radius = m_TrustRegion->getMinTrustRegionRadius();
    return (min_trust_region_radius);
}

void DOTk_TrustRegionAlgorithmsDataMng::setMaxTrustRegionRadius(Real radius_)
{
    m_TrustRegion->setMaxTrustRegionRadius(radius_);
}

Real DOTk_TrustRegionAlgorithmsDataMng::getMaxTrustRegionRadius() const
{
    Real max_trust_region_radius = m_TrustRegion->getMaxTrustRegionRadius();
    return (max_trust_region_radius);
}

void DOTk_TrustRegionAlgorithmsDataMng::setTrustRegionExpansionParameter(Real parameter_)
{
    m_TrustRegion->setExpansionParameter(parameter_);
}

Real DOTk_TrustRegionAlgorithmsDataMng::getTrustRegionExpansionParameter() const
{
    Real trust_region_expansion_parameter = m_TrustRegion->getExpansionParameter();
    return (trust_region_expansion_parameter);
}

void DOTk_TrustRegionAlgorithmsDataMng::setTrustRegionContractionParameter(Real parameter_)
{
    m_TrustRegion->setContractionParameter(parameter_);
}

Real DOTk_TrustRegionAlgorithmsDataMng::getTrustRegionContractionParameter() const
{
    Real trust_region_contraction_parameter = m_TrustRegion->getContractionParameter();
    return (trust_region_contraction_parameter);
}

void DOTk_TrustRegionAlgorithmsDataMng::setMinActualOverPredictedReductionAllowed(Real parameter_)
{
    m_TrustRegion->setMinActualOverPredictedReductionAllowed(parameter_);
}

Real DOTk_TrustRegionAlgorithmsDataMng::getMinActualOverPredictedReductionAllowed() const
{
    Real minimun_actual_over_predicted_reduction_allowed =
            m_TrustRegion->getMinActualOverPredictedReductionAllowed();
    return (minimun_actual_over_predicted_reduction_allowed);
}

void DOTk_TrustRegionAlgorithmsDataMng::setMaxTrustRegionSubProblemIterations(size_t itr_)
{
    m_TrustRegion->setMaxTrustRegionSubProblemIterations(itr_);
}

size_t DOTk_TrustRegionAlgorithmsDataMng::getMaxTrustRegionSubProblemIterations() const
{
    size_t itr = m_TrustRegion->getMaxTrustRegionSubProblemIterations();
    return (itr);
}

void DOTk_TrustRegionAlgorithmsDataMng::invalidCurvatureDetected(bool invalid_curvature_detected_)
{
    bool flag = invalid_curvature_detected_;
    m_TrustRegion->invalidCurvatureDetected(flag);
}

const std::shared_ptr<dotk::DOTk_TrustRegion> & DOTk_TrustRegionAlgorithmsDataMng::getTrustRegion() const
{
    return (m_TrustRegion);
}

Real DOTk_TrustRegionAlgorithmsDataMng::computeDoglegRoot
(Real trust_region_radius_,
 const std::shared_ptr<dotk::Vector<Real> > & vector1_,
 const std::shared_ptr<dotk::Vector<Real> > & vector2_)
{
    Real root = m_TrustRegion->computeDoglegRoot(trust_region_radius_, vector1_, vector2_);
    return (root);
}

void DOTk_TrustRegionAlgorithmsDataMng::setCauchyTrustRegionMethod(Real trust_region_radius_)
{
    dotk::DOTk_TrustRegionFactory factory;
    factory.buildCauchyTrustRegion(m_TrustRegion);
    m_TrustRegion->setTrustRegionRadius(trust_region_radius_);
}

void DOTk_TrustRegionAlgorithmsDataMng::setDoglegTrustRegionMethod(Real trust_region_radius_)
{
    dotk::DOTk_TrustRegionFactory factory;
    factory.buildDoglegTrustRegion(m_TrustRegion);
    m_TrustRegion->setTrustRegionRadius(trust_region_radius_);
}

void DOTk_TrustRegionAlgorithmsDataMng::setDoubleDoglegTrustRegionMethod
(const std::shared_ptr<dotk::Vector<Real> > & vector_,
 Real trust_region_radius_)
{
    dotk::DOTk_TrustRegionFactory factory;
    factory.buildDoubleDoglegTrustRegion(dotk::DOTk_OptimizationDataMng::getTrialStep(), m_TrustRegion);
    m_TrustRegion->setTrustRegionRadius(trust_region_radius_);
}

void DOTk_TrustRegionAlgorithmsDataMng::updateState(const Real new_objective_function_value_,
                                                    const std::shared_ptr<dotk::Vector<Real> > & new_primal_)
{
    Real old_objective_func_value = dotk::DOTk_OptimizationDataMng::getNewObjectiveFunctionValue();
    dotk::DOTk_OptimizationDataMng::setOldObjectiveFunctionValue(old_objective_func_value);

    this->getOldPrimal()->update(1., *this->getNewPrimal(), 0.);
    this->getOldGradient()->update(1., *this->getNewGradient(), 0.);
    this->getNewPrimal()->update(1., *new_primal_, 0.);

    dotk::DOTk_OptimizationDataMng::setNewObjectiveFunctionValue(new_objective_function_value_);
    this->computeGradient();

    Real norm_new_gradient = this->getNewGradient()->norm();
    this->setNormNewGradient(norm_new_gradient);

    Real norm_trial_step = this->getTrialStep()->norm();
    this->setNormTrialStep(norm_trial_step);
}

void DOTk_TrustRegionAlgorithmsDataMng::computeScaledInexactNewtonStep(const bool invalid_curvature_detected_,
                                                                       const std::shared_ptr<dotk::Vector<Real> > & descent_direction_)
{
    m_TrustRegion->invalidCurvatureDetected(invalid_curvature_detected_);
    m_TrustRegion->step(this, descent_direction_, dotk::DOTk_OptimizationDataMng::getTrialStep());
}

void DOTk_TrustRegionAlgorithmsDataMng::setUserDefinedGradient()
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildUserDefinedGradient(m_FirstOrderOperator);
}

void DOTk_TrustRegionAlgorithmsDataMng::checkTrustRegionPtr(std::ostringstream & msg_)
{
    msg_.str(std::string());
    if(m_TrustRegion.use_count() == 0)
    {
        this->setDoglegTrustRegionMethod();
        msg_ << "DOTk WARNING: Trust region method was not specified by the user. Trust region method will be set to DOGLEG.\n";
        dotk::ioUtils::printMessage(msg_);
    }
}

Real DOTk_TrustRegionAlgorithmsDataMng::evaluateObjective()
{
    Real value = m_RoutinesMng->objective(this->getNewPrimal());
    return (value);
}

Real DOTk_TrustRegionAlgorithmsDataMng::evaluateObjective(const std::shared_ptr<dotk::Vector<Real> > & input_)
{
    Real value = m_RoutinesMng->objective(input_);
    return (value);
}

void DOTk_TrustRegionAlgorithmsDataMng::computeGradient()
{
    m_FirstOrderOperator->gradient(this);
}

void DOTk_TrustRegionAlgorithmsDataMng::computeGradient(const std::shared_ptr<dotk::Vector<Real> > & input_,
                                                        const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    m_RoutinesMng->gradient(input_, gradient_);
}

const std::shared_ptr<dotk::DOTk_AssemblyManager> & DOTk_TrustRegionAlgorithmsDataMng::getRoutinesMng() const
{
    return (m_RoutinesMng);
}

}
