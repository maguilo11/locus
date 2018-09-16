/*
 * DOTk_TrustRegionStepMng.cpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <algorithm>
#include "DOTk_TrustRegionStepMng.hpp"

namespace dotk
{

DOTk_TrustRegionStepMng::DOTk_TrustRegionStepMng() :
        m_ActualReduction(0),
        m_TrustRegionRadius(1e4),
        m_PredictedReduction(0),
        m_MinTrustRegionRadius(1e-4),
        m_MaxTrustRegionRadius(1e4),
        m_TrustRegionReduction(0.5),
        m_TrustRegionExpansion(2.),
        m_MinCosineAngleTolerance(1e-2),
        m_TrustRegionRadiusScaling(1.),
        m_ActualOverPredictedReduction(0),
        m_ActualOverPredictedReductionMidBound(0.25),
        m_ActualOverPredictedReductionLowerBound(0.1),
        m_ActualOverPredictedReductionUpperBound(0.75),
        m_AdaptiveGradientInexactnessConstant(1),
        m_AdaptiveGradientInexactnessTolerance(0),
        m_AdaptiveObjectiveInexactnessConstant(1),
        m_AdaptiveObjectiveInexactnessTolerance(0),
        m_InitialTrustRegionSetToGradNorm(true),
        m_NumTrustRegionSubProblemItrDone(0),
        m_MaxNumTrustRegionSubProblemItr(30)
{
}

DOTk_TrustRegionStepMng::~DOTk_TrustRegionStepMng()
{
}

void DOTk_TrustRegionStepMng::setTrustRegionRadius(Real input_)
{
    m_TrustRegionRadius = input_;
}

Real DOTk_TrustRegionStepMng::getTrustRegionRadius() const
{
    return (m_TrustRegionRadius);
}

void DOTk_TrustRegionStepMng::setTrustRegionReduction(Real input_)
{
    m_TrustRegionReduction = input_;
}

Real DOTk_TrustRegionStepMng::getTrustRegionReduction() const
{
    return (m_TrustRegionReduction);
}

void DOTk_TrustRegionStepMng::setTrustRegionExpansion(Real input_)
{
    m_TrustRegionExpansion = input_;
}

Real DOTk_TrustRegionStepMng::getTrustRegionExpansion() const
{
    return (m_TrustRegionExpansion);
}

void DOTk_TrustRegionStepMng::setMinTrustRegionRadius(Real input_)
{
    m_MinTrustRegionRadius = input_;
}

Real DOTk_TrustRegionStepMng::getMinTrustRegionRadius() const
{
    return (m_MinTrustRegionRadius);
}

void DOTk_TrustRegionStepMng::setMaxTrustRegionRadius(Real input_)
{
    m_MaxTrustRegionRadius = input_;
}

Real DOTk_TrustRegionStepMng::getMaxTrustRegionRadius() const
{
    return (m_MaxTrustRegionRadius);
}

void DOTk_TrustRegionStepMng::setAdaptiveGradientInexactnessConstant(Real input_)
{
    m_AdaptiveGradientInexactnessConstant = input_;
}

Real DOTk_TrustRegionStepMng::getAdaptiveGradientInexactnessConstant() const
{
    return (m_AdaptiveGradientInexactnessConstant);
}

void DOTk_TrustRegionStepMng::updateAdaptiveGradientInexactnessTolerance(Real norm_current_gradient_)
{
    m_AdaptiveGradientInexactnessTolerance = this->getAdaptiveGradientInexactnessConstant()
            * std::min(this->getTrustRegionRadius(), norm_current_gradient_);
}

Real DOTk_TrustRegionStepMng::getAdaptiveGradientInexactnessTolerance() const
{
    return (m_AdaptiveGradientInexactnessTolerance);
}

void DOTk_TrustRegionStepMng::setAdaptiveObjectiveInexactnessConstant(Real input_)
{
    m_AdaptiveObjectiveInexactnessConstant = input_;
}

Real DOTk_TrustRegionStepMng::getAdaptiveObjectiveInexactnessConstant() const
{
    return (m_AdaptiveObjectiveInexactnessConstant);
}

void DOTk_TrustRegionStepMng::updateAdaptiveObjectiveInexactnessTolerance()
{
    m_AdaptiveObjectiveInexactnessTolerance = this->getAdaptiveObjectiveInexactnessConstant()
            * m_ActualOverPredictedReductionLowerBound * std::abs(m_PredictedReduction);
}

Real DOTk_TrustRegionStepMng::getAdaptiveObjectiveInexactnessTolerance() const
{
    return (m_AdaptiveObjectiveInexactnessTolerance);
}

void DOTk_TrustRegionStepMng::setActualOverPredictedReductionMidBound(Real input_)
{
    m_ActualOverPredictedReductionMidBound = input_;
}

Real DOTk_TrustRegionStepMng::getActualOverPredictedReductionMidBound() const
{
    return (m_ActualOverPredictedReductionMidBound);
}

void DOTk_TrustRegionStepMng::setActualOverPredictedReductionLowerBound(Real input_)
{
    m_ActualOverPredictedReductionLowerBound = input_;
}

Real DOTk_TrustRegionStepMng::getActualOverPredictedReductionLowerBound() const
{
    return (m_ActualOverPredictedReductionLowerBound);
}

void DOTk_TrustRegionStepMng::setActualOverPredictedReductionUpperBound(Real input_)
{
    m_ActualOverPredictedReductionUpperBound = input_;
}

Real DOTk_TrustRegionStepMng::getActualOverPredictedReductionUpperBound() const
{
    return (m_ActualOverPredictedReductionUpperBound);
}

void DOTk_TrustRegionStepMng::setNumTrustRegionSubProblemItrDone(size_t input_)
{
    m_NumTrustRegionSubProblemItrDone = input_;
}

void DOTk_TrustRegionStepMng::updateNumTrustRegionSubProblemItrDone()
{
    m_NumTrustRegionSubProblemItrDone++;
}

size_t DOTk_TrustRegionStepMng::getNumTrustRegionSubProblemItrDone() const
{
    return (m_NumTrustRegionSubProblemItrDone);
}

void DOTk_TrustRegionStepMng::setMaxNumTrustRegionSubProblemItr(size_t input_)
{
    m_MaxNumTrustRegionSubProblemItr = input_;
}

size_t DOTk_TrustRegionStepMng::getMaxNumTrustRegionSubProblemItr() const
{
    return (m_MaxNumTrustRegionSubProblemItr);
}

void DOTk_TrustRegionStepMng::setActualReduction(Real input_)
{
    m_ActualReduction = input_;
}

Real DOTk_TrustRegionStepMng::getActualReduction() const
{
    return (m_ActualReduction);
}

void DOTk_TrustRegionStepMng::setPredictedReduction(Real input_)
{
    m_PredictedReduction = input_;
}

Real DOTk_TrustRegionStepMng::getPredictedReduction() const
{
    return (m_PredictedReduction);
}

void DOTk_TrustRegionStepMng::setMinCosineAngleTolerance(Real tol_)
{
    m_MinCosineAngleTolerance = tol_;
}

Real DOTk_TrustRegionStepMng::getMinCosineAngleTolerance() const
{
    return (m_MinCosineAngleTolerance);
}

void DOTk_TrustRegionStepMng::setTrustRegionRadiusScaling(Real input_)
{
    m_TrustRegionRadiusScaling = input_;
}

Real DOTk_TrustRegionStepMng::getTrustRegionRadiusScaling() const
{
    return (m_TrustRegionRadiusScaling);
}

void DOTk_TrustRegionStepMng::setActualOverPredictedReduction(Real input_)
{
    m_ActualOverPredictedReduction = input_;
}

Real DOTk_TrustRegionStepMng::getActualOverPredictedReduction() const
{
    return (m_ActualOverPredictedReduction);
}

void DOTk_TrustRegionStepMng::setInitialTrustRegionRadiusToGradNorm(bool input_)
{
    m_InitialTrustRegionSetToGradNorm = input_;
}

bool DOTk_TrustRegionStepMng::isInitialTrustRegionRadiusSetToGradNorm() const
{
    return (m_InitialTrustRegionSetToGradNorm);
}

bool DOTk_TrustRegionStepMng::updateTrustRegionRadius()
{
    {
        Real trust_region_reduction = this->getTrustRegionReduction();
        Real trust_region_expansion = this->getTrustRegionExpansion();
        Real current_trust_region_radius = this->getTrustRegionRadius();
        Real trust_region_lower_bound = this->getActualOverPredictedReductionLowerBound();
        Real trust_region_upper_bound = this->getActualOverPredictedReductionUpperBound();
        Real current_actual_over_pred_reduction = this->getActualOverPredictedReduction();

        bool stop_trust_region_sub_problem = false;
        size_t max_num_itr = this->getMaxNumTrustRegionSubProblemItr();
        if(current_actual_over_pred_reduction >= trust_region_upper_bound)
        {
            Real max_trust_region_radius = this->getMaxTrustRegionRadius();
            current_trust_region_radius = trust_region_expansion * current_trust_region_radius;
            current_trust_region_radius = std::min(max_trust_region_radius, current_trust_region_radius);
            stop_trust_region_sub_problem = true;
        }
        else if(current_actual_over_pred_reduction >= trust_region_lower_bound)
        {
            stop_trust_region_sub_problem = true;
        }
        else if(this->getNumTrustRegionSubProblemItrDone() > max_num_itr)
        {
            current_trust_region_radius = trust_region_reduction * current_trust_region_radius;
            stop_trust_region_sub_problem = true;
        }
        else
        {
            current_trust_region_radius = trust_region_reduction * current_trust_region_radius;
        }
        this->setTrustRegionRadius(current_trust_region_radius);

        return (stop_trust_region_sub_problem);
    }
}

}
