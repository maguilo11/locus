/*
 * TRROM_TrustRegionStepMng.cpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_TrustRegionStepMng.hpp"

namespace trrom
{

TrustRegionStepMng::TrustRegionStepMng() :
        m_ActualReduction(0),
        m_TrustRegionRadius(1e4),
        m_PredictedReduction(0),
        m_MinTrustRegionRadius(1e-4),
        m_MaxTrustRegionRadius(1e4),
        m_TrustRegionExpansion(2.),
        m_TrustRegionContraction(0.5),
        m_MinCosineAngleTolerance(1e-2),
        m_TrustRegionRadiusScaling(1.),
        m_GradientInexactnessTolerance(0),
        m_ObjectiveInexactnessTolerance(0),
        m_ActualOverPredictedReduction(0),
        m_ActualOverPredictedReductionMidBound(0.25),
        m_ActualOverPredictedReductionLowerBound(0.1),
        m_ActualOverPredictedReductionUpperBound(0.75),
        m_GradientInexactnessToleranceConstant(1),
        m_ObjectiveInexactnessToleranceConstant(1),
        m_InitialTrustRegionSetToGradNorm(true),
        m_NumTrustRegionSubProblemItrDone(0),
        m_MaxNumTrustRegionSubProblemItr(30)
{
}

TrustRegionStepMng::~TrustRegionStepMng()
{
}

void TrustRegionStepMng::setTrustRegionRadius(double input_)
{
    m_TrustRegionRadius = input_;
}

double TrustRegionStepMng::getTrustRegionRadius() const
{
    return (m_TrustRegionRadius);
}

void TrustRegionStepMng::setTrustRegionContraction(double input_)
{
    m_TrustRegionContraction = input_;
}

double TrustRegionStepMng::getTrustRegionContraction() const
{
    return (m_TrustRegionContraction);
}

void TrustRegionStepMng::setTrustRegionExpansion(double input_)
{
    m_TrustRegionExpansion = input_;
}

double TrustRegionStepMng::getTrustRegionExpansion() const
{
    return (m_TrustRegionExpansion);
}

void TrustRegionStepMng::setMinTrustRegionRadius(double input_)
{
    m_MinTrustRegionRadius = input_;
}

double TrustRegionStepMng::getMinTrustRegionRadius() const
{
    return (m_MinTrustRegionRadius);
}

void TrustRegionStepMng::setMaxTrustRegionRadius(double input_)
{
    m_MaxTrustRegionRadius = input_;
}

double TrustRegionStepMng::getMaxTrustRegionRadius() const
{
    return (m_MaxTrustRegionRadius);
}

void TrustRegionStepMng::setGradientInexactnessToleranceConstant(double input_)
{
    m_GradientInexactnessToleranceConstant = input_;
}

double TrustRegionStepMng::getGradientInexactnessToleranceConstant() const
{
    return (m_GradientInexactnessToleranceConstant);
}

void TrustRegionStepMng::updateGradientInexactnessTolerance(double norm_gradient_)
{
    double min_value = std::min(this->getTrustRegionRadius(), norm_gradient_);
    m_GradientInexactnessTolerance = m_GradientInexactnessToleranceConstant * min_value;
}

double TrustRegionStepMng::getGradientInexactnessTolerance() const
{
    return (m_GradientInexactnessTolerance);
}

void TrustRegionStepMng::setObjectiveInexactnessToleranceConstant(double input_)
{
    m_ObjectiveInexactnessToleranceConstant = input_;
}

double TrustRegionStepMng::getObjectiveInexactnessToleranceConstant() const
{
    return (m_ObjectiveInexactnessToleranceConstant);
}

void TrustRegionStepMng::updateObjectiveInexactnessTolerance(double predicted_reduction_)
{
    m_ObjectiveInexactnessTolerance = m_ObjectiveInexactnessToleranceConstant * m_ActualOverPredictedReductionLowerBound
            * std::abs(predicted_reduction_);
}

double TrustRegionStepMng::getObjectiveInexactnessTolerance() const
{
    return (m_ObjectiveInexactnessTolerance);
}

void TrustRegionStepMng::setActualOverPredictedReductionMidBound(double input_)
{
    m_ActualOverPredictedReductionMidBound = input_;
}

double TrustRegionStepMng::getActualOverPredictedReductionMidBound() const
{
    return (m_ActualOverPredictedReductionMidBound);
}

void TrustRegionStepMng::setActualOverPredictedReductionLowerBound(double input_)
{
    m_ActualOverPredictedReductionLowerBound = input_;
}

double TrustRegionStepMng::getActualOverPredictedReductionLowerBound() const
{
    return (m_ActualOverPredictedReductionLowerBound);
}

void TrustRegionStepMng::setActualOverPredictedReductionUpperBound(double input_)
{
    m_ActualOverPredictedReductionUpperBound = input_;
}

double TrustRegionStepMng::getActualOverPredictedReductionUpperBound() const
{
    return (m_ActualOverPredictedReductionUpperBound);
}

void TrustRegionStepMng::setNumTrustRegionSubProblemItrDone(int input_)
{
    m_NumTrustRegionSubProblemItrDone = input_;
}

void TrustRegionStepMng::updateNumTrustRegionSubProblemItrDone()
{
    m_NumTrustRegionSubProblemItrDone++;
}

int TrustRegionStepMng::getNumTrustRegionSubProblemItrDone() const
{
    return (m_NumTrustRegionSubProblemItrDone);
}

void TrustRegionStepMng::setMaxNumTrustRegionSubProblemItr(int input_)
{
    m_MaxNumTrustRegionSubProblemItr = input_;
}

int TrustRegionStepMng::getMaxNumTrustRegionSubProblemItr() const
{
    return (m_MaxNumTrustRegionSubProblemItr);
}

void TrustRegionStepMng::setActualReduction(double input_)
{
    m_ActualReduction = input_;
}

double TrustRegionStepMng::getActualReduction() const
{
    return (m_ActualReduction);
}

void TrustRegionStepMng::setPredictedReduction(double input_)
{
    m_PredictedReduction = input_;
}

double TrustRegionStepMng::getPredictedReduction() const
{
    return (m_PredictedReduction);
}

void TrustRegionStepMng::setMinCosineAngleTolerance(double tol_)
{
    m_MinCosineAngleTolerance = tol_;
}

double TrustRegionStepMng::getMinCosineAngleTolerance() const
{
    return (m_MinCosineAngleTolerance);
}

void TrustRegionStepMng::setTrustRegionRadiusScaling(double input_)
{
    m_TrustRegionRadiusScaling = input_;
}

double TrustRegionStepMng::getTrustRegionRadiusScaling() const
{
    return (m_TrustRegionRadiusScaling);
}

void TrustRegionStepMng::setActualOverPredictedReduction(double input_)
{
    m_ActualOverPredictedReduction = input_;
}

double TrustRegionStepMng::getActualOverPredictedReduction() const
{
    return (m_ActualOverPredictedReduction);
}

void TrustRegionStepMng::setInitialTrustRegionRadiusToGradNorm(bool input_)
{
    m_InitialTrustRegionSetToGradNorm = input_;
}

bool TrustRegionStepMng::isInitialTrustRegionRadiusSetToGradNorm() const
{
    return (m_InitialTrustRegionSetToGradNorm);
}

}
